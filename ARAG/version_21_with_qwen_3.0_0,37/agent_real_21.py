import re
import os
import torch
import json
from typing import Optional, List

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

import tools_real

from tools_real import (
    reset_query_state, set_current_question,
    get_logged_urls, get_read_urls,
)

from load_config import load_config
from Qwen_retrieval_3 import generate_answer
from eval_utils import (
    build_common_record, elapsed, now_seconds,
    parse_retrieved_urls, retrieval_metrics, token_estimate,
)

# =====================================================================
# 🛡️ MONKEY PATCH: CHIRURGIA SINTATTICA AUTO-AGGANCIANTE PER QWEN 3
# =====================================================================
import re
from langchain_core.agents import AgentFinish

# 1. Scanner automatico per scovare dov'è finito il Parser di LangChain
_ReActParserClass = None
for _path in [
    "langchain.agents.output_parsers",
    "langchain.agents.output_parsers.react_single_input",
    "langchain.agents.react.output_parser",
    "langchain_classic.agents.output_parsers",
    "langchain_classic.agents.output_parsers.react_single_input"
]:
    try:
        _mod = __import__(_path, fromlist=["ReActSingleInputOutputParser"])
        _ReActParserClass = getattr(_mod, "ReActSingleInputOutputParser")
        print(f"🔧 [MonkeyPatch] Parser di LangChain agganciato con successo da: {_path}")
        break
    except (ImportError, AttributeError):
        continue

if _ReActParserClass is None:
    raise ImportError("Errore critico: impossibile trovare ReActSingleInputOutputParser nella tua versione di LangChain.")

_vecchio_parser = _ReActParserClass.parse

def _parser_blindato(self, text: str):
    
    # Cura il Bug: se Qwen ha usato "Action: Final Answer" come tool
    if "Action: Final Answer" in text_pulito or "Action: final_answer" in text_pulito:
        risposta = text_pulito.split("Action Input:")[-1].strip()
        return AgentFinish({"output": risposta}, text_pulito)
        
    # Elimina gli "spazi fantasma" a fine tool (es. "tool_cerca_wikipedia ")
    text_pulito = re.sub(r'(Action:\s*tool_[a-z_]+)[ \t]+', r'\1\n', text_pulito)
    
    return _vecchio_parser(self, text_pulito)

_ReActParserClass.parse = _parser_blindato
# =====================================================================

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config_dict = load_config()
print(f"[config] Qwen model path: {config_dict['model_path']}")
class Args: pass
args = Args()
for k, v in config_dict.items():
    setattr(args, k, str(v))
args.top_k = 5

tools_real.start_motors(args)

# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------
class QwenServerLLM(LLM):
    current_image_path: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "qwen2.5-vl-custom-multimodal"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        user_content = []

        match = re.search(r"\[IMG\](.*?)\[/IMG\]", prompt)
        if match:
            image_path = match.group(1).strip()
            if os.path.exists(image_path):
                user_content.append({"type": "image", "image": image_path})
            prompt = re.sub(r"\[IMG\].*?\[/IMG\]\n?", "", prompt)
        elif self.current_image_path and os.path.exists(self.current_image_path):
            user_content.append({"type": "image", "image": self.current_image_path})

        wiki_images = re.findall(r"\[IMG_WIKI:\s*(.*?)\]", prompt)
        for img_path in set(wiki_images):
            if os.path.exists(img_path):
                user_content.append({"type": "image", "image": img_path})

        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": (
                "You are a precise AI backend. Use ONLY the ReAct format: "
                "Thought/Action/Action Input. Never add conversational text."
            )},
            {"role": "user", "content": user_content},
        ]

        if tools_real.qwen_model is None:
            raise ValueError("Models not initialized.")

        # ALZATO DA 512 A 1536 PER EVITARE CHE SI SPEGNA MENTRE RAGIONA
        risposta = generate_answer(
            tools_real.qwen_model, tools_real.qwen_processor,
            messages, max_new_tokens=512, 
        )

        for sw in (stop or []) + ["Observation:", "\nObservation:", "Observation:\n"]:
            if sw in risposta:
                risposta = risposta.split(sw)[0]

        risposta_pulita = risposta.strip()
        st = tools_real._state()

        # [GUARDRAIL 1] Sveglia RAM Turno 1
        if getattr(st, "search_done_count", 0) == 0:
            if "Action: tool_cerca_wikipedia" not in risposta_pulita:
                print("🛡️ [GUARDRAIL LLM] Turno 1: Forzo tool_cerca_wikipedia...")
                img_p = self.current_image_path or ""
                return f"Thought: I must start by visually searching the entity in the database.\nAction: tool_cerca_wikipedia\nAction Input: {img_p}"

        # [GUARDRAIL 2] Scudo Salto-Lettura
        if "Final Answer:" in risposta_pulita and len(getattr(st, "read_urls", [])) == 0:
            cands = getattr(st, "available_urls", [])
            if cands:
                print("🛡️ [GUARDRAIL LLM] Tentativo di Final Answer senza leggere. Bloccato.")
                return f"Thought: I cannot give Final Answer yet because I haven't read any Wikipedia page.\nAction: tool_leggi_wikipedia\nAction Input: {cands[0]}"

        # NOTA: Rimosso lo Step 3 di Guardrail sulla ricerca per non sporcare lo scratchpad!

        return risposta_pulita

# PROMPT V4 PURIFICATO (Zero accenni a ri-chiamate o loop)
PROMPT_V4 = """You are a Visual AI Agent extracting facts from Wikipedia.

TOOLS AVAILABLE:
{tools}

MANDATORY WORKFLOW (Execute step-by-step):
STEP 1: Call 'tool_cerca_wikipedia' passing EXACTLY the Image Path provided in the Question.
STEP 2: Read the Observation carefully — it contains the TOP 5 Wikipedia matches ranked by an AI relevance score, each accompanied by a short text 'Preview'.
STEP 3: Evaluate the Previews against the Question and the visual identification. Call 'tool_leggi_wikipedia' on the precise HTTP link whose Preview is most aligned with what the user is asking. (You are NOT forced to blindly pick URL [1] if another candidate's Preview clearly holds or hints at the answer!). (Format MUST be: URL: https://en.wikipedia.org/wiki/... || QUESTION: <question>).
STEP 4: If the read page contains the exact answer → Final Answer. If NOT → select the next most promising unread URL based on the Previews.

CRITICAL CONSTRAINTS:
1. NEVER output 'Final Answer' based only on the caption, previews, or your internal knowledge. You MUST verify facts by reading at least one full Wikipedia page via 'tool_leggi_wikipedia'.
2. NEVER INVENT OR GUESS WIKIPEDIA URLs. You are strictly forbidden from calling 'tool_leggi_wikipedia' with a URL that you invented. You MUST ONLY use the exact URLs provided in the 'tool_cerca_wikipedia' observation.
3. If the page you read does NOT contain the exact answer, immediately transition to the next most promising URL based on the Previews. Never guess.
4. NEVER call 'tool_cerca_wikipedia' more than once.
5. NEVER call 'tool_leggi_wikipedia' with the same URL twice.

STRICT FORMAT:
Question: the input question you must answer
Thought: [Identify current STEP. What do you need to do next?]
Action: the action to take, MUST be one of [{tool_names}]
Action Input: the precise input required by the action tool
Observation: the result of the action (provided by the system)
... (Thought/Action/Action Input/Observation can repeat)
Thought: I found the exact fact in the Wikipedia text.
Final Answer: the concise answer to the user's question

Begin!
Question: {input}
Thought: {agent_scratchpad}"""

prompt_template = PromptTemplate(
    template=PROMPT_V4,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
)

vero_qwen = QwenServerLLM()
agente    = create_react_agent(vero_qwen, tools_real.miei_tools_reali, prompt_template)
esecutore = AgentExecutor(
    agent=agente,
    tools=tools_real.miei_tools_reali,
    verbose=True,
    handle_parsing_errors=(
        "Format error. Use exactly:\nThought: ...\nAction: ...\nAction Input: ..."
    ),
    max_iterations=5,
    early_stopping_method="force",
    return_intermediate_steps=True,
)

def _fallback_answer(image_path: str, question: str, intermediate_steps: list) -> str:
    """Fallback d'emergenza lobotomizzato a risposta secca."""
    context_parts = []
    for action, observation in (intermediate_steps or []):
        if getattr(action, "tool", "") == "tool_leggi_wikipedia":
            obs_str = str(observation)
            if not obs_str.startswith(("ERROR:", "CRITICAL ERROR:", "⚠️")):
                context_parts.append(obs_str[:3000])

    if not context_parts:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": f"Answer this question in 1 to 4 words maximum. Give ONLY the raw answer: {question}"}
        ]}]
    else:
        context = "\n\n---\n\n".join(context_parts)[:8000]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": (
                f"Wikipedia context:\n{context}\n\n"
                f"Answer the question using 1 to 4 words maximum. "
                f"If asked a year or number, output ONLY the raw entity. Give no preamble.\n"
                f"Question: {question}"
            )}
        ]}]

    try:
        return generate_answer(tools_real.qwen_model, tools_real.qwen_processor, messages, max_new_tokens=30).strip()
    except Exception:
        return ""

def _clean_answer(answer: str) -> str:
    answer = answer.strip()
    for pat in [
        r"^(based on (the )?(retrieved|wikipedia|available) (context|information|text)[,.]?\s*)",
        r"^(according to (the )?(wikipedia|context|text)[,.]?\s*)",
        r"^(the answer is:?\s*)",
        r"^(therefore[,.]?\s*)",
        r"^(from the (wikipedia|context)[,.]?\s*)",
    ]:
        answer = re.sub(pat, "", answer, flags=re.IGNORECASE).strip()
    return answer

def _summarize_steps(steps, logged_urls: list, read_urls: list) -> dict:
    tool_calls = []
    observations = []
    thoughts = [] 
    
    for action, observation in (steps or []):
        raw_log = getattr(action, "log", "")
        thought_text = raw_log
        if "Action:" in raw_log:
            thought_text = raw_log.split("Action:")[0].strip()
            
        thoughts.append(thought_text)
        observations.append(str(observation))
        
        tool_calls.append({
            "tool": getattr(action, "tool", None),
            "tool_input": str(getattr(action, "tool_input", None)),
        })

    return {
        "tool_calls": tool_calls,
        "thoughts": thoughts,                 
        "observations": observations,         
        "candidate_urls": logged_urls,       
        "read_urls": read_urls,              
        "retrieved_urls": read_urls,         
        "num_tool_calls": len(tool_calls),
        "observations_tokens_est": token_estimate("\n".join(observations)),
    }

def run_agentic_rag(image_path, question, question_id=None,
                    ground_truth="", question_type="unknown",
                    expected_sources=None):

    reset_query_state()
    set_current_question(question)  
    vero_qwen.current_image_path = image_path

    start  = now_seconds()
    result = {}
    answer = ""
    error  = None

    domanda_per_agente = question
    if question_type == "multi_answer":
        domanda_per_agente += (
            " [SYSTEM NOTE: This question requires multiple items as an answer. "
            "You MUST extract and list ALL applicable items found in the text, separated by 'and' or commas.]"
        )

    try:
        comando = f"[IMG]{image_path}[/IMG]\nQuestion: {domanda_per_agente}\nImage Path: {image_path}\n"
        result = esecutore.invoke({"input": comando})
        answer = result.get("output", "")

        if not answer or "Agent stopped" in answer or "iteration limit" in answer:
            answer = _fallback_answer(image_path, question, result.get("intermediate_steps", []))

        answer = _clean_answer(answer)

    except Exception as e:
        error  = str(e)
        answer = _fallback_answer(image_path, question, result.get("intermediate_steps", []))

    logged_urls = get_logged_urls()
    read_urls   = get_read_urls()

    step_info   = _summarize_steps(result.get("intermediate_steps", []), logged_urls=logged_urls, read_urls=read_urls)
    candidate_urls = step_info.get("candidate_urls") or []
    read_urls_list = step_info.get("read_urls") or []

    ret_metrics = retrieval_metrics(read_urls_list, expected_sources or [], k=args.top_k)

    extra = {"num_steps": step_info["num_tool_calls"], **step_info, **ret_metrics}
    extra.setdefault("candidate_urls", candidate_urls)
    extra.setdefault("read_urls", read_urls_list)
    extra.setdefault("retrieved_urls", read_urls_list)

    return build_common_record(
        question_id=question_id,
        model_name="agentic_rag_v4",
        image_path=image_path,
        question=question, 
        ground_truth=ground_truth,
        answer=answer,
        question_type=question_type,
        latency_seconds=elapsed(start),
        error=error,
        extra=extra,
    )

if __name__ == "__main__":
    pass
