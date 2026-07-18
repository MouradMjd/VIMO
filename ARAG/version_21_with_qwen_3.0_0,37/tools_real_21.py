import threading
import torch
from PIL import Image
import ast
import re
import traceback
import os
from langchain_core.tools import tool
from Qwen_retrieval_3 import extract_features, load_clip_and_index, generate_answer

# ---------------------------------------------------------------------------
# Globali server
# ---------------------------------------------------------------------------
clip_model      = None
clip_processor  = None
knn_index_immagini = None
wiki_map        = None
wiki_data       = None
qwen_model      = None
qwen_processor  = None

# ---------------------------------------------------------------------------
# Stato per-query thread-safe
# ---------------------------------------------------------------------------
_tl = threading.local()

def _state():
    if not hasattr(_tl, "url_visitati"):       _tl.url_visitati       = []
    if not hasattr(_tl, "ultima_immagine"):    _tl.ultima_immagine    = ""
    if not hasattr(_tl, "current_question"):   _tl.current_question   = ""
    if not hasattr(_tl, "available_urls"):     _tl.available_urls     = []
    if not hasattr(_tl, "cached_caption"):     _tl.cached_caption     = ""
    if not hasattr(_tl, "logged_urls"):        _tl.logged_urls        = []   
    if not hasattr(_tl, "read_urls"):          _tl.read_urls          = []   
    if not hasattr(_tl, "search_done_count"):  _tl.search_done_count  = 0    
    if not hasattr(_tl, "last_search_signature"): _tl.last_search_signature = None
    return _tl

def reset_query_state():
    st = _state()
    st.url_visitati      = []
    st.ultima_immagine   = ""
    st.current_question  = ""
    st.available_urls    = []
    st.cached_caption    = ""
    st.logged_urls       = []
    st.read_urls         = []
    st.search_done_count = 0
    st.last_search_signature = None

def set_current_question(question: str):
    _state().current_question = question

def get_logged_urls():
    return list(getattr(_state(), "logged_urls", []))

def get_read_urls():
    return list(getattr(_state(), "read_urls", []))

def start_motors(args):
    global clip_model, clip_processor, knn_index_immagini
    global wiki_map, wiki_data, qwen_model, qwen_processor

    print("Accensione CLIP e FAISS...")
    clip_model, clip_processor, knn_index_immagini, wiki_map, wiki_data = load_clip_and_index(args)

    print("Accensione Qwen3-VL...")
    
    # 🚀 CHIAMIAMO IL LEGGITTIMO PROPRIETARIO DEI PESI: Qwen3VLForConditionalGeneration
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    
    qwen_processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa", # <-- Accende il Turbo su GPU Blackwell
        local_files_only=True, 
        trust_remote_code=True, 
        device_map="cuda:0",
    ).eval()
    print("✅ Motori accesi!")

STOPWORDS_RETRIEVAL = {
    "what","where","when","who","which","how","many","does",
    "this","that","the","and","for","are","was","were","did",
    "have","has","been","its","type","kind","species","plant",
    "bird","animal","insect","fish","tree","flower","building",
    "structure","image","picture","photo","shown","depicted",
}

def _keyword_score(url: str, page: dict, question: str) -> float:
    if not question: return 0.0
    title = page.get("title", "").lower()
    url_text = url.replace("_", " ").lower().split("/wiki/")[-1]
    q_words = set(re.findall(r"\b\w{3,}\b", question.lower())) - STOPWORDS_RETRIEVAL
    if not q_words: return 0.0
    matches = sum(1 for w in q_words if w in title or w in url_text)
    return matches / len(q_words)

# Core retrieval v4: Rete espansa a k=250 e Bonus Sano a +1.20
def _retrieve_candidates_v4(image_features, question: str, vlm_picked_title: str, caption: str, k_fetch: int=250, top_n: int=5) -> list:
    _, I = knn_index_immagini.search(image_features, k_fetch)
    raw = []
    for rank, idx in enumerate(I[0]):
        if idx >= len(wiki_map): continue
        url = wiki_map[idx][0]
        if url not in wiki_data: continue
        page = wiki_data[url]
        raw.append((url, page, rank))

    if not raw: return []
    n_valid = len(raw)
    picked_clean = vlm_picked_title.strip().lower()

    scored = []
    for url, page, rank in raw:
        title = page.get("title", "").lower()
        visual_score = 1.0 - (rank / max(n_valid - 1, 1))

        # Multiplicatore sano se Qwen ha scelto questo titolo dal menu MCQ
        vlm_bonus = 0.0
        if picked_clean != "none" and (picked_clean == title or picked_clean in title):
            vlm_bonus = 1.20

        kw_score = _keyword_score(url, page, question)
        composite = 0.40 * visual_score + 0.45 * vlm_bonus + 0.15 * kw_score
        scored.append((url, page, composite, title))

    scored.sort(key=lambda x: x[2], reverse=True)

    seen_groups = {}
    final = []
    for url, page, score, title in scored:
        group = title.split()[0] if title.split() else "_"
        count = seen_groups.get(group, 0)
        penalized_score = score - 0.15 * count
        final.append((url, page, penalized_score))
        seen_groups[group] = count + 1

    final.sort(key=lambda x: x[2], reverse=True)
    return [(url, page) for url, page, _ in final[:top_n]]

@tool
def tool_cerca_wikipedia(image_path: str) -> str:
    """
    ALWAYS USE THIS TOOL FIRST.
    Analyzes the image to retrieve the TOP 5 most relevant Wikipedia URLs.
    Input MUST be the exact image path.
    """
    st = _state()
    sig = f"{image_path}||{st.current_question}"

    # [GUARDRAIL] Impedisce re-invocazione identica
    if getattr(st, "last_search_signature", None) == sig and getattr(st, "search_done_count", 0) > 0:
        suggerimento = st.available_urls[0] if getattr(st, "available_urls", []) else "https://en.wikipedia.org/wiki/Main_Page"
        return f"⚠️ ALREADY SEARCHED. Call 'tool_leggi_wikipedia' with URL:\n{suggerimento}"

    try:
        clean = image_path.strip()
        if clean.startswith("{"):
            parsed = ast.literal_eval(clean)
            image_path = parsed.get("image_path", clean)
    except Exception: pass
    image_path = str(image_path).strip().split("\n")[0].replace('"','').replace("'","")
    for pfx in ("image_path=", "image="):
        if image_path.startswith(pfx): image_path = image_path[len(pfx):].strip()

    st.ultima_immagine = image_path
    st.url_visitati = []
    st.available_urls = []
    st.logged_urls = []
    st.search_done_count = getattr(st, "search_done_count", 0) + 1
    st.last_search_signature = sig

    if not os.path.exists(image_path): return f"ERROR: File '{image_path}' not found."

    try:
        image_pil = Image.open(image_path).convert("RGB")
        image_features = extract_features(image=image_pil, text=None, model=clip_model, processor=clip_processor, out_dim=512)

        # =========================================================================
        # 🎯 RIFORMA MCQ: Rete espansa a k=400 per estrarre fino a 40 TITOLI UNICI
        # =========================================================================
        _, I_pre = knn_index_immagini.search(image_features, 200)
        titoli_faiss = []
        for idx in I_pre[0]:
            if idx < len(wiki_map):
                u = wiki_map[idx][0]
                if u in wiki_data:
                    t = wiki_data[u].get("title", "").strip()
                    if t and t not in titoli_faiss:
                        titoli_faiss.append(t)
                        if len(titoli_faiss) >= 40: break # <--- Salito da 25 a 40!

        menu_str = "\n".join([f"- {t}" for t in titoli_faiss])
        domanda_contesto = f"\nUser Question: \"{st.current_question}\"\n" if st.current_question else ""

        prompt_mcq = (
            "Look at this image factually."
            f"{domanda_contesto}\n"
            "Below is a candidate entity menu retrieved from a visual database:\n"
            f"{menu_str}\n\n"
            "TASK:\n"
            "1. Write a precise one-sentence factual summary.\n"
            "2. Select the SINGLE most likely title verbatim from the menu above that matches the entity. If none match, output 'NONE'.\n\n"
            "Respond EXACTLY in this format:\n"
            "CAPTION: <summary>\n"
            "PICKED_TITLE: <Exact Title from Menu>"
        )

        messages_vlm = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_mcq}]}]
        risposta_vlm = generate_answer(qwen_model, qwen_processor, messages_vlm, max_new_tokens=90).strip()
        
        caption, picked_title = "Entity analyzed.", "NONE"
        for linea in risposta_vlm.split("\n"):
            linea_pulita = linea.strip()
            if linea_pulita.startswith("CAPTION:"): caption = linea_pulita.replace("CAPTION:", "").strip()
            elif linea_pulita.startswith("PICKED_TITLE:"): picked_title = linea_pulita.replace("PICKED_TITLE:", "").strip()

        st.cached_caption = caption
        print(f"🎯 [VLM MCQ PICK (Menu 40)]: Scelto '{picked_title}'")

        top_candidates = _retrieve_candidates_v4(image_features, st.current_question, picked_title, caption, k_fetch=250, top_n=5)

        if not top_candidates: return "No Wikipedia pages found."

        top_urls = [url for url, _ in top_candidates]
        st.available_urls = top_urls
        st.logged_urls    = top_urls  

        out = f"Visual Caption: {caption}\nAI Selected Title: {picked_title}\n\nTOP 5 CANDIDATES:\n\n"
        for i, (url, page) in enumerate(top_candidates):
            title = page.get("title", "N/A")
            texts = page.get("section_texts", [])
            clean_txt = re.sub(r'\s+', ' ', texts[0][:600]).strip() if texts else "No preview."
            out += f"[{i+1}] URL: {url}\n    Title: {title}\n    Preview: \"{clean_txt}...\"\n\n"

        out += (
            f"NEXT MANDATORY STEP: Read the Previews above carefully. Call 'tool_leggi_wikipedia' "
            f"on the candidate most aligned with the Question using EXACTLY this format:\n"
            f"URL: <selected_url> || QUESTION: {st.current_question}"
        )
        return out

    except Exception as e:
        traceback.print_exc()
        return f"Retrieval error: {e}"

@tool
def tool_leggi_wikipedia(input_str: str) -> str:
    """
    Reads a Wikipedia page. Input MUST be the URL.
    """
    st = _state()

    # [GUARDRAIL] Muro Anti-Veggenza
    if not st.available_urls:
        return (
            "CRITICAL ERROR: STOP! You CANNOT use 'tool_leggi_wikipedia' yet.\n"
            "You are FORBIDDEN from reading before searching.\n"
            "Your VERY NEXT response MUST be:\n"
            "Action: tool_cerca_wikipedia\n"
            "Action Input: <copy the exact Image Path here>"
        )

    # [GUARDRAIL] Autopilot su input vuoto
    if not input_str or not input_str.strip() or input_str == "SEARCH":
        remaining = [u for u in st.available_urls if u not in st.url_visitati]
        if not remaining: return "⚠️ All candidate URLs inspected. Output 'Final Answer:' now."
        input_str = remaining[0]
        print(f"🤖 [AUTO-PILOT] Input vuoto. Leggo in automatico: {input_str}")

    try:
        if "||" in input_str:
            parts = input_str.split("||", 1)
            url_doc = parts[0].replace("URL:", "").strip()
        else:
            url_doc = input_str.replace("URL:", "").strip()
            
        url = str(url_doc).replace('"','').replace("'","").strip()

        # [GUARDRAIL] Scudo Anti-Immagine
        if any(x in url.lower() for x in (".jpg", ".jpeg", ".png", "/work/")):
            safe = st.available_urls[0] if st.available_urls else ""
            return f"CRITICAL ERROR: Pass a Wikipedia URL, not an image path. Try:\n{safe}"

        # [GUARDRAIL] Il Nastro Trasportatore Silenzioso
        if url in st.url_visitati:
            remaining = [u for u in st.available_urls if u not in st.url_visitati]
            if remaining:
                next_u = remaining[0]
                print(f"🔄 [NASTRO TRASPORTATORE] Tentativo di rileggere {url}. Dirotto su {next_u}")
                url = next_u
            else:
                return "⚠️ All candidate URLs already read. Output 'Final Answer:' now."

        if url not in wiki_data:
            remaining = [u for u in st.available_urls if u not in st.url_visitati]
            safe = remaining[0] if remaining else (st.available_urls[0] if st.available_urls else "")
            return f"ERROR: URL not in database. Try:\n{safe}"

        st.url_visitati.append(url)
        if url not in st.read_urls: st.read_urls.append(url)

        page = wiki_data[url]

        # Reranking Sezioni
        q_text = getattr(st, "current_question", "")
        q_words = set(re.findall(r"\b\w{3,}\b", q_text.lower())) - STOPWORDS_RETRIEVAL

        SKIP_SECTIONS = {"see also","references","external links","further reading","bibliography","notes"}
        
        scored_sections = []
        for title, text in zip(page.get("section_titles", []), page.get("section_texts", [])):
            if title.lower() in SKIP_SECTIONS: continue
            sec_lower = title.lower() + " " + text.lower()
            score = sum(1 for w in q_words if w in sec_lower)
            scored_sections.append((title, text, score))
            
        scored_sections.sort(key=lambda x: x[2], reverse=True)
        
        # Budget Cap a 3500 caratteri
        sezioni = []
        corrente_len = 0
        for title, text, score in scored_sections:
            if corrente_len > 12000: break
            blocco = f"--- {title} ---\n{text}"
            sezioni.append(blocco)
            corrente_len += len(blocco)

        tutto = "\n\n".join(sezioni)
        print(f"[READ & PRUNED TO BUDGET] {url} ({corrente_len} chars)")

        if len(tutto) > 12000: 
            tutto = tutto[:12000] + "\n\n...(TEXT TRUNCATED FOR FOCUS)..."

        # =========================================================================
        # 🚨 ELIMINAZIONE TOTALE DEL FOOTER: Restituiamo solo l'intestazione e il testo!
        # =========================================================================
        return f"=== WIKIPEDIA ENTRY: {page.get('title', url)} ===\n\n" + tutto

    except Exception as e:
        return f"Error reading page: {e}"

def _caption_keyword_retrieval(caption: str, question: str, top_k: int = 3) -> list:
    stopwords = {"the","a","an","is","in","of","and","or","with","its","this","that","are","was","were","has","have"}
    words = set(re.findall(r"\b\w{4,}\b", (caption + " " + question).lower())) - stopwords
    if not words: return []
    scored = []
    for url, page in wiki_data.items():
        title = page.get("title","").lower()
        if len(words & set(re.findall(r"\b\w{3,}\b", title))) > 0: scored.append((url, len(words & set(re.findall(r"\b\w{3,}\b", title)))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [url for url, _ in scored[:top_k]]

miei_tools_reali = [tool_cerca_wikipedia, tool_leggi_wikipedia]