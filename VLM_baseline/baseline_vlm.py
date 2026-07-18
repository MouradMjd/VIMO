import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from Qwen_retrieval import generate_answer
from load_config import load_config
from eval_utils import build_common_record, elapsed, now_seconds, token_estimate

# Caricamento configurazione
config = load_config()

print("🚀 Caricamento Plain VLM...")
processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cuda:0"
).eval()

def run_vlm_only(image_path, question, return_metadata=False, question_id=None, ground_truth="", question_type="unknown", expected_sources=None):
    start = now_seconds()
    error = None
    answer = ""
    
    try:
        # Messaggio multimodale diretto
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Generazione risposta (senza aiuti esterni)
        answer = generate_answer(model, processor, messages, max_new_tokens=256)
    except Exception as e:
        error = str(e)
        answer = f"ERRORE VLM: {error}"

    latency = elapsed(start)

    if not return_metadata:
        return answer

    record = build_common_record(
        question_id=question_id,
        model_name="baseline_vlm",
        image_path=image_path,
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        question_type=question_type,
        latency_seconds=latency,
        error=error,
        extra={
            "has_retrieval": 0,
            "retrieval_mode": "none",
            "top_k": 0,
            "retrieved_urls": [],
            "retrieved_sections": [],
            "retrieved_context_chars": 0,
            "context_tokens_est": 0,
            "num_steps": 1,
            "num_tool_calls": 0,
            "num_retrieval_calls": 0,
            "visual_input_used": 1,
        }
    )
    return record

if __name__ == "__main__":
    # Test rapido
    res = run_vlm_only("foto_buia.jpg", "Chi è l'autore e quali sono le sue invenzioni?")
    print(f"Risposta VLM: {res}")