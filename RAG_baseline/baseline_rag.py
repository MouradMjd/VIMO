import os
import json
from PIL import Image
import traceback
import tools_real
from load_config import load_config
from Qwen_retrieval import extract_features, generate_answer

from eval_utils import (
    build_common_record, elapsed, now_seconds,
    parse_retrieved_urls, parse_sections, retrieval_metrics, token_estimate,
)

# Inizializzazione
config = load_config()
class Args: pass
args = Args()
for k, v in config.items():
    setattr(args, k, str(v))
args.top_k = 5  # <--- SETTATO A 5 PER ALLINEARSI ALL'AGENTIC

tools_real.start_motors(args)

def run_standard_rag(image_path, question, return_metadata=False, question_id=None, ground_truth="", question_type="unknown", expected_sources=None, top_k=5):
    print(f"\n🔍 [BASELINE RAG K={top_k}] Avvio ricerca per: {image_path}")
    
    start = now_seconds()
    error = None
    context = ""
    retrieved_urls = []
    sections = []
    answer = ""
    
    try:
        # ==========================================
        # 1. RETRIEVE: Estrazione feature e ricerca
        # ==========================================
        image_pil = Image.open(image_path).convert("RGB")
        features = extract_features(
            image=image_pil, 
            text=None,
            model=tools_real.clip_model, 
            processor=tools_real.clip_processor, 
            out_dim=512
        )
        
        _, I = tools_real.knn_index_immagini.search(features, top_k)
        
        # Filtriamo URL validi
        for idx in I[0]:
            if idx < len(tools_real.wiki_map):
                url = tools_real.wiki_map[idx][0]
                if url in tools_real.wiki_data and url not in retrieved_urls:
                    retrieved_urls.append(url)

        # ==========================================
        # 2. BUILD CONTEXT: Unione di tutti i doc (Nessun filtro)
        # ==========================================
        context_parts = []

        for url in retrieved_urls:
            page = tools_real.wiki_data[url]
            title = page.get("title", "Unknown")
            page_text = f"=== Document Title: {title} (URL: {url}) ===\n"

            sezioni_doc = []
            for sec_title, sec_text in zip(page.get("section_titles", []), page.get("section_texts", [])):
                # Rimosso il filtro SKIP_SECTIONS. Aggiungiamo tutto il testo grezzo.
                sezioni_doc.append(f"--- {sec_title} ---\n{sec_text}")

            tutto_testo = "\n\n".join(sezioni_doc)
            
            # Limite di sicurezza: 10.000 caratteri per doc (circa 50.000 totali)
            # Indispensabile per non causare Out Of Memory su GPU da 48GB.
            limit = 15000
            if len(tutto_testo) > limit:
                tutto_testo = tutto_testo[:limit] + "\n...(TEXT TRUNCATED DUE TO LENGTH)..."

            page_text += tutto_testo
            context_parts.append(page_text)
            sections.append(f"{title} - Full Page Raw")

        context = "\n\n".join(context_parts)

        # ==========================================
        # 3. GENERATION: Prompting Qwen2.5-VL
        # ==========================================
        prompt_text = (
            f"You are a precise AI assistant answering questions based on visual observation and the provided Wikipedia context.\n\n"
            f"--- WIKIPEDIA CONTEXT ---\n"
            f"{context}\n"
            f"-------------------------\n\n"
            f"Question: {question}\n"
            f"Answer the question concisely using ONLY the provided context and the image."
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text}
            ]}
        ]
        
        answer = generate_answer(
            tools_real.qwen_model, 
            tools_real.qwen_processor, 
            messages, 
            max_new_tokens=128
        )

    except Exception as e:
        traceback.print_exc()
        error = str(e)
        answer = f"ERRORE RAG: {error}"

    latency = elapsed(start)

    if not return_metadata:
        return answer

    ret_metrics = retrieval_metrics(retrieved_urls, expected_sources, k=top_k)
    record = build_common_record(
        question_id=question_id,
        model_name="baseline_rag_k5", # Rinominiamo il modello per distinguerlo
        image_path=image_path,
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        question_type=question_type,
        latency_seconds=latency,
        error=error,
        extra={
            "has_retrieval": 1,
            "retrieval_mode": f"visual_k{top_k}",
            "top_k": top_k,
            "retrieved_urls": retrieved_urls,
            "retrieved_sections": sections,
            "retrieved_context_chars": len(context),
            "context_tokens_est": token_estimate(context),
            "num_steps": 1,
            "num_tool_calls": 0,
            "num_retrieval_calls": 1,
            "visual_input_used": 1,
            **ret_metrics,
        }
    )
    return record

if __name__ == "__main__":
    print("🚀 Accensione motori per la Baseline RAG (Top-5 docs, Testo Grezzo)...")
    
    # Facciamo un test dummy per verifica
    immagine_test = "foto_buia.jpg"
    domanda_test = "Chi ha dipinto quest'opera e in che anno?"
    
    if not os.path.exists(immagine_test):
        Image.new('RGB', (224, 224), color = 'black').save(immagine_test)
        
    print("\nTest Esecuzione:")
    record = run_standard_rag(immagine_test, domanda_test, return_metadata=True, top_k=5)
    print("\nRisposta ottenuta:")
    print(record["answer"])