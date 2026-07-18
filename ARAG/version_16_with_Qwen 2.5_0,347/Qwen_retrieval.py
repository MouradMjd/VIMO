import json
import os
import faiss
import numpy as np
import requests
import hashlib
import os
import torch
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer
from transformers import set_seed
from qwen_vl_utils import process_vision_info

set_seed(42) # O qualunque numero preferisci, basta che sia fisso
#d

def load_clip_and_index(args, load_faiss=True):
    # 📍 FIX: Rilevamento intelligente del dispositivo
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_clip = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        dtype_clip = torch.bfloat16
        print(f"🚀 Uso la GPU: {device_clip}")
    else:
        # Se i driver sono vecchi o non c'è GPU, ripieghiamo sulla CPU
        print("⚠️ Driver GPU troppo vecchi o GPU non trovata. Ripiego sulla CPU.")
        device_clip = "cpu"
        dtype_clip = torch.float32 # La CPU lavora meglio in float32
    

    print(f"🔄 Caricamento EVA-CLIP su RAM per riparazione...")
    

    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=dtype_clip, 
        trust_remote_code=True,
        local_files_only=True
    )
    
    print("⚕️ Riparazione degli indici del modello in corso...")
    for name, module in clip_model.named_modules():
        if hasattr(module, "position_ids") and module.position_ids is not None:
            shape = module.position_ids.shape
            if len(shape) == 2:
                seq_len = shape[1]
                module.position_ids = torch.arange(seq_len).unsqueeze(0).to(module.position_ids.device)
    print("✅ Riparazione completata!")

    clip_model = clip_model.to(device_clip).eval()
    
    modelli_dir = os.path.dirname(str(args.retriever_path))
    local_clip_processor_path = os.path.join(modelli_dir, "clip-vit-large-patch14")
    
    print(f"🔄 Caricamento processore visivo da locale: {local_clip_processor_path}")
    
    img_proc = CLIPImageProcessor.from_pretrained(
        local_clip_processor_path, 
        local_files_only=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.retriever_path, 
        trust_remote_code=True, 
        local_files_only=True
    )
    
    class CLIPProcessorWrapper:
        def __init__(self, ip, tk):
            self.image_processor = ip
            self.tokenizer = tk
            
        def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
            if images is not None:
                return self.image_processor(images=images, return_tensors=return_tensors, **kwargs)
            if text is not None:
                return self.tokenizer(text=text, return_tensors=return_tensors, **kwargs)
                
    clip_processor = CLIPProcessorWrapper(img_proc, tokenizer)
        
    index, index_map, wiki = None, None, None
    
    # 📍 IL PUNTO CRUCIALE: Metti il caricamento FAISS sotto l'interruttore
    index, index_map, wiki = None, None, None
    
    if load_faiss:
        print(f"📂 Caricamento indici FAISS da {args.index_path}...")
        index = faiss.read_index(str(args.index_path)) 
        with open(args.index_json_path, "r", encoding="utf-8") as f:
            index_map = json.load(f)
        with open(args.kb_wikipedia_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)  
    else:
        print("⏭️ Salto il caricamento di FAISS (modalità creazione indice).")
        
    return clip_model, clip_processor, index, index_map, wiki


def extract_features(image=None, text=None, model=None, processor=None, out_dim=None):
    device = model.device
    dtype = model.dtype 

    with torch.no_grad():
        if image is not None:
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype=dtype, device=device)
            
            if hasattr(model, "get_image_features"):
                features = model.get_image_features(pixel_values=pixel_values)
            elif hasattr(model, "encode_image"): # 🚀 LA CHIAVE PER EVA-CLIP
                features = model.encode_image(pixel_values)
            else:
                features = model(pixel_values=pixel_values)
            
        elif text is not None:
            inputs = processor.tokenizer(text=text, return_tensors="pt", truncation=True, max_length=77)
            input_ids = inputs["input_ids"].to(device)
            
            # Lucchetto anti device-side assert
            input_ids = torch.clamp(input_ids, min=0, max=49407)
            
            if hasattr(model, "get_text_features"):
                features = model.get_text_features(input_ids=input_ids)
            elif hasattr(model, "encode_text"): # 🚀 LA CHIAVE PER EVA-CLIP
                features = model.encode_text(input_ids)
            else:
                features = model(input_ids=input_ids)
        
        # 🚀 L'apriscatole universale (che ha funzionato prima)
        if not isinstance(features, torch.Tensor):
            if hasattr(features, "image_embeds") and features.image_embeds is not None:
                features = features.image_embeds
            elif hasattr(features, "text_embeds") and features.text_embeds is not None:
                features = features.text_embeds
            elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                features = features.pooler_output
            else:
                features = features[0]
        
        features = features / torch.clamp(features.norm(p=2, dim=-1, keepdim=True), min=1e-7)
        
    return features.to(torch.float32).cpu().numpy()

'''
def generate_answer(model, processor, messages, stop=None, **kwargs):
    clean_messages = []
    for m in messages:
        if isinstance(m["content"], list):
            text = " ".join([c["text"] for c in m["content"] if c["type"] == "text"])
            clean_messages.append({"role": m["role"], "content": text})
        else:
            clean_messages.append(m)

    text = processor.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    # 📍 1. Impostiamo i default in modo sicuro dentro kwargs
    if 'max_new_tokens' not in kwargs:
        kwargs['max_new_tokens'] = 512

    # 📍 2. FIX ANTI-ALLUCINAZIONI: Forziamo Greedy Decoding e togliamo i conflitti
    kwargs['do_sample'] = False
    kwargs.pop('temperature', None)
    kwargs.pop('top_p', None)
    kwargs.pop('top_k', None)

    # 📍 3. Aggiungiamo i token di sistema per sicurezza
    kwargs['pad_token_id'] = processor.tokenizer.pad_token_id
    kwargs['eos_token_id'] = processor.tokenizer.eos_token_id
    kwargs['use_cache'] = True

    with torch.no_grad():
        # Ora passiamo SOLO gli input e i kwargs ripuliti! Nessun parametro doppio.
        outputs = model.generate(
            **inputs, 
            **kwargs 
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
'''

def generate_answer(model, processor, messages, stop=None, **kwargs):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 512

    kwargs["do_sample"] = False
    kwargs.pop("temperature", None)
    kwargs.pop("top_p", None)
    kwargs.pop("top_k", None)

    kwargs["pad_token_id"] = processor.tokenizer.pad_token_id
    kwargs["eos_token_id"] = processor.tokenizer.eos_token_id
    kwargs["use_cache"] = True

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **kwargs
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def download_wiki_image_online(url, save_dir="./tmp_wiki_images"):
    """Scarica l'immagine al volo da internet e la salva temporaneamente."""
    os.makedirs(save_dir, exist_ok=True)
    
    if url.startswith("//"): 
        url = "https:" + url
        
    hash_name = hashlib.md5(url.encode()).hexdigest() + ".jpg"
    save_path = os.path.join(save_dir, hash_name)
    
    if os.path.exists(save_path): 
        return save_path 
    
    try:
        headers = {'User-Agent': 'ViMo_Research_Bot/1.0'}
        r = requests.get(url, stream=True, timeout=5, headers=headers)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(1024): 
                    f.write(chunk)
            return save_path
    except Exception as e:
        print(f"⚠️ Impossibile scaricare l'immagine {url}: {e}")
    return None


def retrieve_topk_pages(features, index, index_map, wiki, k):
    """Fase 1: Ritorna SOLO un riassunto dei documenti trovati."""
    _, I = index.search(features, k)
    
    # In knn.json, index_map[i] è una lista: [URL, Titolo, Path_Leonardo]
    # Usiamo l'URL come chiave di ricerca per il dictionary wiki
    doc_urls = [index_map[i][0] for i in I[0]]
    sommario = []
    
    for url in doc_urls:
        if url not in wiki: 
            continue
        page = wiki[url]
        title = page.get("title", "Senza Titolo")
        sezioni = page.get("section_titles", [])
        
        doc_info = f"📌 [URL_DOC:    {url}]\nTitolo: {title}\nSezioni disponibili per la lettura:"
        for idx, sec_title in enumerate(sezioni):
            doc_info += f"\n  - Sezione {idx}: {sec_title}"
        sommario.append(doc_info)
        
    if not sommario:
        return "Nessun documento utile trovato nel Knowledge Base testuale per questa query."
        
    return "\n\n".join(sommario)


def read_wiki_section_with_images(url_doc, section_idx, use_images, wiki):
    """Fase 2: Legge la singola sezione e scarica MAX 3 immagini valide."""
    if url_doc not in wiki:
        return "Errore: Documento non trovato nel database."
        
    page = wiki[url_doc]
    
    try:
        section_idx = int(section_idx)
        testo_sezione = page["section_texts"][section_idx]
        titolo_sezione = page["section_titles"][section_idx]
    except IndexError:
        return f"Errore: La sezione {section_idx} non esiste in questo documento."

    risposta = f"📖 Testo della Sezione '{titolo_sezione}':\n{testo_sezione}\n"
    
    if use_images:
        img_urls = page.get("image_urls", [])
        img_sec_idx = page.get("image_section_indices", [])
        
        # Trova le immagini di questa sezione e filtra SVG/file non supportati
        immagini_della_sezione = [img_urls[i] for i, s_idx in enumerate(img_sec_idx) if s_idx == section_idx]
        
        # 📍 FIX VRAM & SVG: Prendiamo solo file immagine veri e MAX 3 per non far esplodere la GPU
        immagini_valide = [url for url in immagini_della_sezione if not url.lower().endswith(('.svg', '.pdf', '.gif', '.ogg'))][:3]
        
        if immagini_valide:
            risposta += "\n🖼️ Immagini allegate trovate (scaricate al volo):\n"
            for url in immagini_valide:
                local_path = download_wiki_image_online(url)
                if local_path:
                    risposta += f"[IMG_WIKI: {local_path}]\n"
        else:
            risposta += "\n(Nessuna immagine supportata presente in questa sezione)."
            
    return risposta