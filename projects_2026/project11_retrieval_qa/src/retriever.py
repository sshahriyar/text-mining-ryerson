import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Global variables — set when build_index() is called
_embed_model   = None
_index         = None
_all_passages  = []


def build_index(df, model_name='all-MiniLM-L6-v2', batch_size=64):
    
    global _embed_model, _index, _all_passages

    # Step 1 — Load embedding model
    _embed_model = SentenceTransformer(model_name)
    print("Embedding model loaded!")

    # Step 2 — Extract all passage texts from the dataset
    # Each row has up to 25 context passages stored as JSON strings
    _all_passages = []

    for _, row in df.iterrows():
        for ctx_str in list(row['context']):
            try:
                ctx  = json.loads(ctx_str)   # parse JSON string → dict
                text = ctx.get('text', '').strip()
                if text:
                    _all_passages.append(text)
            except Exception:
                continue   # skip any malformed JSON entries

    print(f"Total passages extracted: {len(_all_passages)}")

    # Step 3 — Encode all passages into vectors
    passage_embeddings = _embed_model.encode(
        _all_passages,
        show_progress_bar=True,
        batch_size=batch_size
    )
    print(f"Embeddings shape: {passage_embeddings.shape}")

    # Step 4 — L2-normalize vectors
    faiss.normalize_L2(passage_embeddings)

    # Step 5 — Build FAISS index
    # IndexFlatIP = exact inner product (cosine) search.
    dimension = passage_embeddings.shape[1]   # 384
    _index    = faiss.IndexFlatIP(dimension)
    _index.add(passage_embeddings.astype('float32'))

    print(f"\nFAISS index built!")
    print(f"  Passages indexed  : {_index.ntotal}")
    print(f"  Vector dimensions : {dimension}")

    return _index, _all_passages, _embed_model


def retrieve(question, k=3):
    
    if _embed_model is None or _index is None:
        raise RuntimeError("Call build_index() before retrieve().")

    # Encode the question into a vector
    query_vec = _embed_model.encode([question])

    # Normalize 
    faiss.normalize_L2(query_vec)

    # Search — D = similarity scores, I = indices of top-k passages
    D, I = _index.search(query_vec.astype('float32'), k)

    # Return actual passage texts using the retrieved indices
    return [_all_passages[i] for i in I[0] if i < len(_all_passages)]
