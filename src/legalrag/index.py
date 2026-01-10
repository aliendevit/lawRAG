import os
import json
import pickle
import time
import numpy as np
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from .config import settings
from .textnorm import tokenize
from .logging_utils import setup_logging
import logging

log = logging.getLogger("legalrag.index")

def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
                if bad <= 3:
                    log.warning("Bad JSONL line %d (showing first 200 chars): %r", i, line[:200])
    log.info("Loaded JSONL chunks=%d (bad=%d)", len(chunks), bad)
    return chunks




def build_bm25(chunks: List[Dict[str, Any]]):
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus_tokens)

def try_build_embeddings(chunks: List[Dict[str, Any]]):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.EMBED_MODEL_NAME)
        texts = [c["text"] for c in chunks]
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:
        log.warning("Embeddings unavailable: %s", e)
        return None

def try_build_faiss_index(embeddings: np.ndarray):
    try:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine if normalized
        index.add(embeddings)
        return index
    except Exception as e:
        log.warning("FAISS unavailable: %s", e)
        return None

def main():
    setup_logging()
    os.makedirs(settings.INDEX_DIR, exist_ok=True)

    chunks_path = os.path.join(settings.PROCESSED_DIR, "statute_chunks.jsonl")
    chunks = load_chunks(chunks_path)
    log.info("Loaded chunks: %d", len(chunks))
    if not chunks:
        raise RuntimeError("No chunks loaded. Check your Markdown path and ingestion parsing rules.")

    t0 = time.time()
    bm25 = build_bm25(chunks)
    with open(os.path.join(settings.INDEX_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    log.info("BM25 built in %.2fs", time.time() - t0)

    # doc store
    store_path = os.path.join(settings.INDEX_DIR, "doc_store.jsonl")
    with open(store_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\\n")
    log.info("Doc store saved: %s", store_path)

    emb = try_build_embeddings(chunks)
    if emb is not None:
        np.save(os.path.join(settings.INDEX_DIR, "embeddings.npy"), emb)
        log.info("Embeddings saved.")

        faiss_index = try_build_faiss_index(emb)
        if faiss_index is not None:
            import faiss
            faiss.write_index(faiss_index, os.path.join(settings.INDEX_DIR, "faiss.index"))
            log.info("FAISS index saved.")

if __name__ == "__main__":
    main()


