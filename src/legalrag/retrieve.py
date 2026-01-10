# src/legalrag/retrieve.py
import os
import json
import pickle
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .textnorm import tokenize

log = logging.getLogger("legalrag.retrieve")


def _load_doc_store(path: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception:
                bad += 1
                continue
    log.info("Loaded doc_store docs=%d (bad=%d)", len(docs), bad)
    return docs


def _parse_query_constraints(q: str) -> Dict[str, Optional[str]]:
    """
    Extracts constraints like:
      - paragraf: from "§ 433"
      - law_short: from "BGB"
    """
    import re

    qq = q.strip()
    m = re.search(r"§\s*([0-9]+[a-zA-Z]?)", qq)
    paragraf = m.group(1) if m else None

    # crude law short detection (you can extend list later)
    law_short = None
    m2 = re.search(r"\b(BGB|StGB|ZPO|VwGO|GG|HGB)\b", qq)
    if m2:
        law_short = m2.group(1)

    return {"paragraf": paragraf, "law_short": law_short}


def _match_constraints(chunk: Dict[str, Any], constraints: Dict[str, Optional[str]]) -> bool:
    p = constraints.get("paragraf")
    law = constraints.get("law_short")

    if not p and not law:
        return True

    citation = (chunk.get("citation") or {})
    locator = (citation.get("locator") or {})
    canonical = (citation.get("canonical") or "")

    # paragraf check
    if p:
        # Prefer structured locator
        loc_p = locator.get("paragraf") or locator.get("artikel")
        if loc_p:
            if str(loc_p) != str(p):
                return False
        else:
            # fallback to canonical string check
            if f"§ {p}" not in canonical and f"Art. {p}" not in canonical:
                return False

    # law short check
    if law:
        law_short = (chunk.get("law") or {}).get("short")
        if law_short:
            if law_short != law:
                return False
        else:
            if law not in canonical:
                return False

    return True


class Retriever:
    def __init__(self) -> None:
        self.index_dir = settings.INDEX_DIR
        self.doc_store_path = os.path.join(self.index_dir, "doc_store.jsonl")
        self.bm25_path = os.path.join(self.index_dir, "bm25_index.pkl")
        self.emb_path = os.path.join(self.index_dir, "embeddings.npy")

        if not os.path.exists(self.doc_store_path):
            raise FileNotFoundError(f"Missing doc_store: {self.doc_store_path}")
        if not os.path.exists(self.bm25_path):
            raise FileNotFoundError(f"Missing bm25_index: {self.bm25_path}")

        self.docs = _load_doc_store(self.doc_store_path)

        with open(self.bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        self.embeddings = None
        if os.path.exists(self.emb_path):
            try:
                self.embeddings = np.load(self.emb_path).astype(np.float32)
                log.info("Loaded embeddings: %s", self.embeddings.shape)
            except Exception as e:
                log.warning("Failed to load embeddings.npy: %s", e)

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Returns list of hits:
          [{ "rank": 1, "score": float, "chunk": <doc> }, ...]
        Applies constraint filtering when query includes "§ NNN" and/or law short like "BGB".
        """
        mode = (settings.RETRIEVAL_MODE or "bm25").lower()
        constraints = _parse_query_constraints(query)

        if mode == "emb" and self.embeddings is not None:
            hits = self._search_embeddings(query, top_k=top_k, constraints=constraints)
        elif mode == "hybrid" and self.embeddings is not None:
            hits = self._search_hybrid(query, top_k=top_k, constraints=constraints)
        else:
            hits = self._search_bm25(query, top_k=top_k, constraints=constraints)

        return hits

    def _search_bm25(self, query: str, top_k: int, constraints: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)  # np array

        # take more than top_k first, then filter
        pre_k = min(len(self.docs), max(top_k * 10, 50))
        idxs = np.argpartition(-scores, pre_k - 1)[:pre_k]
        idxs = idxs[np.argsort(-scores[idxs])]

        results: List[Dict[str, Any]] = []
        for i in idxs:
            chunk = self.docs[int(i)]
            if not _match_constraints(chunk, constraints):
                continue
            results.append({"rank": len(results) + 1, "score": float(scores[int(i)]), "chunk": chunk})
            if len(results) >= top_k:
                break

        return results

    def _search_embeddings(self, query: str, top_k: int, constraints: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.EMBED_MODEL_NAME)
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)[0]

        scores = self.embeddings @ q_emb  # cosine if embeddings normalized
        pre_k = min(len(self.docs), max(top_k * 10, 50))
        idxs = np.argpartition(-scores, pre_k - 1)[:pre_k]
        idxs = idxs[np.argsort(-scores[idxs])]

        results: List[Dict[str, Any]] = []
        for i in idxs:
            chunk = self.docs[int(i)]
            if not _match_constraints(chunk, constraints):
                continue
            results.append({"rank": len(results) + 1, "score": float(scores[int(i)]), "chunk": chunk})
            if len(results) >= top_k:
                break

        return results

    def _search_hybrid(self, query: str, top_k: int, constraints: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        # Simple fusion: normalize bm25 + emb ranks
        bm = self._search_bm25(query, top_k=max(top_k * 10, 50), constraints={"paragraf": None, "law_short": None})
        em = self._search_embeddings(query, top_k=max(top_k * 10, 50), constraints={"paragraf": None, "law_short": None})

        # map chunk_id -> score
        scores: Dict[str, float] = {}
        for rank, h in enumerate(bm, start=1):
            cid = h["chunk"].get("chunk_id")
            scores[cid] = scores.get(cid, 0.0) + 1.0 / rank
        for rank, h in enumerate(em, start=1):
            cid = h["chunk"].get("chunk_id")
            scores[cid] = scores.get(cid, 0.0) + 1.0 / rank

        # rebuild list
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # pick docs by id
        id_to_doc = {d.get("chunk_id"): d for d in self.docs}

        results: List[Dict[str, Any]] = []
        for cid, sc in merged:
            chunk = id_to_doc.get(cid)
            if not chunk:
                continue
            if not _match_constraints(chunk, constraints):
                continue
            results.append({"rank": len(results) + 1, "score": float(sc), "chunk": chunk})
            if len(results) >= top_k:
                break

        return results
