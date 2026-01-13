# src/legalrag/retrieve.py
import os
import json
import pickle
import logging
import numpy as np
from typing import Any, Dict, List, Optional

from .config import settings
from .textnorm import tokenize

log = logging.getLogger("legalrag.retrieve")
# --- Topic anchors: concept -> preferred canonical citation prefixes ---
TOPIC_ANCHORS = {
    # damages / liability
    "schadensersatz": ["BGB § 249", "BGB § 280", "BGB § 823", "BGB § 253"],
    "schadenersatz": ["BGB § 249", "BGB § 280", "BGB § 823", "BGB § 253"],  # typo defense
    "haftung": ["BGB § 823", "BGB § 280", "BGB § 276", "BGB § 278"],
    "verzug": ["BGB § 286", "BGB § 280", "BGB § 288"],

    # admin law
    "verwaltungsakt": ["VwVfG § 35"],
    "widerspruch": ["VwGO § 68"],

    # criminal law
    "diebstahl": ["StGB § 242"],
    "betrug": ["StGB § 263"],

    # contract basics
    "kaufvertrag": ["BGB § 433"],
}

def _is_definition_question(q: str) -> bool:
    qq = q.strip().lower()
    return (
        qq.startswith("was ist")
        or "definition" in qq
        or "begriff" in qq
        or "was bedeutet" in qq
    )

def _load_doc_store(path: str) -> List[Dict[str, Any]]:
    """
    Robust loader for doc_store that supports:
      - proper JSONL (one JSON object per line)
      - concatenated JSON objects on the same line (e.g., ...}{...)
      - extra whitespace / BOM
    """
    docs: List[Dict[str, Any]] = []
    bad = 0
    decoder = json.JSONDecoder()
    buf = ""

    def drain(final: bool) -> None:
        nonlocal buf, bad
        while True:
            s = buf.lstrip()
            if not s:
                buf = ""
                return

            # Handle BOM
            if s[0] == "\ufeff":
                s = s[1:]

            try:
                obj, idx = decoder.raw_decode(s)
            except json.JSONDecodeError:
                if final:
                    if s.strip() == "":
                        buf = ""
                        return
                    bad += 1
                    buf = ""
                    return
                buf = s
                return

            if isinstance(obj, dict):
                docs.append(obj)
            else:
                bad += 1

            buf = s[idx:]  # continue parsing any concatenated objects

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            buf += chunk
            drain(final=False)

    drain(final=True)
    log.info("Loaded doc_store docs=%d (bad=%d)", len(docs), bad)
    return docs


def _parse_query_constraints(q: str) -> Dict[str, Optional[str]]:
    """
    Extract constraints:
      - paragraf: from '§ 433'
      - law_short: from 'BGB', 'VwVfG', etc.
    """
    import re

    qq = q.strip()

    m = re.search(r"§\s*([0-9]+[a-zA-Z]?)", qq)
    paragraf = m.group(1) if m else None

    law_short = None
    m2 = re.search(r"\b(BGB|StGB|ZPO|VwGO|VwVfG|GG|HGB)\b", qq)
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

    # Paragraph / article check
    if p:
        loc_p = locator.get("paragraf") or locator.get("artikel")
        if loc_p:
            if str(loc_p) != str(p):
                return False
        else:
            if (f"§ {p}" not in canonical) and (f"Art. {p}" not in canonical):
                return False

    # Law short check
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

        # Helpful sanity check
        try:
            bm_size = getattr(self.bm25, "corpus_size", None)
            if bm_size is not None and bm_size != len(self.docs):
                log.warning(
                    "BM25 corpus_size (%s) != doc_store size (%s). Retrieval may break.",
                    bm_size,
                    len(self.docs),
                )
        except Exception:
            pass

    # ---- Compatibility helpers (your API / old code may expect these) ----
    def size(self) -> int:
        return len(self.docs)

    def query(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        return self.search(query, top_k=top_k)

    # -------------------------------------------------------------------

    def _mk_hit(self, rank: int, score: float, chunk: Dict[str, Any]) -> Dict[str, Any]:
        citation = (chunk.get("citation") or {}).get("canonical")
        path = (chunk.get("source") or {}).get("path") or chunk.get("path")
        return {
            "rank": int(rank),
            "score": float(score),
            "chunk_id": chunk.get("chunk_id"),
            "citation": citation,
            "path": path,
            "chunk": chunk,
        }
    def _anchor_hits(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        qq = query.lower()
        matched_prefixes: List[str] = []

        for key, prefixes in TOPIC_ANCHORS.items():
            if key in qq:
                matched_prefixes.extend(prefixes)

        if not matched_prefixes:
            return []

    # Scan docs and pick chunks whose canonical citation starts with an anchor prefix
        hits: List[Dict[str, Any]] = []
        for pref in matched_prefixes:
            for d in self.docs:
                canon = ((d.get("citation") or {}).get("canonical") or "")
                if canon.startswith(pref):
                    hits.append(self._mk_hit(len(hits) + 1, 998.0, d))
                    if len(hits) >= top_k:
                        return hits

        return hits

    def _exact_citation_hits(self, constraints: Dict[str, Optional[str]], top_k: int) -> List[Dict[str, Any]]:
        """
        Fast-path when we have BOTH law_short and paragraf:
          query like '§ 433 BGB' or 'Was regelt § 433 BGB?'
        We try to return exact canonical matches first.
        """
        p = constraints.get("paragraf")
        law = constraints.get("law_short")
        if not p or not law:
            return []

        # Prefer 'BGB § 433' exact prefix match in canonical citation
        want_prefix = f"{law} § {p}"
        hits: List[Dict[str, Any]] = []

        for d in self.docs:
            canon = ((d.get("citation") or {}).get("canonical") or "")
            if canon.startswith(want_prefix):
                hits.append(self._mk_hit(len(hits) + 1, 999.0, d))
                if len(hits) >= top_k:
                    break

        return hits

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        if not self.docs:
            return []

        constraints = _parse_query_constraints(query)

    # 1) exact § + law shortcut (you already have)
        exact = self._exact_citation_hits(constraints, top_k=top_k)
        if exact:
            return exact

    # 2) topic anchors: especially important for "Was ist ...?"
        if _is_definition_question(query):
            anchored = self._anchor_hits(query, top_k=top_k)
            if anchored:
                return anchored

    # 3) fallback to your chosen retrieval mode
        mode = (settings.RETRIEVAL_MODE or "bm25").lower()

        if mode == "emb" and self.embeddings is not None:
            return self._search_embeddings(query, top_k=top_k, constraints=constraints)

        if mode == "hybrid" and self.embeddings is not None:
            return self._search_hybrid(query, top_k=top_k, constraints=constraints)

        return self._search_bm25(query, top_k=top_k, constraints=constraints)  


    def _search_bm25(self, query: str, top_k: int, constraints: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)  # np array of length corpus_size

        # If query has constraints, pull a MUCH larger candidate pool
        has_constraints = bool(constraints.get("paragraf") or constraints.get("law_short"))
        if has_constraints:
            pre_k = min(len(self.docs), max(2000, top_k * 500))
        else:
            pre_k = min(len(self.docs), max(top_k * 10, 50))

        # Guard if scores length mismatches doc length
        n = min(len(self.docs), len(scores))
        if n <= 0:
            return []

        pre_k = min(pre_k, n)

        idxs = np.argpartition(-scores[:n], pre_k - 1)[:pre_k]
        idxs = idxs[np.argsort(-scores[idxs])]

        results: List[Dict[str, Any]] = []
        for i in idxs:
            ii = int(i)
            if ii < 0 or ii >= len(self.docs):
                continue
            chunk = self.docs[ii]
            if not _match_constraints(chunk, constraints):
                continue
            results.append(self._mk_hit(len(results) + 1, float(scores[ii]), chunk))
            if len(results) >= top_k:
                break

        return results

    def _search_embeddings(self, query: str, top_k: int, constraints: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.docs:
            return []

        # Safety if embeddings rows don't match docs
        n = min(len(self.docs), int(self.embeddings.shape[0]))
        if n <= 0:
            return []

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.EMBED_MODEL_NAME)
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)[0]

        scores = self.embeddings[:n] @ q_emb

        pre_k = min(n, max(top_k * 10, 50))
        idxs = np.argpartition(-scores, pre_k - 1)[:pre_k]
        idxs = idxs[np.argsort(-scores[idxs])]

        results: List[Dict[str, Any]] = []
        for i in idxs:
            ii = int(i)
            chunk = self.docs[ii]
            if not _match_constraints(chunk, constraints):
                continue
            results.append(self._mk_hit(len(results) + 1, float(scores[ii]), chunk))
            if len(results) >= top_k:
                break

        return results

    def _search_hybrid(self, query: str, top_k: int, constraints: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.docs:
            return self._search_bm25(query, top_k=top_k, constraints=constraints)

        bm = self._search_bm25(
            query,
            top_k=max(top_k * 10, 50),
            constraints={"paragraf": None, "law_short": None},
        )
        em = self._search_embeddings(
            query,
            top_k=max(top_k * 10, 50),
            constraints={"paragraf": None, "law_short": None},
        )

        fused: Dict[str, float] = {}
        for rank, h in enumerate(bm, start=1):
            cid = h.get("chunk_id")
            if not cid:
                continue
            fused[cid] = fused.get(cid, 0.0) + 1.0 / rank

        for rank, h in enumerate(em, start=1):
            cid = h.get("chunk_id")
            if not cid:
                continue
            fused[cid] = fused.get(cid, 0.0) + 1.0 / rank

        if not fused:
            return self._search_bm25(query, top_k=top_k, constraints=constraints)

        id_to_doc = {d.get("chunk_id"): d for d in self.docs}
        merged = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for cid, sc in merged:
            chunk = id_to_doc.get(cid)
            if not chunk:
                continue
            if not _match_constraints(chunk, constraints):
                continue
            results.append(self._mk_hit(len(results) + 1, float(sc), chunk))
            if len(results) >= top_k:
                break

        return results
    