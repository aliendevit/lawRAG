# src/legalrag/api/app.py
import time
import logging
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..logging_utils import setup_logging
from ..retrieve import Retriever
from ..answer import Answerer  # تأكد أن answer.py يحتوي class Answerer (أو عدّل الاسم بالأسفل)

log = logging.getLogger("legalrag.api")

app = FastAPI(title="LegalRAG (DE Bundesrecht)", version="0.1.0")

# ---- Request/Response Schemas ----

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(8, ge=1, le=30)

class AskResponse(BaseModel):
    status: str
    answer: Dict[str, Any]
    hits: list

# ---- Lazy singletons ----
_retriever: Optional[Retriever] = None
_answerer: Optional[Any] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        setup_logging()
        t0 = time.time()
        log.info("Loading Retriever (this can be heavy the first time)...")
        _retriever = Retriever()
        log.info("Retriever loaded in %.2fs | docs=%d", time.time() - t0, _retriever.size())
    return _retriever


def get_answerer():
    global _answerer
    if _answerer is None:
        setup_logging()
        _answerer = Answerer()  # إذا اسم الكلاس مختلف عدّله هنا
    return _answerer


@app.get("/health")
def health():
    try:
        r = get_retriever()
        return {"status": "ok", "docs": r.size(), "retrieval_mode": r.mode}
    except Exception as e:
        # مهم: health ما لازم يفجّر السيرفر بلا رسالة مفهومة
        return {"status": "error", "error": str(e)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        r = get_retriever()
        answerer = get_answerer()

        hits = r.search(req.question, top_k=req.top_k)  # Retriever.search يجب أن تُرجع list من hits
        result = answerer.answer(req.question, hits)     # Answerer.answer يجب أن يُرجع dict (Issue/Rule/Application/Conclusion + citations)

        # رجّع hits مختصرة فقط (حتى لا تنفجر الاستجابة)
        hits_slim = []
        for h in hits:
            c = (h.get("chunk") or {})
            hits_slim.append({
                "score": h.get("score", 0.0),
                "chunk_id": c.get("chunk_id"),
                "citation": (c.get("citation") or {}).get("canonical"),
                "path": (c.get("source") or {}).get("path"),
            })

        return {"status": "ok", "answer": result, "hits": hits_slim}

    except Exception as e:
        log.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(e))
# src/legalrag/api/app.py
import time
import logging
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..logging_utils import setup_logging
from ..retrieve import Retriever
from ..answer import Answerer  # تأكد أن answer.py يحتوي class Answerer (أو عدّل الاسم بالأسفل)

log = logging.getLogger("legalrag.api")

app = FastAPI(title="LegalRAG (DE Bundesrecht)", version="0.1.0")

# ---- Request/Response Schemas ----

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(8, ge=1, le=30)

class AskResponse(BaseModel):
    status: str
    answer: Dict[str, Any]
    hits: list

# ---- Lazy singletons ----
_retriever: Optional[Retriever] = None
_answerer: Optional[Any] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        setup_logging()
        t0 = time.time()
        log.info("Loading Retriever (this can be heavy the first time)...")
        _retriever = Retriever()
    log.info("Retriever loaded in %.2fs | docs=%d", time.time() - t0, len(_retriever.docs))
    return _retriever


def get_answerer():
    global _answerer
    if _answerer is None:
        setup_logging()
        _answerer = Answerer()  # إذا اسم الكلاس مختلف عدّله هنا
    return _answerer


@app.get("/health")
def health():
    try:
        r = get_retriever()
        return {"status": "ok", "docs": r.size(), "retrieval_mode": r.mode}
    except Exception as e:
        # مهم: health ما لازم يفجّر السيرفر بلا رسالة مفهومة
        return {"status": "error", "error": str(e)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        r = get_retriever()
        answerer = get_answerer()

        hits = r.search(req.question, top_k=req.top_k)  # Retriever.search يجب أن تُرجع list من hits
        result = answerer.answer(req.question, hits)     # Answerer.answer يجب أن يُرجع dict (Issue/Rule/Application/Conclusion + citations)

        # رجّع hits مختصرة فقط (حتى لا تنفجر الاستجابة)
        hits_slim = []
        for h in hits:
            c = (h.get("chunk") or {})
            hits_slim.append({
                "score": h.get("score", 0.0),
                "chunk_id": c.get("chunk_id"),
                "citation": (c.get("citation") or {}).get("canonical"),
                "path": (c.get("source") or {}).get("path"),
            })

        return {"status": "ok", "answer": result, "hits": hits_slim}

    except Exception as e:
        log.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(e))
