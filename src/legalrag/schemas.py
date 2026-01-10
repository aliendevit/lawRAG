from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Locator(BaseModel):
    gesetz: str
    paragraf: str
    absatz: Optional[str] = None
    satz: Optional[str] = None

class Citation(BaseModel):
    canonical: str
    locator: Locator

class SourceRef(BaseModel):
    source_id: str
    path: str
    sha256: str

class SnippetSpan(BaseModel):
    start: int
    end: int

class Snippet(BaseModel):
    snippet_id: str
    text: str
    span: SnippetSpan

class StatuteChunk(BaseModel):
    chunk_id: str
    doc_type: str = Field(default="statute")
    law: Dict[str, Any]
    provision: Dict[str, Any]
    citation: Citation
    text: str
    snippets: List[Snippet]
    source: SourceRef

class RetrievalHit(BaseModel):
    chunk_id: str
    score: float
    rank: int
    retrievers: Dict[str, Any]
    citation: str
    snippet: Dict[str, Any]

class RetrievalResponse(BaseModel):
    query: str
    mode: str
    k: int
    results: List[RetrievalHit]
    timing_ms: Dict[str, int]

class IRACStatement(BaseModel):
    statement: str
    citations: List[str]

class FinalAnswer(BaseModel):
    issue: str
    rule: List[IRACStatement]
    application: List[IRACStatement]
    conclusion: IRACStatement
    disclaimer: str

class Refusal(BaseModel):
    type: str
    message: str

class AskResponse(BaseModel):
    question: str
    language: str
    answer: Optional[FinalAnswer]
    sources: List[Dict[str, Any]]
    refusal: Optional[Refusal]
    meta: Dict[str, Any]


