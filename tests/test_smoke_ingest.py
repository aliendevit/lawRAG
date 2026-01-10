import os
from src.legalrag.ingest import ingest_all
from src.legalrag.config import settings

def test_ingest_smoke():
    ingest_all()
    assert os.path.exists(os.path.join(settings.PROCESSED_DIR, \"statute_chunks.jsonl\"))
    assert os.path.exists(os.path.join(settings.PROCESSED_DIR, \"sources.jsonl\"))
