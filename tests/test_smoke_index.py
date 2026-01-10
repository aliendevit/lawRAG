import os
from src.legalrag.index import main as index_main
from src.legalrag.config import settings

def test_index_smoke():
    # assumes ingestion already ran in previous test or data exists
    index_main()
    assert os.path.exists(os.path.join(settings.INDEX_DIR, \"bm25_index.pkl\"))
    assert os.path.exists(os.path.join(settings.INDEX_DIR, \"doc_store.jsonl\"))
