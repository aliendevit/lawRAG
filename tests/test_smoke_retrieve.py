from src.legalrag.retrieve import retrieve

def test_retrieve_smoke():
    r = retrieve(\"Kaufvertrag Pflichten Verkäufer\", mode=\"bm25\", k=5)
    assert \"results\" in r
