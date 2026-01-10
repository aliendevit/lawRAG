# src/legalrag/answer.py
from typing import Any, Dict, List

class Answerer:
    def __init__(self):
        # لاحقاً: اربط LLM أو extractive policy
        pass

    def answer(self, question: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        # V1: grounded extractive/refusal بسيط
        if not hits:
            return {
                "issue": question,
                "rule": "",
                "application": "",
                "conclusion": "لم أجد نصاً داعماً ضمن المصادر المفهرسة.",
                "citations": [],
                "snippets": []
            }

        top = hits[0]["chunk"]
        return {
            "issue": question,
            "rule": "",
            "application": "",
            "conclusion": top.get("text", "")[:800],
            "citations": [ (top.get("citation") or {}).get("canonical") ],
            "snippets": [ (top.get("snippets") or [])[:1] ]
        }
