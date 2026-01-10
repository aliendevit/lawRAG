# src/legalrag/config.py
import os
from dataclasses import dataclass
from pathlib import Path

def _load_dotenv() -> None:
    """
    Minimal .env loader (no extra dependency).
    Reads KEY=VALUE lines from project-root .env if present.
    Does not override existing environment variables.
    """
    # project root = folder that contains /src
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and (k not in os.environ):
            os.environ[k] = v

_load_dotenv()

@dataclass(frozen=True)
class Settings:
    RAW_MD_DIR: str = os.getenv("RAW_MD_DIR", "data_docs/raw/gesetze-master")
    PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", "data_docs/processed")
    INDEX_DIR: str = os.getenv("INDEX_DIR", "data_indexes")  # you will override via .env

    RETRIEVAL_MODE: str = os.getenv("RETRIEVAL_MODE", "hybrid")  # bm25 | emb | hybrid
    TOP_K: int = int(os.getenv("TOP_K", "8"))
    CANDIDATES_BM25: int = int(os.getenv("CANDIDATES_BM25", "60"))
    CANDIDATES_EMB: int = int(os.getenv("CANDIDATES_EMB", "60"))

    EMBED_MODEL_NAME: str = os.getenv(
        "EMBED_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

settings = Settings()
