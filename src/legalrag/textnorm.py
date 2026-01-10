import re
from typing import List

# conservative tokenizer suitable for German + citations
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text)]


