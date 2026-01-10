import json
from pathlib import Path

src = Path("data_docs/processed/statute_chunks.jsonl")
dst = Path("data_docs/processed/statute_chunks.fixed.jsonl")

good = 0
bad_line = None

with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
    for i, line in enumerate(f_in, 1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception as e:
            bad_line = (i, str(e), s[:200])
            break
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        good += 1

print("good_objects =", good)
if bad_line:
    print("STOP at bad line:", bad_line[0])
    print("Error:", bad_line[1])
    print("Preview:", bad_line[2])
