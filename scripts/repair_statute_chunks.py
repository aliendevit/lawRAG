import json
from pathlib import Path

src = Path("data_docs/processed/statute_chunks.jsonl")
dst = Path("data_docs/processed/statute_chunks.fixed.jsonl")

good = 0
bad = None

with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
    for i, line in enumerate(f_in, 1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception as e:
            bad = (i, str(e), s[:200])
            break
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        good += 1

print("good_objects =", good)
if bad:
    print("STOP_AT_BAD_LINE =", bad[0])
    print("ERROR =", bad[1])
    print("PREVIEW =", bad[2])
print("WROTE =", dst)
