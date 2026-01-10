import os
root = "data_docs"
hits = []
for dp, _, fns in os.walk(root):
    c = sum(1 for fn in fns if fn.lower().endswith(".md"))
    if c > 0:
        hits.append((c, dp))
hits = sorted(hits, reverse=True)[:10]
print("Top md directories:")
for c, d in hits:
    print(c, d)
