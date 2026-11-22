# -*- coding: utf-8 -*-
"""Rebuild the official hotel-grouped split (splits/hotel_grouped_v1).
Hotel identity = (lower-cased stripped hotel name, city). Run from the data root
(train_silver.jsonl, val_gold.jsonl, test_gold.jsonl present). Also writes
train_silver_hotel_grouped.jsonl (the grouped training reviews with text) to the data root."""
import json, os, sys
out = sys.argv[1] if len(sys.argv) > 1 else "splits/hotel_grouped_v1"
os.makedirs(out, exist_ok=True)
rd = lambda f: [json.loads(l) for l in open(f, encoding="utf-8-sig") if l.strip()]
hk = lambda o: (o["hotel"].strip().lower(), o["city"])
tr, va, te = rd("train_silver.jsonl"), rd("val_gold.jsonl"), rd("test_gold.jsonl")
vt = {hk(o) for o in va + te}
g = [o for o in tr if hk(o) not in vt]
trh = {hk(o) for o in tr}
unseen = [o for o in te if hk(o) not in trh]
assert not ({hk(o) for o in g} & vt)
w = lambda n, rows: open(os.path.join(out, n), "w").write("\n".join(str(o["review_id"]) for o in rows) + "\n")
w("train_ids.txt", g); w("val_ids.txt", va); w("test_ids.txt", te); w("test_unseen_in_full_train_ids.txt", unseen)
with open("train_silver_hotel_grouped.jsonl", "w", encoding="utf-8") as fo:
    for o in g: fo.write(json.dumps(o, ensure_ascii=False) + "\n")
print(len(g), len(va), len(te), len(unseen))
