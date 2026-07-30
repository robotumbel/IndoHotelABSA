# -*- coding: utf-8 -*-
"""
adjudicate_gold.py — Adjudikasi majority-vote 3 anotator -> gold final,
hitung LLM-human agreement vs gold, dan bangun split eksperimen:

  train_silver.jsonl : silver (LLM) MINUS 500 gold  -> data latih
  val_gold.jsonl     : 100 gold                     -> validasi
  test_gold.jsonl    : 400 gold                     -> uji (human-verified)

Aturan adjudikasi per sel (review x aspek):
  - mayoritas (>=2 dari 3) menang
  - bila ketiganya beda semua (3-way tie, jarang): pakai label LLM sebagai
    pemecah seri dan dicatat jumlahnya.
"""
import json, random, sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

ASP = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
       "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]


def load(p):
    d = {}
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if "review_id" in o:
                d[o["review_id"]] = o
    return d


def tomap(o):
    return {l["aspect"]: l["polarity"] for l in o.get("labels", [])}


anns = [load(f"anotator_{i}_verified.jsonl") for i in (1, 2, 3)]
pre = load("annotation_prefilled.jsonl")
common = sorted(set(anns[0]) & set(anns[1]) & set(anns[2]))
print(f"Review gold (irisan 3 anotator): {len(common)}")

# ── adjudikasi majority-vote ────────────────────────────────
gold = []
ties = 0
for rid in common:
    maps = [tomap(a[rid]) for a in anns]
    labels = []
    for asp in ASP:
        votes = [m.get(asp, "NA") for m in maps]
        cnt = Counter(votes).most_common()
        if cnt[0][1] >= 2:                       # mayoritas
            lab = cnt[0][0]
        else:                                    # 3-way tie -> pemecah: LLM
            ties += 1
            lab = tomap(pre[rid]).get(asp, "NA") if rid in pre else "NA"
        if lab != "NA":
            labels.append({"aspect": asp, "polarity": lab})
    r = anns[0][rid]
    gold.append({"review_id": rid, "hotel": r.get("hotel", ""),
                 "city": r.get("city", ""), "rating": r.get("rating"),
                 "text": r["text"], "labels": labels, "verified": True,
                 "gold": True})
print(f"Sel 3-way tie (dipecah label LLM): {ties}")

with open("gold_final.jsonl", "w", encoding="utf-8") as f:
    for r in gold:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"gold_final.jsonl: {len(gold)} review (adjudicated)")

# ── LLM-human agreement vs gold ─────────────────────────────
agree = tot = 0
for r in gold:
    rid = r["review_id"]
    if rid not in pre:
        continue
    gm, pm = tomap(r), tomap(pre[rid])
    for asp in ASP:
        tot += 1
        if gm.get(asp, "NA") == pm.get(asp, "NA"):
            agree += 1
print(f"LLM-human agreement (vs gold adjudicated): {agree/tot*100:.1f}% "
      f"({agree}/{tot} sel)")

# ── statistik label gold ────────────────────────────────────
pol = Counter(); asp_cnt = Counter()
for r in gold:
    for l in r["labels"]:
        pol[l["polarity"]] += 1
        asp_cnt[l["aspect"]] += 1
print("\nDistribusi polaritas gold:", dict(pol))
print("Anotasi per aspek gold:")
for a, n in asp_cnt.most_common():
    print(f"  {a:<22} {n}")

# ── split eksperimen ────────────────────────────────────────
gold_ids = {r["review_id"] for r in gold}
silver_train = [o for rid, o in sorted(pre.items())
                if rid not in gold_ids and o.get("labels")]
random.shuffle(gold)
val_gold, test_gold = gold[:100], gold[100:]

for name, part in [("train_silver", silver_train),
                   ("val_gold", val_gold), ("test_gold", test_gold)]:
    with open(f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}.jsonl: {len(part)} review")

print("\nSetup eksperimen:")
print("  latih : train_silver.jsonl (silver LLM, di luar gold)")
print("  val   : val_gold.jsonl (100 gold)")
print("  uji   : test_gold.jsonl (400 gold, human-verified)")
