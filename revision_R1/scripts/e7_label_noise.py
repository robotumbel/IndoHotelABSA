# -*- coding: utf-8 -*-
"""E7 (R1.5): how much does silver-label noise matter? IndoBERT-base, same protocol as E5.

(A) Gold vs silver supervision, 5-fold CV over the 500 gold reviews (seed 42 fold split).
    Fold f: test = fold f (100), val = fold f+1 (100), gold-train = other 3 folds (300).
    Conditions: silver-only (2,480), gold-only (300), silver+gold (2,780). Seed 42. 15 runs.
(B) Synthetic noise: each silver cell replaced by a different label (uniform over the other
    three of {N/A, POS, NEG, NEU}) with probability k in {0.05, 0.10, 0.20}. Seeds 42/123/7.
    Evaluated on the standard test-400 and blind-100. 9 runs (k = 0 is the E5 run).
(C) Confidence filtering by cross-validated agreement (confident-learning style):
    5-fold out-of-fold IndoBERT predictions on the silver set; keep silver reviews whose
    out-of-fold prediction disagrees with the silver label on 0 cells ("strict") or <= 1 cell
    ("loose"). Seeds 42/123/7 on each. 5 + 6 runs.
Data in data/e7/, predictions in preds_e7/. Resumable.
"""
import json, os, random, subprocess, sys, time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
D = os.path.join(REV, "data", "e7"); P = os.path.join(REV, "preds_e7"); LG = os.path.join(REV, "logs_e7")
for x in (D, P, LG): os.makedirs(x, exist_ok=True)
ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas", "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
LABS = ["NA", "POSITIF", "NEGATIF", "NETRAL"]
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def load(p): return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
def dump(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
def cells(o):
    m = {a: "NA" for a in ASPECTS}
    for l in o.get("labels") or []:
        if l.get("aspect") in m and l.get("polarity") in LABS[1:]: m[l["aspect"]] = l["polarity"]
    return m
def with_cells(o, m):
    o = dict(o); o["labels"] = [{"aspect": a, "polarity": p} for a, p in m.items() if p != "NA"]; return o


def train(tag, trainf, valf, testf, seed):
    out = os.path.join(P, f"{tag}_{seed}.jsonl")
    if os.path.exists(out): print("[skip]", tag, seed, flush=True); return out
    cmd = [sys.executable, "finetune_indobert_absa.py", "--train", trainf, "--val", valf, "--test", testf,
           "--epochs", "4", "--model", "indobenchmark/indobert-base-p1", "--seed", str(seed), "--pred_out", out]
    t0 = time.time(); print("[run ]", tag, seed, flush=True)
    with open(os.path.join(LG, f"{tag}_{seed}.log"), "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(f"       {'ok' if r.returncode == 0 else 'FAILED'} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return out


silver = load(os.path.join(ROOT, "train_silver.jsonl"))
gold = load(os.path.join(ROOT, "gold_final.jsonl"))
SF = os.path.join(ROOT, "train_silver.jsonl"); VAL = os.path.join(ROOT, "val_gold.jsonl"); TEST = os.path.join(ROOT, "test_gold.jsonl")

# ---- (A) CV over gold
ids = sorted(o["review_id"] for o in gold); random.Random(42).shuffle(ids)
folds = [set(ids[i::5]) for i in range(5)]
json.dump([sorted(f) for f in folds], open(os.path.join(D, "gold_folds.json"), "w"))
for f in range(5):
    te, va = folds[f], folds[(f + 1) % 5]
    gtr = [o for o in gold if o["review_id"] not in te | va]
    paths = {k: os.path.join(D, f"cv{f}_{k}.jsonl") for k in ("test", "val", "goldtrain", "silvergold")}
    dump([o for o in gold if o["review_id"] in te], paths["test"]); dump([o for o in gold if o["review_id"] in va], paths["val"])
    dump(gtr, paths["goldtrain"]); dump(silver + gtr, paths["silvergold"])
    train(f"cv{f}_silver", SF, paths["val"], paths["test"], 42)
    train(f"cv{f}_gold", paths["goldtrain"], paths["val"], paths["test"], 42)
    train(f"cv{f}_silvergold", paths["silvergold"], paths["val"], paths["test"], 42)

# ---- (B) synthetic noise
for k in (0.05, 0.10, 0.20):
    rng = random.Random(int(k * 1000)); noisy = []; changed = 0
    for o in silver:
        m = cells(o)
        for a in ASPECTS:
            if rng.random() < k:
                m[a] = rng.choice([x for x in LABS if x != m[a]]); changed += 1
        noisy.append(with_cells(o, m))
    pth = os.path.join(D, f"silver_noise{int(k*100):02d}.jsonl"); dump(noisy, pth)
    print(f"noise {k}: {changed} of {7*len(silver)} cells changed", flush=True)
    for s in (42, 123, 7):
        train(f"noise{int(k*100):02d}", pth, VAL, TEST, s)

# ---- (C) confidence filtering via out-of-fold predictions
sids = [o["review_id"] for o in silver]; random.Random(7).shuffle(sids)
sf = [set(sids[i::5]) for i in range(5)]
oof = {}
for f in range(5):
    tr = os.path.join(D, f"oof{f}_train.jsonl"); ho = os.path.join(D, f"oof{f}_heldout.jsonl")
    dump([o for o in silver if o["review_id"] not in sf[f]], tr); dump([o for o in silver if o["review_id"] in sf[f]], ho)
    out = train(f"oof{f}", tr, VAL, ho, 42)
    for o in load(out): oof[o["review_id"]] = cells(o)
dis = {o["review_id"]: sum(cells(o)[a] != oof[o["review_id"]][a] for a in ASPECTS) for o in silver}
for name, thr in (("strict", 0), ("loose", 1)):
    keep = [o for o in silver if dis[o["review_id"]] <= thr]
    pth = os.path.join(D, f"silver_cl_{name}.jsonl"); dump(keep, pth)
    print(f"confidence filter {name}: kept {len(keep)} of {len(silver)}", flush=True)
    for s in (42, 123, 7):
        train(f"cl_{name}", pth, VAL, TEST, s)
json.dump(dis, open(os.path.join(D, "oof_disagreements.json"), "w"))
print("ALL DONE", flush=True)
