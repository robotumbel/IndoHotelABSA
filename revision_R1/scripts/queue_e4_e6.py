# -*- coding: utf-8 -*-
"""E4 (R1.4) hotel-grouped split and E6 (R1.7) extra ablation variants, 5 seeds each.

E4: IndoBERT and HI-ABSA trained on
    data/train_silver_hotel_grouped.jsonl  (no val/test hotel in training, 2,130 reviews)
    data/train_silver_random_drop.jsonl    (size-matched random subsample, 2,130 reviews)
E6: HI-ABSA pooling/fusion variants on the full silver training set:
    pool=mean, pool=attn1 (one shared learned query), pool=frozenq (random frozen aspect
    queries) -- all without LGF, to compare with AQA-only (hiabsa_nolgf) and shared [CLS]
    (hiabsa_none); and fusion=add (LGF without the gate).
Predictions: preds_e4/<id>_<seed>.jsonl, preds_seeds/<id>_<seed>.jsonl. Resumable.
"""
import os, subprocess, sys, time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
SEEDS = [42, 123, 7, 2024, 99]
G = os.path.join(REV, "data", "train_silver_hotel_grouped.jsonl")
R = os.path.join(REV, "data", "train_silver_random_drop.jsonl")
IB = ["finetune_indobert_absa.py", "--epochs", "4", "--model", "indobenchmark/indobert-base-p1"]
HI = ["finetune_hiabsa.py", "--epochs", "4"]
JOBS = [  # (output dir, id, train file, script + args)
    ("preds_e4", "grouped_indobert", G, IB), ("preds_e4", "grouped_hiabsa", G, HI),
    ("preds_e4", "randdrop_indobert", R, IB), ("preds_e4", "randdrop_hiabsa", R, HI),
    ("preds_seeds", "hiabsa_mean_nolgf", "train_silver.jsonl", HI + ["--pool", "mean", "--no_lgf"]),
    ("preds_seeds", "hiabsa_attn1_nolgf", "train_silver.jsonl", HI + ["--pool", "attn1", "--no_lgf"]),
    ("preds_seeds", "hiabsa_frozenq_nolgf", "train_silver.jsonl", HI + ["--pool", "frozenq", "--no_lgf"]),
    ("preds_seeds", "hiabsa_addfusion", "train_silver.jsonl", HI + ["--fusion", "add"]),
]
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
logs = os.path.join(REV, "logs_e4_e6"); os.makedirs(logs, exist_ok=True)
for d, jid, trainf, cmd in JOBS:
    os.makedirs(os.path.join(REV, d), exist_ok=True)
    for s in SEEDS:
        out = os.path.join(REV, d, f"{jid}_{s}.jsonl")
        if os.path.exists(out):
            print(f"[skip] {jid} seed={s}", flush=True); continue
        full = [sys.executable, cmd[0], "--train", trainf, "--val", "val_gold.jsonl", "--test", "test_gold.jsonl"] \
               + cmd[1:] + ["--seed", str(s), "--pred_out", out]
        t0 = time.time(); print(f"[run ] {jid} seed={s}", flush=True)
        with open(os.path.join(logs, f"{jid}_{s}.log"), "w", encoding="utf-8") as fh:
            fh.write(" ".join(full) + "\n")
            r = subprocess.run(full, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
        ok = r.returncode == 0 and os.path.exists(out)
        print(f"       {'ok' if ok else 'FAILED rc=%d' % r.returncode} ({(time.time()-t0)/60:.1f} min)", flush=True)
print("ALL DONE", flush=True)
