# -*- coding: utf-8 -*-
"""R2 (reviewer 2, point 2): remaining encoders trained on the official hotel-grouped
training set (no validation or test hotel in training; 2,130 silver reviews), 5 seeds.
IndoBERT and HI-ABSA were already run in R1 (E4); this adds mBERT, XLM-R, IndoBERTweet.
Predictions: preds_r2/grouped_<id>_<seed>.jsonl. Resumable."""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, REV
SEEDS = [42, 123, 7, 2024, 99]
G = os.path.join(REV, "data", "train_silver_random_drop.jsonl")
FT = ["finetune_indobert_absa.py", "--epochs", "4", "--model"]
JOBS = [("mbert", FT + ["bert-base-multilingual-cased"]),
        ("xlmr", FT + ["xlm-roberta-base"]),
        ("indobertweet", FT + ["indolem/indobertweet-base-uncased"])]
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
logs = os.path.join(REV, "logs_r2"); os.makedirs(logs, exist_ok=True)
os.makedirs(os.path.join(REV, "preds_r2"), exist_ok=True)
for jid, cmd in JOBS:
    for s in SEEDS:
        out = os.path.join(REV, "preds_r2", f"randdrop_{jid}_{s}.jsonl")
        if os.path.exists(out):
            print(f"[skip] {jid} seed={s}", flush=True); continue
        full = [sys.executable, cmd[0], "--train", G, "--val", "val_gold.jsonl", "--test", "test_gold.jsonl"] \
               + cmd[1:] + ["--seed", str(s), "--pred_out", out]
        t0 = time.time(); print(f"[run ] randdrop_{jid} seed={s}", flush=True)
        with open(os.path.join(logs, f"randdrop_{jid}_{s}.log"), "w", encoding="utf-8") as fh:
            fh.write(" ".join(full) + "\n")
            r = subprocess.run(full, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
        ok = r.returncode == 0 and os.path.exists(out)
        print(f"       {'ok' if ok else 'FAILED rc=%d' % r.returncode} ({(time.time()-t0)/60:.1f} min)", flush=True)
print("ALL DONE", flush=True)
