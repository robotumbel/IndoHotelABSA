# -*- coding: utf-8 -*-
"""E5 (R2.6, R1.7): train every fine-tuned system on 5 seeds, same data split.

Predictions go to revision_R1/preds_seeds/<system>_<seed>.jsonl, logs to
revision_R1/logs_seeds/. Resumable: a run whose prediction file exists is skipped.
Metrics are computed separately (scripts/eval_seeds.py), so this script only trains.

Existing IndoBERT / HI-ABSA seed predictions from 30 Jul 2026 (same code, same
seeds, 4 epochs) are copied in rather than recomputed.
"""
import os, shutil, subprocess, sys, time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
PRED = os.path.join(REV, "preds_seeds")
LOGS = os.path.join(REV, "logs_seeds")
os.makedirs(PRED, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
os.chdir(ROOT)

PY = sys.executable
SEEDS = [42, 123, 7, 2024, 99]
DATA = ["--train", "train_silver.jsonl", "--val", "val_gold.jsonl", "--test", "test_gold.jsonl"]

# (system id, script, extra args); encoders use 4 epochs as in the seed study
SYSTEMS = [
    ("indobert",      "finetune_indobert_absa.py", ["--epochs", "4", "--model", "indobenchmark/indobert-base-p1"]),
    ("hiabsa",        "finetune_hiabsa.py",        ["--epochs", "4"]),
    ("hiabsa_noaqa",  "finetune_hiabsa.py",        ["--epochs", "4", "--no_aqa"]),
    ("hiabsa_nolgf",  "finetune_hiabsa.py",        ["--epochs", "4", "--no_lgf"]),
    ("hiabsa_none",   "finetune_hiabsa.py",        ["--epochs", "4", "--no_aqa", "--no_lgf"]),
    ("indobertweet",  "finetune_indobert_absa.py", ["--epochs", "4", "--model", "indolem/indobertweet-base-uncased"]),
    ("mbert",         "finetune_indobert_absa.py", ["--epochs", "4", "--model", "bert-base-multilingual-cased"]),
    ("xlmr",          "finetune_indobert_absa.py", ["--epochs", "4", "--model", "xlm-roberta-base"]),
    ("mt5",           "finetune_mt5_absa.py",      []),   # script defaults: mt5-small, 5 epochs
    ("lora_qwen05b",  "finetune_llm_lora.py",      ["--model", "Qwen/Qwen2.5-0.5B-Instruct"]),  # 3 epochs
]

# optional: restrict to some systems, e.g.  --systems lora_qwen05b mt5
if "--systems" in sys.argv:
    keep = sys.argv[sys.argv.index("--systems") + 1:]
    SYSTEMS = [x for x in SYSTEMS if x[0] in keep]

# reuse 30 Jul seed predictions
for s in SEEDS:
    for old, new in (("indobert", "indobert"), ("hiabsa", "hiabsa")):
        src = os.path.join(ROOT, f"pred_seed_{old}_{s}.jsonl")
        dst = os.path.join(PRED, f"{new}_{s}.jsonl")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

for sysid, script, extra in SYSTEMS:
    for s in SEEDS:
        out = os.path.join(PRED, f"{sysid}_{s}.jsonl")
        if os.path.exists(out):
            print(f"[skip] {sysid} seed={s}", flush=True); continue
        cmd = [PY, script] + DATA + extra + ["--seed", str(s), "--pred_out", out]
        log = os.path.join(LOGS, f"{sysid}_{s}.log")
        t0 = time.time()
        print(f"[run ] {sysid} seed={s}", flush=True)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(" ".join(cmd) + "\n")
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        dt = (time.time() - t0) / 60
        status = "ok" if r.returncode == 0 and os.path.exists(out) else f"FAILED rc={r.returncode}"
        print(f"       {status} ({dt:.1f} min) log={log}", flush=True)

print("ALL DONE", flush=True)
