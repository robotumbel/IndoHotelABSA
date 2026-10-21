# -*- coding: utf-8 -*-
"""Wait for E5 (run_seeds_all.py) to finish, then run GPU jobs that cannot share the 6 GB GPU.

Queue: 1) Qwen2.5-7B-Instruct zero-shot, 4-bit, on the 400 gold test reviews.
"""
import os, subprocess, sys, time
from huggingface_hub import snapshot_download

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
E5_LOG = os.path.join(REV, "logs_run_seeds_all.txt")

while True:
    try:
        if "ALL DONE" in open(E5_LOG, encoding="utf-8", errors="ignore").read():
            break
    except FileNotFoundError:
        pass
    time.sleep(60)
print("E5 finished; starting queue", flush=True)

snapshot_download("Qwen/Qwen2.5-7B-Instruct")   # no-op if already complete
cmd = [sys.executable, os.path.join(REV, "scripts", "llm_zeroshot_local.py"),
       "--model", "Qwen/Qwen2.5-7B-Instruct", "--load_4bit",
       "--out", os.path.join(REV, "preds_llm", "qwen25_7b_zeroshot.jsonl")]
print(" ".join(cmd), flush=True)
r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
print("qwen7b rc", r.returncode, flush=True)
print("QUEUE DONE", flush=True)
