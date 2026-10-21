# -*- coding: utf-8 -*-
"""Resume order after the 20 Oct stop: LoRA seeds (fast), Qwen-7B zero-shot, then remaining mT5 seeds (~2.4 h each)."""
import os, subprocess, sys
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
steps = [
    [sys.executable, "-u", os.path.join(REV, "scripts", "run_seeds_all.py"), "--systems", "lora_qwen05b"],
    [sys.executable, "-u", os.path.join(REV, "scripts", "llm_zeroshot_local.py"), "--model", "Qwen/Qwen2.5-7B-Instruct",
     "--load_4bit", "--embed_on_cpu", "--out", os.path.join(REV, "preds_llm", "qwen25_7b_zeroshot.jsonl")],
    [sys.executable, "-u", os.path.join(REV, "scripts", "run_seeds_all.py"), "--systems", "mt5"],
]
for cmd in steps:
    print(">>>", " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    print("rc", r.returncode, flush=True)
print("QUEUE DONE", flush=True)
