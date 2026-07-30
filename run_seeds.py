# -*- coding: utf-8 -*-
"""#4 Multiple-seed robustness: latih IndoBERT & HI-ABSA pada 5 seed,
laporkan mean +/- std Joint-F1 pada gold test set."""
import subprocess, sys, os, json, statistics
os.chdir(r"D:\Code-PhD\Churn Prediction\paper2_sentiment")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from evaluate_absa import load_jsonl, ASPECTS

PY = sys.executable
SEEDS = [42, 123, 7, 2024, 99]
GOLD = "test_gold.jsonl"
RUNS = [
    ("IndoBERT", "finetune_indobert_absa.py", []),
    ("HI-ABSA",  "finetune_hiabsa.py", []),
]

def joint_f1(pred_path):
    g = load_jsonl(GOLD); p = load_jsonl(pred_path)
    tp = fp = fn = 0
    for rid in g:
        gg = g[rid]["pairs"]; pp = p.get(rid, {}).get("pairs", set())
        tp += len(gg & pp); fp += len(pp - gg); fn += len(gg - pp)
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
    return 2*prec*rec/(prec+rec) if prec+rec else 0

results = {}
for name, script, extra in RUNS:
    scores = []
    for s in SEEDS:
        out = f"pred_seed_{name.lower().replace('-','')}_{s}.jsonl"
        cmd = [PY, script, "--train", "train_silver.jsonl", "--val", "val_gold.jsonl",
               "--test", "test_gold.jsonl", "--epochs", "4", "--seed", str(s),
               "--pred_out", out] + extra
        print(f"[{name} seed={s}] running ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED seed={s}\n{r.stderr[-800:]}", flush=True); continue
        f1 = joint_f1(out); scores.append(f1)
        print(f"  seed={s} Joint-F1={f1:.4f}", flush=True)
    if scores:
        results[name] = {"seeds": SEEDS[:len(scores)], "scores": scores,
                         "mean": statistics.mean(scores),
                         "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0}

print("\n==== MULTI-SEED SUMMARY (Joint-F1, gold test) ====", flush=True)
for name, r in results.items():
    print(f"{name}: mean={r['mean']:.4f} +/- {r['std']:.4f}  (n={len(r['scores'])}) "
          f"scores={[round(x,4) for x in r['scores']]}", flush=True)
json.dump(results, open("seeds_results.json", "w"), indent=2)
print("saved seeds_results.json", flush=True)
