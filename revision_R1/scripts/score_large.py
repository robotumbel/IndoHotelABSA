# -*- coding: utf-8 -*-
"""E11 (R2.2): large encoders from Kaggle (preds_large/) next to the base encoders.
Writes results/E11_large_encoders.md
"""
import os, random, statistics, sys
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import score_llm_blind as S

REV = S.REV
SEEDS = [42, 123, 7, 2024, 99]


def runs(d, sid):
    return [{r: S.cells(o["labels"]) for r, o in S.load(os.path.join(REV, d, f"{sid}_{s}.jsonl")).items()} for s in SEEDS]


def mentioned(p):
    f = []
    for k in S.POLS:
        tp = sum(S.gold[r][a] == k and p[r][a] == k for r in S.IDS for a in S.ASPECTS)
        fp = sum(S.gold[r][a] in S.POLS and S.gold[r][a] != k and p[r][a] == k for r in S.IDS for a in S.ASPECTS)
        fn = sum(S.gold[r][a] == k and p[r][a] != k for r in S.IDS for a in S.ASPECTS)
        f.append(S.f1(tp, fp, fn))
    return sum(f) / 3


def ms(x): return f"{statistics.mean(x):.3f}±{statistics.pstdev(x):.3f}"


SYS = [("XLM-RoBERTa base", "preds_seeds", "xlmr"), ("XLM-RoBERTa large", "preds_large", "xlmr_large"),
       ("IndoBERT base", "preds_seeds", "indobert"), ("IndoBERT large", "preds_large", "indobert_large"),
       ("HI-ABSA (base)", "preds_seeds", "hiabsa")]
R = {n: runs(d, s) for n, d, s in SYS}
L = ["# E11 — Large encoders (Kaggle, Tesla T4) vs base encoders, 5 seeds", "",
     "Large: fp16 autocast, micro-batch 8 × 2 accumulation, lr 1e-5, transformers 5.16.1 (Kaggle). "
     "Base: fp32, batch 16, lr 2e-5, transformers 4.57.6 (local RTX 4050). Everything else identical.", "",
     "| System | ACD-mF1 | Mentioned-mF1 | Joint-F1 test-400 | blind-100 vs gold | blind-100 vs blind |",
     "|---|---:|---:|---:|---:|---:|"]
for n, P in R.items():
    L.append(f"| {n} | {ms([S.acd(S.gold, p, S.IDS) for p in P])} | {ms([mentioned(p) for p in P])} | "
             f"{ms([S.joint(S.gold, p, S.IDS) for p in P])} | {ms([S.joint(S.gold, p, S.ids_b, S.ties) for p in P])} | "
             f"{ms([S.joint(S.bgold, p, S.ids_b, S.ties) for p in P])} |")

# seed-matched review-level bootstrap of Joint-F1 differences (test-400)
B = 2000


def gap(a, b):
    out = []
    for i, s in enumerate(SEEDS):
        rg = random.Random(s)
        pa, pb = R[a][i], R[b][i]
        obs = S.joint(S.gold, pa, S.IDS) - S.joint(S.gold, pb, S.IDS)
        d = sorted(S.joint(S.gold, pa, smp) - S.joint(S.gold, pb, smp)
                   for smp in ([rg.choice(S.IDS) for _ in S.IDS] for _ in range(B)))
        out.append((obs, d[int(.025 * B)], d[int(.975 * B)]))
    return out


L += ["", "Seed-matched review-level bootstrap of Joint-F1 differences (test-400):", "",
      "| Comparison | Δ mean | Δ range | seeds with 95% CI excluding 0 |", "|---|---:|---|---:|"]
for a, b in [("IndoBERT large", "IndoBERT base"), ("XLM-RoBERTa large", "XLM-RoBERTa base"),
             ("IndoBERT large", "XLM-RoBERTa large"), ("IndoBERT large", "HI-ABSA (base)")]:
    g = gap(a, b); d = [x[0] for x in g]
    L.append(f"| {a} − {b} | {statistics.mean(d):+.3f} | [{min(d):+.3f}, {max(d):+.3f}] | "
             f"{sum(x[1] > 0 or x[2] < 0 for x in g)}/5 |")
open(os.path.join(REV, "results", "E11_large_encoders.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
