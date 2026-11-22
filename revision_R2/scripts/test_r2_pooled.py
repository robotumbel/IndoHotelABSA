# -*- coding: utf-8 -*-
"""Per-seed detail of family A and a pooled test over the five seeds: per-review counts are
summed over seeds before resampling, so the test asks whether there is a systematic effect
of the training set, averaged over training randomness. Appends to results/R2_grouped_tests.md"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_r2_grouped as T
L = ["## Per-seed detail, family A (grouped minus random drop)", "",
     "| System | " + " | ".join(f"seed {s}" for s in T.SEEDS) + " |", "|---|" + "---:|" * 5]
for n, _ in T.SYS:
    xs = sorted([x for x in T.fam["A"] if x[1] == n], key=lambda x: T.SEEDS.index(x[0]))
    L.append(f"| {n} | " + " | ".join(f"{x[2]:+.3f}{'*' if x[6] < .05 else ''}" for x in xs) + " |")
L += ["", "\* Holm p < .05.", "",
      "## Pooled over five seeds (per-review counts summed over seeds), Holm over the five systems", "",
      "| Comparison | System | Δ (pooled) | 95% CI | p | Holm p |", "|---|---|---:|---|---:|---:|"]
for key, a, b in (("A grouped − random", "grouped", "randdrop"), ("B grouped − standard", "grouped", "standard")):
    rows = []
    for n, sid in T.SYS:
        ca = sum(T.counts(a, sid, s) for s in T.SEEDS); cb = sum(T.counts(b, sid, s) for s in T.SEEDS)
        rows.append((n, *T.test(ca, cb, 1)))
    h = T.holm(np.array([r[4] for r in rows]))
    L += [f"| {key} | {n} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.4f} | {hh:.4f} |" for (n, d, lo, hi, p), hh in zip(rows, h)]
rows = []
for n, sid in T.SYS:
    if sid == "hiabsa": continue
    ca = sum(T.counts("grouped", "hiabsa", s) for s in T.SEEDS); cb = sum(T.counts("grouped", sid, s) for s in T.SEEDS)
    rows.append((n, *T.test(ca, cb, 1)))
h = T.holm(np.array([r[4] for r in rows]))
L += [f"| C HI-ABSA − system (grouped) | {n} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.4f} | {hh:.4f} |" for (n, d, lo, hi, p), hh in zip(rows, h)]
f = os.path.join(T.REV2, "results", "R2_grouped_tests.md")
open(f, "a", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
