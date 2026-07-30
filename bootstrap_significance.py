# -*- coding: utf-8 -*-
"""
bootstrap_significance.py — Paired bootstrap test selisih Joint-F1 antar model.
Resample 400 review test (1.000x), hitung distribusi selisih F1, 95% CI, p-value.
Pemakaian: python bootstrap_significance.py predA.jsonl predB.jsonl
"""
import json, random, sys

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)
GOLD = "test_gold.jsonl"


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
                d[o["review_id"]] = {(l["aspect"], l["polarity"])
                                     for l in o.get("labels", [])}
    return d


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def per_review_counts(gold, pred, ids):
    out = []
    for rid in ids:
        g = gold.get(rid, set()); p = pred.get(rid, set())
        out.append((len(g & p), len(p - g), len(g - p)))
    return out


def joint_f1(counts, idx):
    tp = sum(counts[i][0] for i in idx)
    fp = sum(counts[i][1] for i in idx)
    fn = sum(counts[i][2] for i in idx)
    return f1_from_counts(tp, fp, fn)


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    gold = load(GOLD)
    A, B = load(pa), load(pb)
    ids = sorted(gold)
    ca = per_review_counts(gold, A, ids)
    cb = per_review_counts(gold, B, ids)
    n = len(ids)
    full = list(range(n))
    fa, fb = joint_f1(ca, full), joint_f1(cb, full)
    obs = fa - fb

    diffs = []
    for _ in range(1000):
        idx = [random.randrange(n) for _ in range(n)]
        diffs.append(joint_f1(ca, idx) - joint_f1(cb, idx))
    diffs.sort()
    lo, hi = diffs[24], diffs[974]                      # 95% CI
    # p-value dua arah: proporsi resample yang berlawanan tanda dgn observasi
    if obs >= 0:
        p = 2 * sum(1 for d in diffs if d <= 0) / len(diffs)
    else:
        p = 2 * sum(1 for d in diffs if d >= 0) / len(diffs)
    p = min(p, 1.0)

    print(f"A = {pa}: Joint-F1 {fa:.4f}")
    print(f"B = {pb}: Joint-F1 {fb:.4f}")
    print(f"Selisih (A-B) = {obs:+.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}] | p = {p:.3f}")
    print("Signifikan (p<0.05)?", "YA" if p < 0.05 else "TIDAK")


if __name__ == "__main__":
    main()
