# -*- coding: utf-8 -*-
"""R2 point 2: significance tests for Table 18 (hotel-grouped split).
Same protocol as the paper (Section 4.6): seed-matched comparisons, paired bootstrap
(2,000 resamples of whole reviews) for 95% CIs, paired sign-flip permutation test
(5,000 permutations over whole reviews) for p-values, Holm correction within each seed
over the comparisons of one family. Writes results/R2_grouped_tests.md

Families
  A  grouped vs random drop (same size)      -> does hotel separation matter?
  B  grouped vs standard (full) training     -> effect of the 350 dropped reviews
  C  HI-ABSA vs each other encoder, grouped  -> is the ranking kept on the grouped split?
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0]]
import score_llm_blind as S

sys.stdout.reconfigure(encoding="utf-8")
REV2, ROOT = S.REV, S.ROOT
R1 = os.path.join(ROOT, "submission_NLP_Journal", "revision_R1")
SEEDS = [42, 123, 7, 2024, 99]
B, P = 2000, 5000
ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas", "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
IDS = S.IDS
gold = S.gold


def path(kind, sid, s):
    if kind == "standard":
        return os.path.join(R1, "preds_seeds", f"{sid}_{s}.jsonl")
    base = R1 if sid in ("indobert", "hiabsa") else REV2
    sub = "preds_e4" if sid in ("indobert", "hiabsa") else "preds_r2"
    return os.path.join(base, sub, f"{kind}_{sid}_{s}.jsonl")


def counts(kind, sid, s):
    """Per-review (tp, fp, fn) for Joint-F1 over (aspect, polarity) pairs."""
    pred = {r: S.cells(o["labels"]) for r, o in S.load(path(kind, sid, s)).items()}
    out = np.zeros((len(IDS), 3))
    for i, r in enumerate(IDS):
        for a in ASPECTS:
            g, p = gold[r][a], pred[r][a]
            if g != "NA" and g == p:
                out[i, 0] += 1
            else:
                out[i, 1] += p != "NA"
                out[i, 2] += g != "NA"
    return out


def f1(c):  # c: (..., 3) summed counts
    t, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    return 2 * t / np.maximum(2 * t + fp + fn, 1e-12)


def test(ca, cb, seed):
    rg = np.random.default_rng(seed)
    n = len(ca)
    obs = f1(ca.sum(0)) - f1(cb.sum(0))
    idx = rg.integers(0, n, size=(B, n))
    d = f1(ca[idx].sum(1)) - f1(cb[idx].sum(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    sw = rg.random((P, n)) < .5                                   # swap the two systems for these reviews
    A = np.where(sw[..., None], cb[None], ca[None]).sum(1)
    Bc = np.where(sw[..., None], ca[None], cb[None]).sum(1)
    ext = (np.abs(f1(A) - f1(Bc)) >= abs(obs) - 1e-12).sum()
    return obs, lo, hi, (ext + 1) / (P + 1)


def holm(ps):
    order = np.argsort(ps); out = np.zeros(len(ps)); m = 0
    for k, i in enumerate(order):
        m = max(m, min(1, (len(ps) - k) * ps[i])); out[i] = m
    return out


SYS = [("mBERT", "mbert"), ("XLM-RoBERTa", "xlmr"), ("IndoBERTweet", "indobertweet"),
       ("IndoBERT", "indobert"), ("HI-ABSA", "hiabsa")]
fam = {"A": [], "B": [], "C": []}
for s in SEEDS:
    for key, pairs in (("A", [(n, ("grouped", sid), ("randdrop", sid)) for n, sid in SYS]),
                       ("B", [(n, ("grouped", sid), ("standard", sid)) for n, sid in SYS]),
                       ("C", [(n, ("grouped", "hiabsa"), ("grouped", sid)) for n, sid in SYS if sid != "hiabsa"])):
        rows = [(n, *test(counts(*a, s), counts(*b, s), s)) for n, a, b in pairs]
        h = holm(np.array([r[4] for r in rows]))
        fam[key] += [(s, *r, hh) for r, hh in zip(rows, h)]
    print("seed", s, "done", flush=True)

title = {"A": "A. Grouped minus random drop (equal size, 2,130 reviews): effect of hotel separation",
         "B": "B. Grouped (2,130) minus standard (2,480): effect of the smaller training set",
         "C": "C. HI-ABSA minus each encoder, all trained on the grouped split: is the ranking kept?"}
L = ["# R2: significance tests for the hotel-grouped split (Joint-F1, test-400)", "",
     f"Seed-matched comparisons; paired bootstrap ({B}) for the 95% CI and paired sign-flip permutation "
     f"({P}) for p, both over whole reviews; Holm correction within each seed and family. "
     f"Smallest attainable p = {1/(P+1):.4f}.", ""]
for k in "ABC":
    L += [f"## {title[k]}", "", "| System | Δ mean | Δ range over seeds | 95% CI (widest) | Holm p range | seeds significant (Holm < .05) |",
          "|---|---:|---|---|---|---:|"]
    for n, _ in SYS:
        xs = [x for x in fam[k] if x[1] == n]
        if not xs: continue
        d = [x[2] for x in xs]
        L.append(f"| {n} | {np.mean(d):+.3f} | [{min(d):+.3f}, {max(d):+.3f}] | [{min(x[3] for x in xs):+.3f}, {max(x[4-0] for x in xs):+.3f}] | "
                 f"{min(x[6] for x in xs):.4f}–{max(x[6] for x in xs):.4f} | {sum(x[6] < .05 for x in xs)}/5 |")
    L.append("")
open(os.path.join(REV2, "results", "R2_grouped_tests.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
