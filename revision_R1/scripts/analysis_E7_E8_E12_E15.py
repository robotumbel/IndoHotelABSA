# -*- coding: utf-8 -*-
"""E8 gold-aspect ACP, E15 neutral calibration (from logits_e8/), E12 faithfulness (faith_e12/),
E7 label-noise study (preds_e7/). Writes results/E8_goldaspect.md, E15_neutral.md, E12_faith.md, E7_noise.md
"""
import json, os, statistics, sys
import numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import score_llm_blind as S

REV, ROOT = S.REV, S.ROOT
SEEDS = [42, 123, 7, 2024, 99]
SYS = [("IndoBERT", "indobert"), ("HI-ABSA", "hiabsa"), ("XLM-RoBERTa", "xlmr"), ("mBERT", "mbert"),
       ("IndoBERTweet", "indobertweet")]
CI = {"NA": 0, "POSITIF": 1, "NEGATIF": 2, "NETRAL": 3}
val = S.load(os.path.join(ROOT, "val_gold.jsonl"))
G = {"test": {r: [CI[S.gold[r][a]] for a in S.ASPECTS] for r in S.IDS},
     "val": {r: [CI[v] for v in S.cells(o["labels"]).values()] for r, o in val.items()}}


def ms(x): return f"{statistics.mean(x):.3f}±{statistics.pstdev(x):.3f}"


def f1(tp, fp, fn): return S.f1(tp, fp, fn)


def load_npz(sid, s):
    z = np.load(os.path.join(REV, "logits_e8", f"{sid}_{s}.npz"))
    return {"test": (z["test_ids"], z["test_logits"]), "val": (z["val_ids"], z["val_logits"])}


def yg(split, ids): return np.array([G[split][int(r)] for r in ids])          # (N, 7)


def pol_metrics(y, p, mask):
    """Polarity metrics on cells selected by mask (y, p in 1..3)."""
    yt, pt = y[mask], p[mask]; out = {"n": int(mask.sum()), "acc": float((yt == pt).mean())}
    fs = []
    for k in (1, 2, 3):
        fs.append(f1(((yt == k) & (pt == k)).sum(), ((yt != k) & (pt == k)).sum(), ((yt == k) & (pt != k)).sum()))
    out["mF1"], out["F1_NEU"] = float(np.mean(fs)), float(fs[2])
    return out


def joint_f1(y, p):
    tp = ((y > 0) & (y == p)).sum(); fp = ((p > 0) & (y != p)).sum(); fn = ((y > 0) & (y != p)).sum()
    return f1(tp, fp, fn)


def cell4(y, p):
    return float(np.mean([f1(((y == k) & (p == k)).sum(), ((y != k) & (p == k)).sum(), ((y == k) & (p != k)).sum())
                          for k in range(4)]))


def mentioned(y, p):
    fs = [f1(((y == k) & (p == k)).sum(), ((y > 0) & (y != k) & (p == k)).sum(), ((y == k) & (p != k)).sum())
          for k in (1, 2, 3)]
    return float(np.mean(fs)), float(fs[2])


# ================================================================ E8 gold-aspect
L = ["# E8 — Polarity under gold aspects vs predicted aspects (5 seeds, test-400)", "",
     "Gold-aspect: every gold-mentioned cell, polarity = argmax over the POS/NEG/NEU logits (N/A logit ignored); "
     "the item set is identical for every system. Predicted-aspect: submitted protocol (cells both gold and model mark).",
     "Logits come from the E8 re-training (IndoBERT and HI-ABSA reproduce E5 predictions on 400/400 reviews in every seed; "
     "XLM-R, mBERT and IndoBERTweet reproduce 308–400/400, GPU non-determinism), so predicted-aspect numbers may differ "
     "slightly from E5 for those three.", "",
     "| System | gold-aspect Acc | gold-aspect mF1 | gold-aspect F1-NEU | pred-aspect Acc | pred-aspect mF1 | pred-aspect items |",
     "|---|---:|---:|---:|---:|---:|---:|"]
E8 = {}
for name, sid in SYS:
    ga, pa = [], []
    for s in SEEDS:
        ids, lg = load_npz(sid, s)["test"]; y = yg("test", ids)
        pg = lg[:, :, 1:].argmax(-1) + 1; pp = lg.argmax(-1)
        ga.append(pol_metrics(y, pg, y > 0)); pa.append(pol_metrics(y, pp, (y > 0) & (pp > 0)))
    E8[name] = (ga, pa)
    L.append(f"| {name} | {ms([x['acc'] for x in ga])} | {ms([x['mF1'] for x in ga])} | {ms([x['F1_NEU'] for x in ga])} | "
             f"{ms([x['acc'] for x in pa])} | {ms([x['mF1'] for x in pa])} | {statistics.mean(x['n'] for x in pa):.0f} |")
L += ["", f"Gold-mentioned test cells: {int((yg('test', S.IDS) > 0).sum())} (of which neutral: "
      f"{int((yg('test', S.IDS) == 3).sum())})."]
open(os.path.join(REV, "results", "E8_goldaspect.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))

# ================================================================ E15 neutral calibration
GRID = np.arange(-1.0, 6.01, 0.25)
L = ["# E15 — Neutral calibration pilot: additive bias on the NEU logit (5 seeds)", "",
     f"Bias b chosen on the 100-review validation set (only {int((np.array(list(G['val'].values())) == 3).sum())} neutral cells) "
     "by maximising 4-class cell macro-F1, grid −1 … 6 step 0.25, then applied unchanged to test-400. "
     "'Oracle' = best b on test itself, an upper bound only.", "",
     "| System | b (val, per seed) | F1-NEU before → after | Mentioned-mF1 before → after | Joint-F1 before → after | oracle F1-NEU |",
     "|---|---|---|---|---|---:|"]
for name, sid in SYS[:2]:
    bs, nb, na, mb, ma, jb, ja, orc = [], [], [], [], [], [], [], []
    for s in SEEDS:
        d = load_npz(sid, s)
        vi, vl = d["val"]; ti, tl = d["test"]; yv, yt = yg("val", vi), yg("test", ti)
        def apply(lg, b): x = lg.copy(); x[:, :, 3] += b; return x.argmax(-1)
        b = max(GRID, key=lambda b: cell4(yv, apply(vl, b))); bs.append(float(b))
        p0, p1 = apply(tl, 0.0), apply(tl, b)
        m0, n0 = mentioned(yt, p0); m1, n1 = mentioned(yt, p1)
        nb.append(n0); na.append(n1); mb.append(m0); ma.append(m1); jb.append(joint_f1(yt, p0)); ja.append(joint_f1(yt, p1))
        orc.append(max(mentioned(yt, apply(tl, bb))[1] for bb in GRID))
    L.append(f"| {name} | {', '.join(f'{b:g}' for b in bs)} | {ms(nb)} → {ms(na)} | {ms(mb)} → {ms(ma)} | "
             f"{ms(jb)} → {ms(ja)} | {ms(orc)} |")
open(os.path.join(REV, "results", "E15_neutral.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))

# ================================================================ E12 faithfulness
L = ["# E12 — Faithfulness of token explanations (5 seeds, test cells mentioned in gold and predicted correctly)", "",
     "Comprehensiveness (higher = more faithful): drop in p(predicted class) after removing the top-k% content tokens. "
     "Sufficiency (lower = more faithful): drop when only the top-k% are kept. AOPC = mean over k ∈ {10, 20, 50}%. "
     "Random = random token ranking (control). Tokens removed through the attention mask; HI-ABSA lexicon input fixed.", "",
     "| Model | Explanation | cells/seed | comp@10 | comp@20 | comp@50 | AOPC comp | AOPC suff |", "|---|---|---:|---:|---:|---:|---:|---:|"]
extra = []
for name, sid in [("HI-ABSA", "hiabsa"), ("IndoBERT", "indobert")]:
    F = [json.load(open(os.path.join(REV, "faith_e12", f"{sid}_{s}.json"))) for s in SEEDS]
    for meth in F[0]["erasure"]:
        g = lambda key: [statistics.mean(f["erasure"][meth][key]) for f in F]
        comp = [statistics.mean(statistics.mean(f["erasure"][meth][f"comp@{k}"]) for k in (0.1, 0.2, 0.5)) for f in F]
        suff = [statistics.mean(statistics.mean(f["erasure"][meth][f"suff@{k}"]) for k in (0.1, 0.2, 0.5)) for f in F]
        L.append(f"| {name} | {meth} | {statistics.mean(f['cells'] for f in F):.0f} | {ms(g('comp@0.1'))} | {ms(g('comp@0.2'))} | "
                 f"{ms(g('comp@0.5'))} | {ms(comp)} | {ms(suff)} |")
    if sid == "hiabsa":
        for kind in ("perm", "uniform"):
            extra.append(f"| HI-ABSA | {kind} attention | {ms([statistics.mean(f[kind]['flip']) for f in F])} | "
                         f"{ms([statistics.mean(f[kind]['tvd']) for f in F])} |")
        agree = (ms([statistics.mean(f['agree']['spearman']) for f in F]), ms([statistics.mean(f['agree']['top3_overlap']) for f in F]))
L += ["", "Replacing the aspect-query attention of HI-ABSA (Jain & Wallace-style test):", "",
      "| Model | Intervention | prediction flip rate | mean total variation of p(class) |", "|---|---|---:|---:|"] + extra
L += ["", f"Agreement between HI-ABSA attention and IG rankings over content tokens: Spearman {agree[0]}, top-3 overlap {agree[1]}."]
open(os.path.join(REV, "results", "E12_faith.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))

# ================================================================ E7 label noise
def pr(tag, s): return {r: S.cells(o["labels"]) for r, o in S.load(os.path.join(REV, "preds_e7", f"{tag}_{s}.jsonl")).items()}


gold500 = {r: S.cells(o["labels"]) for r, o in S.load(os.path.join(ROOT, "gold_final.jsonl")).items()}
L = ["# E7 — Silver vs gold supervision and label noise (IndoBERT-base)", "",
     "## (A) 5-fold CV over the 500 gold reviews (seed 42; test fold 100, val fold 100, gold-train 300)", "",
     "Pooled over the five test folds (all 500 gold reviews scored once); blind columns use the 100 blind-annotated reviews.", "",
     "| Training data | Joint-F1 (500, pooled) | per-fold SD | blind-100 vs gold | blind-100 vs blind |", "|---|---:|---:|---:|---:|"]
for cond, lab in [("silver", "silver only (2,480)"), ("gold", "gold only (300)"), ("silvergold", "silver + gold (2,780)")]:
    pooled, per = {}, []
    for f in range(5):
        p = pr(f"cv{f}_{cond}", 42); pooled.update(p)
        per.append(S.joint(gold500, p, sorted(p)))
    L.append(f"| {lab} | {S.joint(gold500, pooled, sorted(gold500)):.3f} | {statistics.pstdev(per):.3f} | "
             f"{S.joint(gold500, pooled, S.ids_b, S.ties):.3f} | {S.joint(S.bgold, pooled, S.ids_b, S.ties):.3f} |")
E5 = [{r: S.cells(o["labels"]) for r, o in S.load(os.path.join(REV, "preds_seeds", f"indobert_{s}.jsonl")).items()} for s in (42, 123, 7)]
L += ["", "## (B) Synthetic noise on the silver labels and (C) confidence filtering (seeds 42/123/7, test-400)", "",
      "| Training data | reviews | Joint-F1 test-400 | blind-100 vs blind |", "|---|---:|---:|---:|",
      f"| silver, no added noise (E5) | 2,480 | {ms([S.joint(S.gold, p, S.IDS) for p in E5])} | {ms([S.joint(S.bgold, p, S.ids_b, S.ties) for p in E5])} |"]
for tag, lab, n in [("noise05", "+5% cells flipped (853)", "2,480"), ("noise10", "+10% cells flipped (1,697)", "2,480"),
                    ("noise20", "+20% cells flipped (3,540)", "2,480"), ("cl_loose", "filtered, ≤1 OOF disagreement", "1,730"),
                    ("cl_strict", "filtered, 0 OOF disagreements", "822")]:
    P = [pr(tag, s) for s in (42, 123, 7)]
    L.append(f"| {lab} | {n} | {ms([S.joint(S.gold, p, S.IDS) for p in P])} | {ms([S.joint(S.bgold, p, S.ids_b, S.ties) for p in P])} |")
open(os.path.join(REV, "results", "E7_noise.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
