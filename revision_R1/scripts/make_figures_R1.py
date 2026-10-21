# -*- coding: utf-8 -*-
"""Figures for the revised manuscript, from the 5-seed predictions. Writes PDFs into
revision_R1/manuscript/ and prints every number quoted in the text next to a figure.
"""
import json, math, os, statistics, sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import score_llm_blind as S

REV, ROOT = S.REV, S.ROOT
OUT = os.path.join(REV, "manuscript")
SEEDS = [42, 123, 7, 2024, 99]
NAVY, ORANGE, GREEN, GREY, TEAL, RED = "#143a5c", "#b25a00", "#1b5e20", "#8a8a8a", "#2a9d8f", "#b0202a"
EN = ["Location", "Cleanliness", "Service", "Room & Fac.", "Price", "Food & Bev.", "Public Fac."]
L4 = ["NA", "POSITIF", "NEGATIF", "NETRAL"]


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


gem = [{r: S.cells(o["labels"]) for r, o in S.load(os.path.join(ROOT, "pred_gemini_fewshot.jsonl")).items()}]
q7 = [{r: S.cells(o["labels_lenient"]) for r, o in S.load(os.path.join(REV, "preds_llm", "qwen25_7b_zeroshot.jsonl")).items()}]
lex = [{r: S.cells(o["labels"]) for r, o in S.load(os.path.join(ROOT, "pred_lex_gold.jsonl")).items()}]
SYS = [("Lexicon", lex, GREY), ("mBERT", runs("preds_seeds", "mbert"), NAVY), ("XLM-R", runs("preds_seeds", "xlmr"), NAVY),
       ("XLM-R\nlarge", runs("preds_large", "xlmr_large"), NAVY), ("IndoBERT-\ntweet", runs("preds_seeds", "indobertweet"), NAVY),
       ("IndoBERT", runs("preds_seeds", "indobert"), NAVY), ("IndoBERT\nlarge", runs("preds_large", "indobert_large"), NAVY),
       ("HI-ABSA", runs("preds_seeds", "hiabsa"), ORANGE), ("mT5", runs("preds_seeds", "mt5"), TEAL),
       ("LoRA\n0.5B", runs("preds_seeds", "lora_qwen05b"), TEAL), ("Qwen2.5\n7B ZS", q7, GREEN)]
for tag, lab in [("qwen25_14b_zeroshot", "Qwen2.5\n14B ZS"), ("qwen25_32b_zeroshot", "Qwen2.5\n32B ZS")]:
    p = os.path.join(REV, "preds_qwen_scale", f"{tag}.jsonl")
    if os.path.exists(p) and len(S.load(p)) == 400:
        SYS.append((lab, [{r: S.cells(o["labels_lenient"]) for r, o in S.load(p).items()}], GREEN))
SYS.append(("Gemini\ndraft$^\\dagger$", gem, RED))

# ---------------- Fig: model comparison (gold test-400 vs blind-100)
names = [n for n, _, _ in SYS]; x = np.arange(len(SYS)); w = 0.27
def mstat(fn, P): v = [fn(p) for p in P]; return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else 0)
acd = [mstat(lambda p: S.acd(S.gold, p, S.IDS), P) for _, P, _ in SYS]
jt = [mstat(lambda p: S.joint(S.gold, p, S.IDS), P) for _, P, _ in SYS]
bl = [mstat(lambda p: S.joint(S.bgold, p, S.ids_b, S.ties), P) for _, P, _ in SYS]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.bar(x - w, [a[0] for a in acd], w, yerr=[a[1] for a in acd], label="ACD macro-F1 (test-400)", color=NAVY, capsize=1.5)
ax.bar(x, [a[0] for a in jt], w, yerr=[a[1] for a in jt], label="Joint-F1 (test-400, gold)", color=ORANGE, capsize=1.5)
ax.bar(x + w, [a[0] for a in bl], w, yerr=[a[1] for a in bl], label="Joint-F1 (blind-100, blind labels)", color=GREEN, capsize=1.5)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel("Score"); ax.set_ylim(0.3, 1.0); ax.legend(fontsize=7, ncol=3, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_model_multi.pdf")); plt.close()
print("fig_model_multi:", [(n.replace("\n", " "), round(j[0], 3), round(b[0], 3)) for n, j, b in zip(names, jt, bl)])

# ---------------- Fig: confusion matrix HI-ABSA, summed over 5 seeds (shown) and seed 42 counts printed
hi = runs("preds_seeds", "hiabsa")
M = np.zeros((4, 4), int)
for p in hi:
    for r in S.IDS:
        for a in S.ASPECTS:
            M[L4.index(S.gold[r][a]), L4.index(p[r][a])] += 1
norm = M / M.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
lab = ["N/A", "POS", "NEG", "NEU"]
ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(lab); ax.set_yticklabels(lab)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.set_title("HI-ABSA, test-400, 5 seeds pooled", fontsize=10)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{M[i, j]/5:.1f}\n{norm[i, j]*100:.0f}%", ha="center", va="center", fontsize=9,
                color="white" if norm[i, j] > 0.55 else "#222222")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("row-normalised (recall)", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_confusion.pdf"), bbox_inches="tight"); plt.close()
print("confusion (mean per seed):\n", np.round(M / 5, 1), "\nrecall %:", np.round(np.diag(norm) * 100, 1))

# ---------------- Fig: per-aspect detection F1, HI-ABSA and IndoBERT (5-seed means)
ib = runs("preds_seeds", "indobert")
def per_aspect(P):
    out = []
    for a in S.ASPECTS:
        v = []
        for p in P:
            tp = sum(S.gold[r][a] != "NA" and p[r][a] != "NA" for r in S.IDS)
            fp = sum(S.gold[r][a] == "NA" and p[r][a] != "NA" for r in S.IDS)
            fn = sum(S.gold[r][a] != "NA" and p[r][a] == "NA" for r in S.IDS)
            v.append(S.f1(tp, fp, fn))
        out.append((statistics.mean(v), statistics.pstdev(v)))
    return out
pa_hi, pa_ib = per_aspect(hi), per_aspect(ib)
order = sorted(range(7), key=lambda i: pa_hi[i][0])
fig, ax = plt.subplots(figsize=(5.4, 3.2)); y = np.arange(7); h = 0.38
ax.barh(y + h / 2, [pa_hi[i][0] for i in order], h, xerr=[pa_hi[i][1] for i in order], color=ORANGE, label="HI-ABSA")
ax.barh(y - h / 2, [pa_ib[i][0] for i in order], h, xerr=[pa_ib[i][1] for i in order], color=NAVY, label="IndoBERT")
ax.set_yticks(y); ax.set_yticklabels([EN[i] for i in order], fontsize=8)
ax.set_xlabel("Detection F1 (mean of 5 seeds)"); ax.set_xlim(0.6, 1.0); ax.legend(fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_per_aspect.pdf")); plt.close()
print("per-aspect HI-ABSA:", {EN[i]: round(pa_hi[i][0], 3) for i in range(7)})
print("per-aspect IndoBERT:", {EN[i]: round(pa_ib[i][0], 3) for i in range(7)})

# ---------------- Fig: error breakdown (mean per seed) HI-ABSA, IndoBERT, Qwen-7B, Gemini
def errors(p):
    c = Counter()
    for r in S.IDS:
        for a in S.ASPECTS:
            g, q = S.gold[r][a], p[r][a]
            if g == q: continue
            if g == "NA": c["spurious"] += 1
            elif q == "NA": c["missed"] += 1
            else: c["polarity"] += 1; c[f"{g}->{q}"] += 1
    return c
def merr(P):
    cs = [errors(p) for p in P]; keys = set().union(*cs)
    return {k: statistics.mean(c[k] for c in cs) for k in keys}
E = {"HI-ABSA": merr(hi), "IndoBERT": merr(ib), "Qwen2.5-7B ZS": merr(q7), "Gemini draft": merr(gem)}
cats = ["missed", "spurious", "polarity"]
fig, ax = plt.subplots(figsize=(5.6, 3.0)); xx = np.arange(3); w = 0.2
for k, (n, col) in enumerate(zip(E, [ORANGE, NAVY, GREEN, RED])):
    vals = [E[n].get(c, 0) for c in cats]
    ax.bar(xx + (k - 1.5) * w, vals, w, label=n, color=col)
ax.set_xticks(xx); ax.set_xticklabels(["Missed aspect", "Spurious aspect", "Wrong polarity"], fontsize=9)
ax.set_ylabel("Error cells (of 2,800)"); ax.legend(fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_error_breakdown.pdf")); plt.close()
for n, e in E.items():
    tot = sum(e.get(c, 0) for c in cats)
    print("errors", n, {c: round(e.get(c, 0), 1) for c in cats}, "total", round(tot, 1),
          "shares", {c: round(100 * e.get(c, 0) / tot) for c in cats},
          "top flips", sorted(((k, round(v, 1)) for k, v in e.items() if "->" in k), key=lambda t: -t[1])[:3])

# ---------------- Fig: dashboard, gold labels and HI-ABSA, balanced vs post-stratified
corpus = S.load(os.path.join(ROOT, "dataset_merged.jsonl")); test = S.load(os.path.join(ROOT, "test_gold.jsonl"))
pc = Counter(int(o["rating"]) for o in corpus.values()); pt = Counter(int(o["rating"]) for o in test.values())
wts = {r: (pc[int(o["rating"])] / len(corpus)) / (pt[int(o["rating"])] / len(test)) for r, o in test.items()}
def net(lbl, wt=None):
    out = {}
    for a in S.ASPECTS:
        c = Counter()
        for r in S.IDS: c[lbl[r][a]] += 1.0 if wt is None else wt[r]
        d = c["POSITIF"] + c["NEGATIF"] + c["NETRAL"]; out[a] = (c["POSITIF"] - c["NEGATIF"]) / d
    return out
hi42 = hi[0]
panels = [("gold, rating-stratified test set", net(S.gold)), ("gold, re-weighted to corpus ratings", net(S.gold, wts)),
          ("HI-ABSA, rating-stratified", net(hi42)), ("HI-ABSA, re-weighted", net(hi42, wts))]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
order = sorted(range(7), key=lambda i: panels[0][1][S.ASPECTS[i]])
for ax, (t1, a1), (t2, a2), title in [(axes[0], panels[0], panels[1], "(a) Gold labels"), (axes[1], panels[2], panels[3], "(b) HI-ABSA predictions (seed 42)")]:
    y = np.arange(7); h = 0.38
    ax.barh(y + h / 2, [a1[S.ASPECTS[i]] for i in order], h, color=GREY, label="rating-stratified test set")
    ax.barh(y - h / 2, [a2[S.ASPECTS[i]] for i in order], h, color=NAVY, label="re-weighted to corpus ratings")
    ax.axvline(0, color="k", lw=0.8); ax.set_xlim(-1, 1); ax.set_xlabel("Net sentiment $s_a$")
    ax.set_yticks(y); ax.set_yticklabels([EN[i] for i in order], fontsize=8); ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_dashboard.pdf"), bbox_inches="tight"); plt.close()
for t, v in panels: print("dashboard", t, {EN[i]: round(v[S.ASPECTS[i]], 2) for i in range(7)})
