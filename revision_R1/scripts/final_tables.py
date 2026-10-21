# -*- coding: utf-8 -*-
"""Consolidated numbers for the revised manuscript. Writes LaTeX table bodies to
results/latex/*.tex and a flat results/final_numbers.json, so every number in the
manuscript can be traced to one script.
"""
import json, os, random, statistics, sys
import numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import score_llm_blind as S

REV, ROOT = S.REV, S.ROOT
OUT = os.path.join(REV, "results", "latex"); os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 123, 7, 2024, 99]
CI = {"NA": 0, "POSITIF": 1, "NEGATIF": 2, "NETRAL": 3}


def runs(d, sid):
    return [{r: S.cells(o["labels"]) for r, o in S.load(os.path.join(REV, d, f"{sid}_{s}.jsonl")).items()} for s in SEEDS]


def single(path, field="labels"):
    return [{r: S.cells(o.get(field)) for r, o in S.load(path).items()}]


def mentioned(p):
    f = []
    for k in S.POLS:
        tp = sum(S.gold[r][a] == k and p[r][a] == k for r in S.IDS for a in S.ASPECTS)
        fp = sum(S.gold[r][a] in S.POLS and S.gold[r][a] != k and p[r][a] == k for r in S.IDS for a in S.ASPECTS)
        fn = sum(S.gold[r][a] == k and p[r][a] != k for r in S.IDS for a in S.ASPECTS)
        f.append(S.f1(tp, fp, fn))
    return sum(f) / 3, f[2]


def pr(p):
    t = sum(S.gold[r][a] != "NA" and S.gold[r][a] == p[r][a] for r in S.IDS for a in S.ASPECTS)
    fp = sum(p[r][a] != "NA" and S.gold[r][a] != p[r][a] for r in S.IDS for a in S.ASPECTS)
    fn = sum(S.gold[r][a] != "NA" and S.gold[r][a] != p[r][a] for r in S.IDS for a in S.ASPECTS)
    return t / (t + fp), t / (t + fn)


def goldaspect(sid):
    if not os.path.exists(os.path.join(REV, "logits_e8", f"{sid}_42.npz")): return None
    out = []
    for s in SEEDS:
        z = np.load(os.path.join(REV, "logits_e8", f"{sid}_{s}.npz"))
        y = np.array([[CI[S.gold[int(r)][a]] for a in S.ASPECTS] for r in z["test_ids"]])
        p = z["test_logits"][:, :, 1:].argmax(-1) + 1; m = y > 0
        fs = [S.f1(((y[m] == k) & (p[m] == k)).sum(), ((y[m] != k) & (p[m] == k)).sum(), ((y[m] == k) & (p[m] != k)).sum())
              for k in (1, 2, 3)]
        out.append(float(np.mean(fs)))
    return out


Q = os.path.join(REV, "preds_llm")
SYSTEMS = [  # label, list of prediction dicts, logits id for gold-aspect, group
    ("Lexicon baseline", single(os.path.join(ROOT, "pred_lex_gold.jsonl")), None, "Classical"),
    ("mBERT (base)", runs("preds_seeds", "mbert"), "mbert", "Multilingual encoders"),
    ("XLM-RoBERTa (base)", runs("preds_seeds", "xlmr"), "xlmr", "Multilingual encoders"),
    ("XLM-RoBERTa (large)", runs("preds_large", "xlmr_large"), None, "Multilingual encoders"),
    ("IndoBERTweet (base)", runs("preds_seeds", "indobertweet"), "indobertweet", "Indonesian encoders"),
    ("IndoBERT (base)", runs("preds_seeds", "indobert"), "indobert", "Indonesian encoders"),
    ("IndoBERT (large)", runs("preds_large", "indobert_large"), None, "Indonesian encoders"),
    ("HI-ABSA (base)", runs("preds_seeds", "hiabsa"), "hiabsa", "Indonesian encoders"),
    ("mT5-small (seq2seq)", runs("preds_seeds", "mt5"), None, "Generative / LLM"),
    ("Qwen2.5-0.5B LoRA", runs("preds_seeds", "lora_qwen05b"), None, "Generative / LLM"),
    ("Qwen2.5-7B zero-shot$^{\\ddagger}$", single(os.path.join(Q, "qwen25_7b_zeroshot.jsonl"), "labels_lenient"), None, "Generative / LLM"),
]
for tag, lab in [("qwen25_14b_zeroshot", "Qwen2.5-14B zero-shot$^{\\ddagger}$"), ("qwen25_32b_zeroshot", "Qwen2.5-32B zero-shot$^{\\ddagger}$")]:
    p = os.path.join(REV, "preds_qwen_scale", f"{tag}.jsonl")
    SYSTEMS.append((lab, single(p, "labels_lenient") if os.path.exists(p) and len(S.load(p)) == 400 else None, None, "Generative / LLM"))
SYSTEMS.append(("Gemini silver draft$^{\\dagger}$", single(os.path.join(ROOT, "pred_gemini_fewshot.jsonl")), None, "Generative / LLM"))


def fmt(xs, d=3):
    if xs is None: return "--"
    return f"{statistics.mean(xs):.{d}f}" if len(xs) == 1 else f"{statistics.mean(xs):.{d}f}$\\pm${statistics.pstdev(xs):.{d}f}"


rows, NUM, grp = [], {}, None
for name, P, gid, g in SYSTEMS:
    if g != grp:
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{g}}}}} \\\\"); grp = g
    if P is None:
        rows.append(f"\\quad {name} & \\multicolumn{{7}}{{c}}{{[pending]}} \\\\"); continue
    acd = [S.acd(S.gold, p, S.IDS) for p in P]; ga = goldaspect(gid) if gid else None
    men = [mentioned(p)[0] for p in P]; P_, R_ = zip(*[pr(p) for p in P])
    jf = [S.joint(S.gold, p, S.IDS) for p in P]; bl = [S.joint(S.bgold, p, S.ids_b, S.ties) for p in P]
    NUM[name] = {"n_runs": len(P), "ACD": acd, "goldaspect_mF1": ga, "mentioned_mF1": men, "P": P_, "R": R_,
                 "Joint": jf, "blind_vs_blind": bl, "blind_vs_gold": [S.joint(S.gold, p, S.ids_b, S.ties) for p in P],
                 "neu_mentioned": [mentioned(p)[1] for p in P]}
    rows.append(f"\\quad {name} & {fmt(acd)} & {fmt(ga)} & {fmt(men)} & {fmt(list(P_))} & {fmt(list(R_))} & "
                f"{fmt(jf)} & {fmt(bl)} \\\\")
open(os.path.join(OUT, "main_table_rows.tex"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
json.dump({k: {kk: (list(vv) if isinstance(vv, (list, tuple)) else vv) for kk, vv in v.items()} for k, v in NUM.items()},
          open(os.path.join(REV, "results", "final_numbers.json"), "w"), indent=1)
print("\n".join(rows))
