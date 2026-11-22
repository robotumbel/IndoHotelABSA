# -*- coding: utf-8 -*-
"""R2 point 2: all five trained systems on the standard (random review-level) split vs the
official hotel-grouped split. Test-400 Joint-F1, five seeds, plus the 186 unseen-hotel reviews.
Writes results/R2_grouped_all.md"""
import os, statistics, sys
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import score_llm_blind as S
REV2, ROOT = S.REV, S.ROOT
R1 = os.path.join(ROOT, "submission_NLP_Journal", "revision_R1")
SEEDS = [42, 123, 7, 2024, 99]
hk = lambda o: (o["hotel"].strip().lower(), o["city"])
train, test = S.load(os.path.join(ROOT, "train_silver.jsonl")), S.load(os.path.join(ROOT, "test_gold.jsonl"))
seen = {hk(o) for o in train.values()}
UNSEEN = [r for r in S.IDS if hk(test[r]) not in seen]

def runs(path_fmt):
    out = []
    for s in SEEDS:
        p = path_fmt.format(s=s)
        if os.path.exists(p):
            out.append({r: S.cells(o["labels"]) for r, o in S.load(p).items()})
    return out

ms = lambda xs: f"{statistics.mean(xs):.3f}±{statistics.pstdev(xs):.3f}" if xs else "—"
SYS = [("mBERT", "mbert", "grouped_mbert"), ("XLM-RoBERTa", "xlmr", "grouped_xlmr"),
       ("IndoBERTweet", "indobertweet", "grouped_indobertweet"),
       ("IndoBERT", "indobert", "grouped_indobert"), ("HI-ABSA", "hiabsa", "grouped_hiabsa")]
L = ["# R2 — standard split vs official hotel-grouped split, all trained encoders (5 seeds)", "",
     "| System | standard (2,480) | random drop (2,130) | grouped (2,130) | grouped − random | standard, unseen-hotel (186) | grouped, unseen-hotel (186) | runs std/rand/grp |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for name, std, grp in SYS:
    A = runs(os.path.join(R1, "preds_seeds", std + "_{s}.jsonl"))
    gdir = os.path.join(R1 if grp in ("grouped_indobert", "grouped_hiabsa") else REV2, "preds_e4" if grp in ("grouped_indobert", "grouped_hiabsa") else "preds_r2")
    B = runs(os.path.join(gdir, grp + "_{s}.jsonl"))
    rid = grp.replace("grouped_", "randdrop_")
    rdir = os.path.join(R1 if rid in ("randdrop_indobert", "randdrop_hiabsa") else REV2, "preds_e4" if rid in ("randdrop_indobert", "randdrop_hiabsa") else "preds_r2")
    C = runs(os.path.join(rdir, rid + "_{s}.jsonl"))
    fc = [S.joint(S.gold, p, S.IDS) for p in C]
    fa = [S.joint(S.gold, p, S.IDS) for p in A]; fb = [S.joint(S.gold, p, S.IDS) for p in B]
    ua = [S.joint(S.gold, p, UNSEEN) for p in A]; ub = [S.joint(S.gold, p, UNSEEN) for p in B]
    d = f"{statistics.mean(fb)-statistics.mean(fc):+.3f}" if fc and fb else "—"
    L.append(f"| {name} | {ms(fa)} | {ms(fc)} | {ms(fb)} | {d} | {ms(ua)} | {ms(ub)} | {len(A)}/{len(C)}/{len(B)} |")
os.makedirs(os.path.join(REV2, "results"), exist_ok=True)
open(os.path.join(REV2, "results", "R2_grouped_all.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
