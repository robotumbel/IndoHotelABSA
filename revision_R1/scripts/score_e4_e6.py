# -*- coding: utf-8 -*-
"""Score E4 (grouped vs random-drop vs full training) and E6 (ablation variants), 5 seeds.
Test-400 split into reviews whose hotel occurs in the full training set ('seen') and
the rest ('unseen'). Writes results/E4_grouped.md and results/E6_ablation.md
"""
import json, os, statistics, sys
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import score_llm_blind as S

REV, ROOT = S.REV, S.ROOT
SEEDS = [42, 123, 7, 2024, 99]
hk = lambda o: (o["hotel"].strip().lower(), o["city"])
train = S.load(os.path.join(ROOT, "train_silver.jsonl"))
test = S.load(os.path.join(ROOT, "test_gold.jsonl"))
seen_h = {hk(o) for o in train.values()}
SEEN = [r for r in S.IDS if hk(test[r]) in seen_h]
UNSEEN = [r for r in S.IDS if hk(test[r]) not in seen_h]


def runs(d, sid):
    out = []
    for s in SEEDS:
        p = os.path.join(REV, d, f"{sid}_{s}.jsonl")
        if os.path.exists(p):
            out.append({r: S.cells(o["labels"]) for r, o in S.load(p).items()})
    return out


def ms(xs):
    return f"{statistics.mean(xs):.3f}±{statistics.pstdev(xs):.3f}" if xs else "—"


def row(name, P):
    f = lambda ids: [S.joint(S.gold, p, ids) for p in P]
    return (f"| {name} | {len(P)} | {ms(f(S.IDS))} | {ms(f(SEEN))} | {ms(f(UNSEEN))} | "
            f"{ms([S.joint(S.bgold, p, S.ids_b, S.ties) for p in P])} |")


L = ["# E4 — Hotel-grouped training vs size-matched random drop (5 seeds)", "",
     f"Test-400 split by whether the review's hotel occurs in the full silver training set: seen {len(SEEN)}, unseen {len(UNSEEN)}.",
     "Grouped training removes every val/test hotel (2,130 reviews); random drop keeps 2,130 random reviews.", "",
     "| Training set | runs | Joint-F1 test-400 | seen-hotel reviews | unseen-hotel reviews | blind-100 vs blind |",
     "|---|---:|---:|---:|---:|---:|"]
for model, full in [("IndoBERT", "indobert"), ("HI-ABSA", "hiabsa")]:
    L.append(row(f"{model}, full silver (2,480)", runs("preds_seeds", full)))
    L.append(row(f"{model}, random drop (2,130)", runs("preds_e4", f"randdrop_{full}")))
    L.append(row(f"{model}, hotel-grouped (2,130)", runs("preds_e4", f"grouped_{full}")))
open(os.path.join(REV, "results", "E4_grouped.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))


def acd_aspect(P):
    out = {}
    for a in S.ASPECTS:
        vals = []
        for p in P:
            tp = sum(S.gold[r][a] != "NA" and p[r][a] != "NA" for r in S.IDS)
            fp = sum(S.gold[r][a] == "NA" and p[r][a] != "NA" for r in S.IDS)
            fn = sum(S.gold[r][a] != "NA" and p[r][a] == "NA" for r in S.IDS)
            vals.append(S.f1(tp, fp, fn))
        out[a] = statistics.mean(vals)
    return out


def mentioned_mf1(p):
    f = []
    for k in S.POLS:
        tp = sum(S.gold[r][a] == k and p[r][a] == k for r in S.IDS for a in S.ASPECTS)
        fp = sum(S.gold[r][a] in S.POLS and S.gold[r][a] != k and p[r][a] == k for r in S.IDS for a in S.ASPECTS)
        fn = sum(S.gold[r][a] == k and p[r][a] != k for r in S.IDS for a in S.ASPECTS)
        f.append(S.f1(tp, fp, fn))
    return sum(f) / 3


V = [("Full HI-ABSA (AQA + gated LGF)", "hiabsa"), ("AQA + LGF without gate", "hiabsa_addfusion"),
     ("AQA only (no LGF)", "hiabsa_nolgf"), ("Random frozen aspect queries (no LGF)", "hiabsa_frozenq_nolgf"),
     ("One shared learned query (no LGF)", "hiabsa_attn1_nolgf"), ("Mean pooling (no LGF)", "hiabsa_mean_nolgf"),
     ("Shared [CLS] + LGF", "hiabsa_noaqa"), ("Shared [CLS] (no LGF)", "hiabsa_none")]
EN = {"Lokasi": "Loc", "Kebersihan": "Clean", "Pelayanan": "Serv", "Kamar & Fasilitas": "Room",
      "Harga": "Price", "Makanan & Minuman": "F&B", "Fasilitas Pendukung": "PubFac"}
L = ["# E6 — Ablation, 5 seeds (test-400)", "",
     "| Variant | runs | ACD-mF1 | Mentioned-mF1 | Joint-F1 | blind-100 Joint-F1 | " + " | ".join(EN.values()) + " (ACD-F1) |",
     "|---|---:|---:|---:|---:|---:|" + "---:|" * 7]
for name, sid in V:
    P = runs("preds_seeds", sid)
    if not P:
        L.append(f"| {name} | 0 | — | — | — | — |" + " — |" * 7); continue
    pa = acd_aspect(P)
    L.append(f"| {name} | {len(P)} | {ms([S.acd(S.gold, p, S.IDS) for p in P])} | {ms([mentioned_mf1(p) for p in P])} | "
             f"{ms([S.joint(S.gold, p, S.IDS) for p in P])} | {ms([S.joint(S.bgold, p, S.ids_b, S.ties) for p in P])} | "
             + " | ".join(f"{pa[a]:.3f}" for a in S.ASPECTS) + " |")
open(os.path.join(REV, "results", "E6_ablation.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
