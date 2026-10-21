# -*- coding: utf-8 -*-
"""Revision R1 analyses that need no training.

E1  (R1.2, R2.1, R3.S2)  every system scored on the 100 blind-annotated test reviews,
                         against the adjudicated gold and against the blind majority.
E2  (R1.3)               agreement beyond overall Fleiss' kappa.
E3  (R1.1, R2.4)         aspect x polarity distributions.
E16 (R1.6)               coverage of the Lexicon-Gated Fusion signal.

Usage:   python scripts/analysis_E1_E2_E3_E16.py      (from revision_R1/)
Writes:  results/E1_blind100.md, E2_agreement.md, E3_distribution.md, E16_lexicon.md
         and results/analysis_E1_E2_E3_E16.json
"""
import json, os, random, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
OUT = os.path.join(REV, "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
from lexicon_baseline import ASPECT_KEYWORDS, POS, NEG  # the lexicon HI-ABSA imports

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
EN = {"Lokasi": "Location", "Kebersihan": "Cleanliness", "Pelayanan": "Service/Staff",
      "Kamar & Fasilitas": "Room & Facilities", "Harga": "Price/Value",
      "Makanan & Minuman": "Food & Beverage", "Fasilitas Pendukung": "Public Facilities"}
CATS = ["NA", "POSITIF", "NEGATIF", "NETRAL"]
POLS = ["POSITIF", "NEGATIF", "NETRAL"]
J = {}


def load(path):
    d = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                d[o["review_id"]] = o
    return d


def cells(o):
    m = {a: "NA" for a in ASPECTS}
    for l in (o.get("labels") or []):
        if l.get("aspect") in m and l.get("polarity") in POLS:
            m[l["aspect"]] = l["polarity"]
    return m


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def scores(gold, pred, ids, ignore=frozenset()):
    """gold/pred: rid -> {aspect: label}. ignore: set of (rid, aspect) cells left out."""
    acd = {a: [0, 0, 0] for a in ASPECTS}
    jt = jf = jn = 0
    conf = Counter()
    for r in ids:
        g, p = gold[r], pred.get(r, {a: "NA" for a in ASPECTS})
        for a in ASPECTS:
            if (r, a) in ignore:
                continue
            gl, pl = g[a], p[a]
            conf[(gl, pl)] += 1
            if gl != "NA" and pl != "NA": acd[a][0] += 1
            elif pl != "NA": acd[a][1] += 1
            elif gl != "NA": acd[a][2] += 1
            if gl != "NA" and gl == pl: jt += 1
            else:
                if pl != "NA": jf += 1
                if gl != "NA": jn += 1
    acd_f1 = sum(f1(*acd[a]) for a in ASPECTS) / len(ASPECTS)
    cls_f1 = []
    for c in CATS:
        tp = conf[(c, c)]
        fp = sum(v for (g, p), v in conf.items() if p == c and g != c)
        fn = sum(v for (g, p), v in conf.items() if g == c and p != c)
        cls_f1.append(f1(tp, fp, fn))
    return {"ACD_macroF1": acd_f1, "Joint_F1": f1(jt, jf, jn),
            "Cell4_macroF1": sum(cls_f1) / 4, "F1_NEU_cell": cls_f1[3]}


# ---------------------------------------------------------------- data
gold_all = {r: cells(o) for r, o in load(os.path.join(ROOT, "gold_final.jsonl")).items()}
test = load(os.path.join(ROOT, "test_gold.jsonl"))
val = load(os.path.join(ROOT, "val_gold.jsonl"))
train = load(os.path.join(ROOT, "train_silver.jsonl"))
ids_blind = [int(x) for x in open(os.path.join(BLIND, "blind_subset_ids.txt"))
             if x.strip() and not x.startswith("#")]
assert len(ids_blind) == 100 and set(ids_blind) <= set(test)

bl = [{r: cells(o) for r, o in load(os.path.join(BLIND, BLIND_FILE.format(i=i))).items()}
      for i in (1, 2, 3)]
blind_gold, blind_ties = {}, set()
for r in ids_blind:
    blind_gold[r] = {}
    for a in ASPECTS:
        v = Counter(b[r][a] for b in bl)
        lab, n = v.most_common(1)[0]
        if n == 1:
            blind_ties.add((r, a)); lab = "NA"   # placeholder; tie cells are ignored
        blind_gold[r][a] = lab

# ---------------------------------------------------------------- E1
SYSTEMS = [  # label, prediction file (single-seed runs reported in submitted Table 12)
    ("Lexicon baseline", "pred_lex_gold.jsonl"),
    ("mBERT", "pred_mbert.jsonl"),
    ("XLM-RoBERTa", "pred_xlmr.jsonl"),
    ("IndoBERTweet", "pred_indobertweet.jsonl"),
    ("IndoBERT", "pred_indobert_gold.jsonl"),
    ("HI-ABSA", "pred_hiabsa_full.jsonl"),
    ("mT5-small", "pred_mt5.jsonl"),
    ("LoRA Qwen2.5-0.5B", "pred_llm_lora_gold.jsonl"),
    ("Gemini draft (silver)", "pred_gemini_fewshot.jsonl"),
]
preds = {n: {r: cells(o) for r, o in load(os.path.join(ROOT, f)).items()} for n, f in SYSTEMS}

# sanity: the Gemini row is the silver draft itself
pre = load(os.path.join(ROOT, "annotation_prefilled.jsonl") if os.path.exists(os.path.join(ROOT, "annotation_prefilled.jsonl"))
           else os.path.join(DEP, "annotation_prefilled_gold500.jsonl"))
same = sum(cells(pre[r]) == preds["Gemini draft (silver)"][r] for r in test)

rows = []
for n, _ in SYSTEMS:
    s400 = scores(gold_all, preds[n], sorted(test))
    s_g = scores(gold_all, preds[n], ids_blind, blind_ties)
    s_b = scores(blind_gold, preds[n], ids_blind, blind_ties)
    rows.append((n, s400, s_g, s_b))


def boot_gap(a, b, gold, ids, ignore, B=2000, seed=0):
    """Review-level paired bootstrap of Joint-F1(a) - Joint-F1(b)."""
    rng = random.Random(seed)
    obs = scores(gold, preds[a], ids, ignore)["Joint_F1"] - scores(gold, preds[b], ids, ignore)["Joint_F1"]
    d = []
    for _ in range(B):
        smp = [rng.choice(ids) for _ in ids]
        # duplicates matter: expand ids with multiplicity
        d.append(_joint_multi(gold, preds[a], smp, ignore) - _joint_multi(gold, preds[b], smp, ignore))
    d.sort()
    return obs, d[int(0.025 * B)], d[int(0.975 * B)]


def _joint_multi(gold, pred, smp, ignore):
    jt = jf = jn = 0
    for r in smp:
        for a in ASPECTS:
            if (r, a) in ignore: continue
            gl, pl = gold[r][a], pred[r][a]
            if gl != "NA" and gl == pl: jt += 1
            else:
                if pl != "NA": jf += 1
                if gl != "NA": jn += 1
    return f1(jt, jf, jn)


gaps = {}
for ref in ("IndoBERT", "HI-ABSA"):
    gaps[ref] = {"vs_gold": boot_gap("Gemini draft (silver)", ref, gold_all, ids_blind, blind_ties),
                 "vs_blind": boot_gap("Gemini draft (silver)", ref, blind_gold, ids_blind, blind_ties)}

# Agreement of the adjudicated gold itself with the blind majority (ceiling reference)
g_vs_b = scores(blind_gold, gold_all, ids_blind, blind_ties)

L = ["# E1 — Systems scored on the 100 blind-annotated test reviews", "",
     f"Blind subset: {len(ids_blind)} reviews, all in the 400-review test set; "
     f"{len(blind_ties)} three-way-tie cells in the blind round are ignored (697 cells scored).",
     f"Sanity check: the 'Gemini' predictions equal the silver drafts on {same}/400 test reviews.",
     "Predictions are the single-seed runs behind the submitted Table 12 (E5 will replace them).", "",
     "| System | Joint-F1 test-400 (gold) | Joint-F1 blind-100 vs gold | Joint-F1 blind-100 vs blind | Δ (blind − gold) | ACD-mF1 vs blind | 4-class cell mF1 vs blind |",
     "|---|---:|---:|---:|---:|---:|---:|"]
for n, s400, sg, sb in rows:
    L.append(f"| {n} | {s400['Joint_F1']:.3f} | {sg['Joint_F1']:.3f} | {sb['Joint_F1']:.3f} | "
             f"{sb['Joint_F1'] - sg['Joint_F1']:+.3f} | {sb['ACD_macroF1']:.3f} | {sb['Cell4_macroF1']:.3f} |")
L += ["", f"Reference: adjudicated gold scored against the blind majority on the same cells: "
      f"Joint-F1 {g_vs_b['Joint_F1']:.3f}, ACD-mF1 {g_vs_b['ACD_macroF1']:.3f}, cell mF1 {g_vs_b['Cell4_macroF1']:.3f}.", "",
      "Review-level paired bootstrap (2,000 resamples of the 100 reviews), Joint-F1 gap Gemini draft − encoder:", "",
      "| Encoder | vs adjudicated gold: gap [95% CI] | vs blind majority: gap [95% CI] |", "|---|---|---|"]
for ref, g in gaps.items():
    L.append(f"| {ref} | {g['vs_gold'][0]:+.3f} [{g['vs_gold'][1]:+.3f}, {g['vs_gold'][2]:+.3f}] | "
             f"{g['vs_blind'][0]:+.3f} [{g['vs_blind'][1]:+.3f}, {g['vs_blind'][2]:+.3f}] |")
open(os.path.join(OUT, "E1_blind100.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
J["E1"] = {"rows": [{"system": n, "test400_gold": a, "blind100_gold": b, "blind100_blind": c}
                    for n, a, b, c in rows],
           "gold_vs_blind": g_vs_b, "gaps": gaps, "ties": len(blind_ties), "gemini_equals_draft": same}

# ---------------------------------------------------------------- E2
adj = [json.loads(l) for l in open(os.path.join(DEP, "adjudication_record.jsonl"), encoding="utf-8")]
assert len(adj) == 3500
votes = [(o["review_id"], o["aspect"], [o["annotator_1"], o["annotator_2"], o["annotator_3"]]) for o in adj]


def fleiss(items, cats):
    """items: list of label lists (n raters each). Returns overall and per-category kappa."""
    N, n = len(items), len(items[0])
    cnt = [Counter(it) for it in items]
    p = {c: sum(m[c] for m in cnt) / (N * n) for c in cats}
    Pi = [(sum(m[c] ** 2 for c in cats) - n) / (n * (n - 1)) for m in cnt]
    Pbar, Pe = sum(Pi) / N, sum(v * v for v in p.values())
    k = (Pbar - Pe) / (1 - Pe) if Pe < 1 else float("nan")
    per = {}
    for c in cats:
        if 0 < p[c] < 1:
            per[c] = 1 - sum(m[c] * (n - m[c]) for m in cnt) / (N * n * (n - 1) * p[c] * (1 - p[c]))
        else:
            per[c] = float("nan")
    return k, per


def kripp_nominal(items, cats):
    """Krippendorff's alpha, nominal, complete data."""
    o = Counter()
    for it in items:
        m = len(it)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(it[i], it[j])] += 1 / (m - 1)
    n_c = Counter()
    for (c, k), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    Do = sum(v for (c, k), v in o.items() if c != k)
    De = sum(n_c[c] * n_c[k] for c in cats for k in cats if c != k) / (n - 1)
    return 1 - Do / De if De else float("nan")


def agreement_block(vs, label):
    allc = [v for _, _, v in vs]
    out = {}
    k, per = fleiss(allc, CATS)
    out["fleiss_4class_all"] = k
    out["fleiss_per_category"] = per
    out["alpha_4class_all"] = kripp_nominal(allc, CATS)
    det = [["M" if x != "NA" else "NA" for x in v] for v in allc]
    out["fleiss_detection_binary"] = fleiss(det, ["M", "NA"])[0]
    nz = [v for v in allc if any(x != "NA" for x in v)]
    out["n_cells_any_mention"] = len(nz)
    out["fleiss_4class_excl_unanimous_NA"] = fleiss(nz, CATS)[0]
    out["alpha_4class_excl_unanimous_NA"] = kripp_nominal(nz, CATS)
    allm = [v for v in allc if all(x != "NA" for x in v)]
    out["n_cells_all_three_mention"] = len(allm)
    kp, perp = fleiss(allm, POLS)
    out["fleiss_polarity_given_mention"] = kp
    out["fleiss_polarity_per_class"] = perp
    out["alpha_polarity_given_mention"] = kripp_nominal(allm, POLS)
    conf = Counter()
    for v in allc:
        for i in range(3):
            for j in range(3):
                if i != j:
                    conf[(v[i], v[j])] += 1
    out["pairwise_confusion"] = {f"{a}|{b}": c for (a, b), c in conf.items()}
    unan = sum(len(set(v)) == 1 for v in allc)
    tie = sum(len(set(v)) == 3 for v in allc)
    out["unanimous"] = unan; out["majority_2of3"] = len(allc) - unan - tie; out["three_way_ties"] = tie
    out["n_cells"] = len(allc)
    return out


E2 = {"verification_500": agreement_block(votes, "verification")}
# per aspect, excluding unanimous NA
E2["per_aspect"] = {}
for a in ASPECTS:
    va = [v for r, asp, v in votes if asp == a]
    nz = [v for v in va if any(x != "NA" for x in v)]
    E2["per_aspect"][a] = {"fleiss_all": fleiss(va, CATS)[0], "fleiss_excl_NA": fleiss(nz, CATS)[0],
                           "n_any_mention": len(nz)}
# blind vs verification on the same 100 reviews
ver = {(r, a): v for r, a, v in votes}
bvotes = [(r, a, [b[r][a] for b in bl]) for r in ids_blind for a in ASPECTS]
vvotes = [(r, a, ver[(r, a)]) for r in ids_blind for a in ASPECTS]
E2["blind_100"] = agreement_block(bvotes, "blind")
E2["verification_same_100"] = agreement_block(vvotes, "verification")

# adjudication: polarity conflicts on 2-1 splits
split = Counter()
for _, _, v in votes:
    if len(set(v)) == 2:
        maj = Counter(v).most_common(1)[0][0]
        minority = [x for x in v if x != maj][0]
        split[(maj, minority)] += 1
E2["majority_minority_pairs"] = {f"{a}<-{b}": c for (a, b), c in split.most_common()}

def fmt(x): return "n/a" if x != x else f"{x:.3f}"

L = ["# E2 — Agreement beyond overall Fleiss' kappa", "",
     "Unit: one (review, aspect) cell, four values {NA, POS, NEG, NEU}; 3 annotators.", "",
     "| Statistic | Verification, 500 reviews | Verification, blind-100 reviews | Blind round, same 100 |",
     "|---|---:|---:|---:|"]
V, VS, BB = E2["verification_500"], E2["verification_same_100"], E2["blind_100"]
for key, name in [("n_cells", "cells"), ("fleiss_4class_all", "Fleiss κ, 4-class, all cells"),
                  ("alpha_4class_all", "Krippendorff α, 4-class, all cells"),
                  ("fleiss_detection_binary", "Fleiss κ, detection (mentioned vs NA)"),
                  ("n_cells_any_mention", "cells with ≥1 annotator marking the aspect"),
                  ("fleiss_4class_excl_unanimous_NA", "Fleiss κ, 4-class, excl. unanimous-NA cells"),
                  ("alpha_4class_excl_unanimous_NA", "Krippendorff α, 4-class, excl. unanimous-NA cells"),
                  ("n_cells_all_three_mention", "cells all three mark as mentioned"),
                  ("fleiss_polarity_given_mention", "Fleiss κ, polarity only (3-class), all-three-mention cells"),
                  ("alpha_polarity_given_mention", "Krippendorff α, polarity only"),
                  ("unanimous", "unanimous cells"), ("majority_2of3", "2-of-3 majority cells"),
                  ("three_way_ties", "three-way ties")]:
    f = (lambda x: str(x)) if isinstance(V[key], int) else fmt
    L.append(f"| {name} | {f(V[key])} | {f(VS[key])} | {f(BB[key])} |")
L += ["", "Category-specific Fleiss κ (Fleiss 1971), 4-class, all cells:", "",
      "| Category | Verification 500 | Verification same 100 | Blind 100 |", "|---|---:|---:|---:|"]
for c in CATS:
    L.append(f"| {c} | {fmt(V['fleiss_per_category'][c])} | {fmt(VS['fleiss_per_category'][c])} | {fmt(BB['fleiss_per_category'][c])} |")
L += ["", "Per-class κ within polarity-only cells:", "", "| Polarity | Verification 500 | Blind 100 |", "|---|---:|---:|"]
for c in POLS:
    L.append(f"| {c} | {fmt(V['fleiss_polarity_per_class'][c])} | {fmt(BB['fleiss_polarity_per_class'][c])} |")
L += ["", "Per aspect (verification, 500 reviews):", "", "| Aspect | κ all cells | κ excl. unanimous NA | cells ≥1 mention |", "|---|---:|---:|---:|"]
for a in ASPECTS:
    p = E2["per_aspect"][a]
    L.append(f"| {EN[a]} | {fmt(p['fleiss_all'])} | {fmt(p['fleiss_excl_NA'])} | {p['n_any_mention']} |")
L += ["", "Pairwise inter-annotator confusion (verification 500; counts over ordered annotator pairs, rows = annotator i, cols = annotator j):", "",
      "| | " + " | ".join(CATS) + " |", "|---|" + "---:|" * 4]
for a in CATS:
    L.append(f"| {a} | " + " | ".join(str(V["pairwise_confusion"].get(f"{a}|{b}", 0)) for b in CATS) + " |")
L += ["", "Pairwise inter-annotator confusion (blind 100):", "", "| | " + " | ".join(CATS) + " |", "|---|" + "---:|" * 4]
for a in CATS:
    L.append(f"| {a} | " + " | ".join(str(BB["pairwise_confusion"].get(f"{a}|{b}", 0)) for b in CATS) + " |")
L += ["", "2-of-3 splits in the verification round, majority ← minority label:", ""]
L += [f"- {k}: {v}" for k, v in E2["majority_minority_pairs"].items()]
open(os.path.join(OUT, "E2_agreement.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
J["E2"] = E2

# ---------------------------------------------------------------- E3
def dist(rows_):
    t = {a: Counter() for a in ASPECTS}
    for o in rows_.values():
        for a, p in cells(o).items():
            t[a][p] += 1
    return t

sets = {"silver train (2,480)": train, "gold val (100)": val, "gold test (400)": test,
        "gold all (500)": load(os.path.join(ROOT, "gold_final.jsonl"))}
E3 = {}
L = ["# E3 — Aspect × polarity distributions", ""]
for name, rws in sets.items():
    t = dist(rws); E3[name] = {a: dict(t[a]) for a in ASPECTS}
    L += [f"## {name}", "", "| Aspect | POS | NEG | NEU | mentioned | NEU % of mentioned |", "|---|---:|---:|---:|---:|---:|"]
    tot = Counter()
    for a in ASPECTS:
        c = t[a]; m = c["POSITIF"] + c["NEGATIF"] + c["NETRAL"]; tot.update(c)
        L.append(f"| {EN[a]} | {c['POSITIF']} | {c['NEGATIF']} | {c['NETRAL']} | {m} | {100*c['NETRAL']/m if m else 0:.1f} |")
    m = tot["POSITIF"] + tot["NEGATIF"] + tot["NETRAL"]
    L += [f"| **Total** | {tot['POSITIF']} | {tot['NEGATIF']} | {tot['NETRAL']} | {m} | {100*tot['NETRAL']/m:.1f} |", ""]
bt = {a: Counter(blind_gold[r][a] for r in ids_blind if (r, a) not in blind_ties) for a in ASPECTS}
gt = {a: Counter(gold_all[r][a] for r in ids_blind if (r, a) not in blind_ties) for a in ASPECTS}
L += ["## Blind-100: blind majority vs adjudicated gold (same 697 cells)", "",
      "| Aspect | blind POS/NEG/NEU | gold POS/NEG/NEU |", "|---|---|---|"]
for a in ASPECTS:
    L.append(f"| {EN[a]} | {bt[a]['POSITIF']}/{bt[a]['NEGATIF']}/{bt[a]['NETRAL']} | {gt[a]['POSITIF']}/{gt[a]['NEGATIF']}/{gt[a]['NETRAL']} |")
# rating bucket x aspect polarity (gold 500): does star rating predict aspect polarity?
g500 = sets["gold all (500)"]
bucket = lambda s: "1-2★" if s <= 2 else ("3★" if s == 3 else "4-5★")
rb = defaultdict(Counter); mixed = Counter(); nrev = Counter()
for o in g500.values():
    b = bucket(int(o["rating"])); nrev[b] += 1
    ps = [p for p in cells(o).values() if p != "NA"]
    rb[b].update(ps)
    if "POSITIF" in ps and "NEGATIF" in ps: mixed[b] += 1
L += ["", "## Star-rating bucket × aspect polarity (gold 500)", "",
      "| Rating bucket | reviews | POS | NEG | NEU | reviews with both POS and NEG aspects |", "|---|---:|---:|---:|---:|---:|"]
for b in ["1-2★", "3★", "4-5★"]:
    L.append(f"| {b} | {nrev[b]} | {rb[b]['POSITIF']} | {rb[b]['NEGATIF']} | {rb[b]['NETRAL']} | {mixed[b]} |")
E3["rating_bucket"] = {b: dict(rb[b]) for b in rb}; E3["mixed_reviews"] = dict(mixed)
open(os.path.join(OUT, "E3_distribution.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
J["E3"] = E3

# ---------------------------------------------------------------- E16
def lex_feature(text):  # identical logic to finetune_hiabsa.lexicon_feature
    t = text.lower(); feats = []
    for asp in ASPECTS:
        kws = ASPECT_KEYWORDS.get(asp, [])
        if any(k in t for k in kws):
            feats.append([float(sum(t.count(w) for w in POS)), float(sum(t.count(w) for w in NEG))])
        else:
            feats.append([0.0, 0.0])
    return feats

def whole_word(k, t):
    return re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", t) is not None

E16 = {}
L = ["# E16 — Coverage of the Lexicon-Gated Fusion signal", "",
     f"Lexicon: {len(POS)} positive, {len(NEG)} negative terms; cue words per aspect: "
     + ", ".join(f"{EN[a]} {len(ASPECT_KEYWORDS[a])}" for a in ASPECTS) + ".",
     "Matching is substring on the lower-cased review; ℓ_a = whole-review POS/NEG counts when any cue of aspect a occurs.", ""]
for name, rws in [("silver train", train), ("gold test", test)]:
    nrev = len(rws); any_sig = 0; fire = Counter(); fire_m = Counter(); fire_na = Counter()
    ment = Counter(); na = Counter(); miss = Counter(); sub_only = Counter(); cue_hit = Counter()
    for o in rws.values():
        t = o["text"].lower(); f = lex_feature(o["text"]); c = cells(o)
        if any(sum(x) > 0 for x in f): any_sig += 1
        for i, a in enumerate(ASPECTS):
            hit = any(k in t for k in ASPECT_KEYWORDS[a])
            nonzero = sum(f[i]) > 0
            if hit:
                cue_hit[a] += 1
                if not any(whole_word(k, t) for k in ASPECT_KEYWORDS[a]): sub_only[a] += 1
            if nonzero: fire[a] += 1
            if c[a] != "NA":
                ment[a] += 1; fire_m[a] += nonzero; miss[a] += (not hit)
            else:
                na[a] += 1; fire_na[a] += nonzero
    E16[name] = {"reviews": nrev, "reviews_any_nonzero": any_sig,
                 "per_aspect": {a: {"cue_hit": cue_hit[a], "cue_hit_substring_only": sub_only[a],
                                    "nonzero": fire[a], "mentioned": ment[a], "nonzero_when_mentioned": fire_m[a],
                                    "cue_missing_when_mentioned": miss[a], "NA": na[a], "nonzero_when_NA": fire_na[a]}
                                for a in ASPECTS}}
    L += [f"## {name} ({nrev} reviews; label source = {'silver' if name.startswith('silver') else 'gold'})", "",
          f"Reviews with any non-zero ℓ: {any_sig} ({100*any_sig/nrev:.1f}%).", "",
          "| Aspect | cue hit (reviews) | of which substring-only | ℓ≠0 when aspect mentioned | cue absent when mentioned | ℓ≠0 when aspect NA |",
          "|---|---:|---:|---:|---:|---:|"]
    for a in ASPECTS:
        p = E16[name]["per_aspect"][a]
        L.append(f"| {EN[a]} | {p['cue_hit']} | {p['cue_hit_substring_only']} | "
                 f"{p['nonzero_when_mentioned']}/{p['mentioned']} ({100*p['nonzero_when_mentioned']/max(p['mentioned'],1):.0f}%) | "
                 f"{p['cue_missing_when_mentioned']}/{p['mentioned']} | "
                 f"{p['nonzero_when_NA']}/{p['NA']} ({100*p['nonzero_when_NA']/max(p['NA'],1):.0f}%) |")
    L.append("")
# substring false triggers: which cue words fire only inside other words, gold test
ex = Counter()
for o in test.values():
    t = o["text"].lower()
    for a in ASPECTS:
        for k in ASPECT_KEYWORDS[a]:
            if k in t and not whole_word(k, t):
                ex[(EN[a], k)] += 1
L += ["Cue words matching only inside other words (gold test, reviews). This mixes legitimate affixed forms "
          "(kamarnya, makanan, kebersihan, staff) with false hits (merasa -> rasa, 'ac' inside unrelated words); "
          "it is an upper bound on false triggers, not a count of them.", ""]
L += [f"- {a} / '{k}': {c}" for (a, k), c in ex.most_common(15)]
E16["substring_only_cues_test"] = {f"{a}/{k}": c for (a, k), c in ex.most_common()}
open(os.path.join(OUT, "E16_lexicon.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
J["E16"] = E16

json.dump(J, open(os.path.join(OUT, "analysis_E1_E2_E3_E16.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
print("written to", OUT)
