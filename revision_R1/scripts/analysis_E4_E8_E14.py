# -*- coding: utf-8 -*-
"""Revision R1 analyses on existing predictions (no training).

E4  (R1.4, R2.8)  hotel / city / region overlap between train, val, test; reviews per
                  hotel; hotel-grouped training file for the later grouped-split run.
E8  (R1.9, R2.4)  full metric set and review-level significance tests.
E14a (R2.7)       post-stratification of test results to the corpus rating distribution.

Usage:  python scripts/analysis_E4_E8_E14.py [--preds single|seeds]
        single = prediction files behind the submitted Table 12 (default)
        seeds  = revision_R1/preds_seeds/<system>_<seed>.jsonl, metrics averaged over seeds
Writes: results/E4_overlap.md, results/E8_metrics_<preds>.md, results/E14_natural_<preds>.md,
        data/train_silver_hotel_grouped.jsonl, data/train_silver_random_drop.jsonl
"""
import argparse, ast, json, os, random, statistics, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
OUT = os.path.join(REV, "results"); DATA = os.path.join(REV, "data")
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument("--preds", default="single", choices=["single", "seeds"])
args = ap.parse_args()

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
CATS = ["NA", "POSITIF", "NEGATIF", "NETRAL"]; POLS = CATS[1:]
SEEDS = [42, 123, 7, 2024, 99]


def load(path):
    d = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                o = json.loads(line); d[o["review_id"]] = o
    return d


def cells(o):
    m = {a: "NA" for a in ASPECTS}
    for l in (o.get("labels") or []):
        if l.get("aspect") in m and l.get("polarity") in POLS:
            m[l["aspect"]] = l["polarity"]
    return m


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# region map: parse the REGION literal from make_corpus_figures.py without running it
src = open(os.path.join(ROOT, "make_corpus_figures.py"), encoding="utf-8").read()
node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "REGION")
REGION = ast.literal_eval(node.value)

train = load(os.path.join(ROOT, "train_silver.jsonl"))
val = load(os.path.join(ROOT, "val_gold.jsonl"))
test = load(os.path.join(ROOT, "test_gold.jsonl"))
corpus = load(os.path.join(ROOT, "dataset_merged.jsonl"))
hk = lambda o: (o["hotel"].strip().lower(), o["city"])

# ================================================================ E4
L = ["# E4 — Overlap between splits and hotel coverage", "",
     "Hotel key = (lower-cased hotel name, city).", ""]
S = {"train (silver)": train, "val (gold)": val, "test (gold)": test}
lev = {"hotel": hk, "city": lambda o: o["city"], "region": lambda o: REGION.get(o["city"], "Other")}
L += ["| Level | train | val | test | test units also in train | test reviews whose unit is in train | val units in train | val ∩ test units |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
E4 = {}
for name, fn in lev.items():
    u = {k: {fn(o) for o in s.values()} for k, s in S.items()}
    tr, va, te = u["train (silver)"], u["val (gold)"], u["test (gold)"]
    rev_in = sum(fn(o) in tr for o in test.values())
    E4[name] = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "test_in_train": len(te & tr),
                "test_reviews_unit_in_train": rev_in, "val_in_train": len(va & tr), "val_test": len(va & te)}
    L.append(f"| {name} | {len(tr)} | {len(va)} | {len(te)} | {len(te & tr)} ({100*len(te & tr)/len(te):.0f}%) | "
             f"{rev_in}/400 ({100*rev_in/400:.0f}%) | {len(va & tr)} | {len(va & te)} |")
assert "Other" not in {REGION.get(o["city"], "Other") for o in corpus.values()}, "unmapped city"

# reviews per hotel in the corpus (R2.8)
per_h = Counter(hk(o) for o in corpus.values())
v = sorted(per_h.values())
dist = Counter(min(x, 6) for x in v)
L += ["", "## Reviews per hotel, full corpus (R2.8)", "",
      f"Hotels (name × city): {len(per_h)}; reviews: {len(corpus)}; mean {statistics.mean(v):.2f}, "
      f"median {statistics.median(v)}, min {v[0]}, max {v[-1]}.",
      "The Places API returns at most five reviews per place; fewer than five remain where the place had fewer "
      "reviews or reviews were removed by the length filter or deduplication.", "",
      "| reviews per hotel | hotels |", "|---|---:|"]
L += [f"| {k if k < 6 else '6+'} | {dist[k]} |" for k in sorted(dist)]
per_h_s = Counter(hk(o) for s in S.values() for o in s.values())
vs = sorted(per_h_s.values())
L += ["", f"Labelled sample (train+val+test = {sum(len(s) for s in S.values())} reviews): {len(per_h_s)} hotels, "
      f"mean {statistics.mean(vs):.2f}, median {statistics.median(vs)}, max {vs[-1]} reviews per hotel."]
E4["reviews_per_hotel"] = {"hotels": len(per_h), "mean": statistics.mean(v), "median": statistics.median(v), "max": v[-1]}

# grouped-split training files for the later E4 experiment
test_h = {hk(o) for o in test.values()} | {hk(o) for o in val.values()}
grouped = [o for o in train.values() if hk(o) not in test_h]
n_drop = len(train) - len(grouped)
rng = random.Random(42)
random_drop = rng.sample(list(train.values()), len(grouped))
for fname, rows in [("train_silver_hotel_grouped.jsonl", grouped), ("train_silver_random_drop.jsonl", random_drop)]:
    with open(os.path.join(DATA, fname), "w", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
L += ["", "## Files prepared for the grouped-split experiment (not yet trained)", "",
      f"- `data/train_silver_hotel_grouped.jsonl`: {len(grouped)} reviews — silver training with every hotel that occurs "
      f"in val or test removed ({n_drop} reviews dropped).",
      f"- `data/train_silver_random_drop.jsonl`: {len(random_drop)} reviews — size-matched random subsample of the full "
      "silver training set (seed 42), the control for the size reduction."]
open(os.path.join(OUT, "E4_overlap.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ================================================================ predictions
SINGLE = [("Lexicon baseline", "pred_lex_gold.jsonl", None), ("mBERT", "pred_mbert.jsonl", "mbert"),
          ("XLM-RoBERTa", "pred_xlmr.jsonl", "xlmr"), ("IndoBERTweet", "pred_indobertweet.jsonl", "indobertweet"),
          ("IndoBERT", "pred_indobert_gold.jsonl", "indobert"), ("HI-ABSA", "pred_hiabsa_full.jsonl", "hiabsa"),
          ("HI-ABSA w/o AQA", "pred_hiabsa_noaqa.jsonl", "hiabsa_noaqa"),
          ("HI-ABSA w/o LGF", "pred_hiabsa_nolgf.jsonl", "hiabsa_nolgf"),
          ("HI-ABSA w/o AQA & LGF", "pred_hiabsa_none.jsonl", "hiabsa_none"),
          ("mT5-small", "pred_mt5.jsonl", "mt5"), ("LoRA Qwen2.5-0.5B", "pred_llm_lora_gold.jsonl", "lora_qwen05b"),
          ("Gemini draft (silver)", "pred_gemini_fewshot.jsonl", None)]


def runs_for(name, fname, sysid):
    """List of prediction dicts (one per seed, or one)."""
    if args.preds == "seeds" and sysid:
        ps = [os.path.join(REV, "preds_seeds", f"{sysid}_{s}.jsonl") for s in SEEDS]
        ps = [p for p in ps if os.path.exists(p)]
        if ps:
            return [{r: cells(o) for r, o in load(p).items()} for p in ps]
    return [{r: cells(o) for r, o in load(os.path.join(ROOT, fname)).items()}]


gold = {r: cells(o) for r, o in test.items()}
IDS = sorted(gold)
SYS = [(n, runs_for(n, f, s)) for n, f, s in SINGLE]


def counts(pred, ids, w=None):
    """Per-review contributions so that resampling and weighting are cheap."""
    c = defaultdict(float)
    for r in ids:
        wt = 1.0 if w is None else w[r]
        g, p = gold[r], pred[r]
        for a in ASPECTS:
            gl, pl = g[a], p[a]
            c[("cell", gl, pl)] += wt
            if gl != "NA" and pl != "NA": c[("acd", a, "tp")] += wt
            elif pl != "NA": c[("acd", a, "fp")] += wt
            elif gl != "NA": c[("acd", a, "fn")] += wt
            if gl != "NA" and gl == pl: c["jtp"] += wt
            else:
                if pl != "NA": c["jfp"] += wt
                if gl != "NA": c["jfn"] += wt
    return c


def metrics(c):
    m = {}
    m["ACD_mF1"] = statistics.mean(f1(c[("acd", a, "tp")], c[("acd", a, "fp")], c[("acd", a, "fn")]) for a in ASPECTS)
    jp = c["jtp"] / (c["jtp"] + c["jfp"]) if c["jtp"] + c["jfp"] else 0; jr = c["jtp"] / (c["jtp"] + c["jfn"]) if c["jtp"] + c["jfn"] else 0
    m["Joint_P"], m["Joint_R"], m["Joint_F1"] = jp, jr, f1(c["jtp"], c["jfp"], c["jfn"])
    cell = lambda g, p: c[("cell", g, p)]
    # ACP, predicted-aspect setting (submitted protocol): cells where both mark the aspect
    det = [(g, p) for g in POLS for p in POLS]
    n_det = sum(cell(g, p) for g, p in det)
    m["ACP_pred_n"] = n_det
    m["ACP_pred_Acc"] = sum(cell(g, g) for g in POLS) / n_det if n_det else 0
    pf = []
    for k in POLS:
        tp = cell(k, k); fp = sum(cell(g, k) for g in POLS if g != k); fn = sum(cell(k, p) for p in POLS if p != k)
        pf.append(f1(tp, fp, fn))
    m["ACP_pred_mF1"] = statistics.mean(pf); m["ACP_pred_F1_NEU"] = pf[2]
    # mentioned-aspect setting: every gold-mentioned cell; predicting NA counts as wrong
    mf = []
    for k in POLS:
        tp = cell(k, k); fn = sum(cell(k, p) for p in CATS if p != k); fp = sum(cell(g, k) for g in POLS if g != k)
        mf.append(f1(tp, fp, fn))
    m["Mentioned_mF1"] = statistics.mean(mf); m["Mentioned_F1_NEU"] = mf[2]
    # 4-class macro-F1 over all 7 x N cells
    cf = []
    for k in CATS:
        tp = cell(k, k); fp = sum(cell(g, k) for g in CATS if g != k); fn = sum(cell(k, p) for p in CATS if p != k)
        cf.append(f1(tp, fp, fn))
    m["Cell4_mF1"] = statistics.mean(cf)
    return m


def add(cs):
    t = defaultdict(float)
    for c in cs:
        for k, v in c.items(): t[k] += v
    return t


# per-review count dicts, per system per run
PR = {n: [{r: counts(p, [r]) for r in IDS} for p in runs] for n, runs in SYS}


def sys_metrics(n, ids=IDS, w=None):
    """Mean (and SD over runs) of each metric."""
    per = []
    for run in PR[n]:
        if w is None:
            per.append(metrics(add(run[r] for r in ids)))
        else:
            per.append(metrics(add({k: v * w[r] for k, v in run[r].items()} for r in ids)))
    keys = per[0].keys()
    return {k: (statistics.mean(x[k] for x in per), statistics.pstdev(x[k] for x in per) if len(per) > 1 else 0.0)
            for k in keys}, len(per)


# ================================================================ E8
COLS = ["ACD_mF1", "ACP_pred_Acc", "ACP_pred_mF1", "Mentioned_mF1", "Cell4_mF1", "Joint_P", "Joint_R", "Joint_F1"]
L = [f"# E8 — Full metric set on the 400-review gold test set ({args.preds} predictions)", "",
     "ACP-pred = polarity over cells where gold and prediction both mark the aspect (submitted protocol; the item set differs by system).",
     "Mentioned-mF1 = polarity macro-F1 over every gold-mentioned cell, predicting NA counts as an error (same items for every system).",
     "Cell4-mF1 = 4-class macro-F1 over all 7 × 400 cells. The gold-aspect setting (forced polarity on gold aspects) needs new inference and is not here.",
     "" if args.preds == "single" else "Values are mean ± SD over seeds (n runs in last column).", "",
     "| System | " + " | ".join(COLS) + " | ACP-pred items | runs |", "|---|" + "---:|" * (len(COLS) + 2)]
E8 = {}
for n, _ in SYS:
    m, k = sys_metrics(n); E8[n] = m
    fm = (lambda x: f"{x[0]:.3f}") if k == 1 else (lambda x: f"{x[0]:.3f}±{x[1]:.3f}")
    L.append(f"| {n} | " + " | ".join(fm(m[c]) for c in COLS) + f" | {m['ACP_pred_n'][0]:.0f} | {k} |")

# neutral F1 with review-level bootstrap CI
B = 2000; rng = random.Random(0)
boots = [[rng.choice(IDS) for _ in IDS] for _ in range(B)]


def boot_metric(n, key, run=0):
    xs = sorted(metrics(add(PR[n][run][r] for r in smp))[key] for smp in boots)
    return xs[int(.025 * B)], xs[int(.975 * B)]


L += ["", "## Neutral F1 with 95% review-level bootstrap CI (first run)", "",
      f"Gold test has {sum(gold[r][a] == 'NETRAL' for r in IDS for a in ASPECTS)} neutral cells.", "",
      "| System | F1-NEU (ACP-pred) [95% CI] | F1-NEU (mentioned) [95% CI] |", "|---|---|---|"]
for n in ["IndoBERT", "HI-ABSA", "LoRA Qwen2.5-0.5B", "Gemini draft (silver)"]:
    a = boot_metric(n, "ACP_pred_F1_NEU"); b = boot_metric(n, "Mentioned_F1_NEU")
    L.append(f"| {n} | {E8[n]['ACP_pred_F1_NEU'][0]:.3f} [{a[0]:.3f}, {a[1]:.3f}] | "
             f"{E8[n]['Mentioned_F1_NEU'][0]:.3f} [{b[0]:.3f}, {b[1]:.3f}] |")


# review-level significance: paired bootstrap CI + paired permutation (swap whole reviews), Holm
def joint(cs): return f1(cs["jtp"], cs["jfp"], cs["jfn"])


def _jt(c): return (c["jtp"], c["jfp"], c["jfn"])


def _jf1(rows_):
    tp = sum(x[0] for x in rows_); fp = sum(x[1] for x in rows_); fn = sum(x[2] for x in rows_)
    return f1(tp, fp, fn)


def paired(a, b, run=0, P=5000, seed=1):
    ca = {r: _jt(PR[a][run][r]) for r in IDS}; cb = {r: _jt(PR[b][run][r]) for r in IDS}
    obs = _jf1([ca[r] for r in IDS]) - _jf1([cb[r] for r in IDS])
    d = sorted(_jf1([ca[r] for r in s]) - _jf1([cb[r] for r in s]) for s in boots)
    rg = random.Random(seed); ext = 0
    for _ in range(P):
        A, Bc = [], []
        for r in IDS:
            if rg.random() < .5: A.append(ca[r]); Bc.append(cb[r])
            else: A.append(cb[r]); Bc.append(ca[r])
        if abs(_jf1(A) - _jf1(Bc)) >= abs(obs) - 1e-12: ext += 1
    return obs, d[int(.025 * B)], d[int(.975 * B)], (ext + 1) / (P + 1)


L += ["", "## Review-level paired tests of HI-ABSA against each system (Joint-F1, first run)", "",
      "Bootstrap and permutation both resample whole reviews, so the 7 cells of a review stay together "
      "(replaces the per-cell McNemar test of submitted Table 11). Holm correction over the comparisons.", "",
      "| HI-ABSA vs | Δ Joint-F1 | 95% CI | permutation p | Holm p |", "|---|---:|---|---:|---:|"]
others = [n for n, _ in SYS if n != "HI-ABSA"]
res = [(o, *paired("HI-ABSA", o)) for o in others]
order = sorted(range(len(res)), key=lambda i: res[i][4]); holm = [0] * len(res); run_max = 0
for rank, i in enumerate(order):
    run_max = max(run_max, min(1, (len(res) - rank) * res[i][4])); holm[i] = run_max
for (o, d, lo, hi, p), h in zip(res, holm):
    L.append(f"| {o} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.4f} | {h:.4f} |")
open(os.path.join(OUT, f"E8_metrics_{args.preds}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ================================================================ E14a
star = lambda o: int(o["rating"])
pc = Counter(star(o) for o in corpus.values()); pt = Counter(star(o) for o in test.values())
w = {r: (pc[star(o)] / len(corpus)) / (pt[star(o)] / len(test)) for r, o in test.items()}
L = [f"# E14a — Test results re-weighted to the corpus rating distribution ({args.preds} predictions)", "",
     "Each test review is weighted by (corpus share of its star rating) / (test share). This estimates performance on "
     "reviews drawn at the natural rating distribution; it cannot correct for anything other than rating.", "",
     "| Stars | corpus share | test share | weight |", "|---|---:|---:|---:|"]
for s in sorted(pc):
    L.append(f"| {s} | {pc[s]/len(corpus):.3f} | {pt[s]/len(test):.3f} | {(pc[s]/len(corpus))/(pt[s]/len(test)):.2f} |")
ess = sum(w.values()) ** 2 / sum(x * x for x in w.values())
L += ["", f"Effective sample size after weighting: {ess:.0f} of 400 reviews.", "",
      "| System | Joint-F1 balanced | Joint-F1 natural | ACD-mF1 balanced | ACD-mF1 natural | Mentioned-mF1 balanced | Mentioned-mF1 natural |",
      "|---|---:|---:|---:|---:|---:|---:|"]
E14 = {}
for n, _ in SYS:
    mb, _k = sys_metrics(n); mn, _k = sys_metrics(n, w=w); E14[n] = mn
    L.append(f"| {n} | {mb['Joint_F1'][0]:.3f} | {mn['Joint_F1'][0]:.3f} | {mb['ACD_mF1'][0]:.3f} | {mn['ACD_mF1'][0]:.3f} | "
             f"{mb['Mentioned_mF1'][0]:.3f} | {mn['Mentioned_mF1'][0]:.3f} |")


# dashboard net sentiment s_a (Eq. net_sent): gold labels and HI-ABSA, balanced vs natural
def net(labels_by_r, weights=None):
    out = {}
    for a in ASPECTS:
        c = Counter()
        for r in IDS:
            c[labels_by_r[r][a]] += 1.0 if weights is None else weights[r]
        d = c["POSITIF"] + c["NEGATIF"] + c["NETRAL"]
        out[a] = (c["POSITIF"] - c["NEGATIF"]) / d if d else float("nan")
    return out


hi = SYS[[n for n, _ in SYS].index("HI-ABSA")][1][0]
rows = [("gold, balanced", net(gold)), ("gold, natural", net(gold, w)),
        ("HI-ABSA, balanced", net(hi)), ("HI-ABSA, natural", net(hi, w))]
L += ["", "## Dashboard net sentiment s_a (Figure 8a) under both distributions", "",
      "| | " + " | ".join(ASPECTS) + " |", "|---|" + "---:|" * 7]
for name, s in rows:
    L.append(f"| {name} | " + " | ".join(f"{s[a]:+.2f}" for a in ASPECTS) + " |")
open(os.path.join(OUT, f"E14_natural_{args.preds}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

json.dump({"E4": E4, "E8": E8, "E14": E14}, open(os.path.join(OUT, f"analysis_E4_E8_E14_{args.preds}.json"), "w",
          encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("done", args.preds)
