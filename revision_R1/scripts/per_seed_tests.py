# -*- coding: utf-8 -*-
"""R1.9: review-level paired tests of HI-ABSA vs each system, repeated for every seed
(seed-matched for trained systems), with Holm correction within each seed.
Also F1-NEU per seed. Writes results/E8_per_seed_tests.md
"""
import json, os, random, statistics, sys

sys.stdout.reconfigure(encoding="utf-8")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas", "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
POLS = ["POSITIF", "NEGATIF", "NETRAL"]
SEEDS = [42, 123, 7, 2024, 99]
P, B = 5000, 2000


def load(p):
    return {o["review_id"]: o for o in (json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip())}


def cells(labels):
    m = {a: "NA" for a in ASPECTS}
    for l in labels or []:
        if l.get("aspect") in m and l.get("polarity") in POLS:
            m[l["aspect"]] = l["polarity"]
    return m


gold = {r: cells(o["labels"]) for r, o in load(os.path.join(ROOT, "test_gold.jsonl")).items()}
IDS = sorted(gold)


def per_review(pred):
    out = {}
    for r in IDS:
        t = fp = fn = 0
        for a in ASPECTS:
            g, p = gold[r][a], pred[r][a]
            if g != "NA" and g == p: t += 1
            else: fp += p != "NA"; fn += g != "NA"
        out[r] = (t, fp, fn)
    return out


def jf1(rows):
    t = sum(x[0] for x in rows); fp = sum(x[1] for x in rows); fn = sum(x[2] for x in rows)
    p = t / (t + fp) if t + fp else 0; r = t / (t + fn) if t + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def neu_f1(pred):
    tp = sum(gold[r][a] == "NETRAL" and pred[r][a] == "NETRAL" for r in IDS for a in ASPECTS)
    fp = sum(gold[r][a] != "NETRAL" and pred[r][a] == "NETRAL" for r in IDS for a in ASPECTS)
    fn = sum(gold[r][a] == "NETRAL" and pred[r][a] != "NETRAL" for r in IDS for a in ASPECTS)
    p = tp / (tp + fp) if tp + fp else 0; rr = tp / (tp + fn) if tp + fn else 0
    return 2 * p * rr / (p + rr) if p + rr else 0


def test(ca, cb, seed):
    obs = jf1(ca.values()) - jf1(cb.values())
    rg = random.Random(seed)
    d = sorted(jf1([ca[r] for r in s]) - jf1([cb[r] for r in s])
               for s in ([rg.choice(IDS) for _ in IDS] for _ in range(B)))
    ext = 0
    for _ in range(P):
        A, Bc = [], []
        for r in IDS:
            if rg.random() < .5: A.append(ca[r]); Bc.append(cb[r])
            else: A.append(cb[r]); Bc.append(ca[r])
        ext += abs(jf1(A) - jf1(Bc)) >= abs(obs) - 1e-12
    return obs, d[int(.025 * B)], d[int(.975 * B)], (ext + 1) / (P + 1)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = 0
    for k, i in enumerate(order):
        m = max(m, min(1, (len(ps) - k) * ps[i])); out[i] = m
    return out


TRAINED = [("mBERT", "mbert"), ("XLM-RoBERTa", "xlmr"), ("IndoBERTweet", "indobertweet"), ("IndoBERT", "indobert"),
           ("HI-ABSA w/o AQA", "hiabsa_noaqa"), ("HI-ABSA w/o LGF", "hiabsa_nolgf"),
           ("HI-ABSA w/o AQA & LGF", "hiabsa_none"), ("mT5-small", "mt5"), ("LoRA Qwen2.5-0.5B", "lora_qwen05b")]
qraw = load(os.path.join(REV, "preds_llm", "qwen25_7b_zeroshot.jsonl"))
FIXED = {"Lexicon baseline": {r: cells(o["labels"]) for r, o in load(os.path.join(ROOT, "pred_lex_gold.jsonl")).items()},
         "Qwen2.5-7B zero-shot (lenient)": {r: cells(o.get("labels_lenient")) for r, o in qraw.items()},
         "Qwen2.5-14B zero-shot (lenient)": {r: cells(o.get("labels_lenient")) for r, o in load(os.path.join(REV, "preds_qwen_scale", "qwen25_14b_zeroshot.jsonl")).items()},
         "Gemini draft (silver)": {r: cells(o["labels"]) for r, o in load(os.path.join(ROOT, "pred_gemini_fewshot.jsonl")).items()}}
pr = lambda sid, s: {r: cells(o["labels"]) for r, o in load(os.path.join(REV, "preds_seeds", f"{sid}_{s}.jsonl")).items()}

res = {}      # name -> list over seeds of (d, lo, hi, p, holm)
neu = {}
for s in SEEDS:
    hi = pr("hiabsa", s); chi = per_review(hi)
    names, rows = [], []
    for name, sid in TRAINED:
        p = pr(sid, s); names.append(name); rows.append(test(chi, per_review(p), s))
        neu.setdefault(name, []).append(neu_f1(p))
    for name, p in FIXED.items():
        names.append(name); rows.append(test(chi, per_review(p), s))
    neu.setdefault("HI-ABSA", []).append(neu_f1(hi))
    hp = holm([x[3] for x in rows])
    for n, x, h in zip(names, rows, hp):
        res.setdefault(n, []).append((*x, h))
    print("seed", s, "done", flush=True)

L = ["# E8 — Review-level paired tests per seed (HI-ABSA vs each system, Joint-F1, test-400)", "",
     f"For each seed, HI-ABSA of that seed is compared with the same-seed run of each trained system (deterministic "
     f"systems are the same in every seed). Paired bootstrap ({B}) and sign-flip permutation ({P}) over whole reviews; "
     "Holm correction over the 13 comparisons within each seed. Smallest attainable p = 0.0002.", "",
     "| HI-ABSA vs | Δ mean | Δ range over seeds | 95% CI (widest) | p range | Holm p range | seeds significant (Holm < .05) |",
     "|---|---:|---|---|---|---|---:|"]
for n, xs in res.items():
    d = [x[0] for x in xs]; lo = min(x[1] for x in xs); hi_ = max(x[2] for x in xs)
    L.append(f"| {n} | {statistics.mean(d):+.3f} | [{min(d):+.3f}, {max(d):+.3f}] | [{lo:+.3f}, {hi_:+.3f}] | "
             f"{min(x[3] for x in xs):.4f}–{max(x[3] for x in xs):.4f} | {min(x[4] for x in xs):.4f}–{max(x[4] for x in xs):.4f} | "
             f"{sum(x[4] < .05 for x in xs)}/5 |")
L += ["", "## F1 on the neutral cell class (4-class cells, test-400), mean ± SD over seeds", "",
      "| System | F1-NEU |", "|---|---:|"]
for n, v in neu.items():
    L.append(f"| {n} | {statistics.mean(v):.3f} ± {statistics.pstdev(v):.3f} |")
for n, p in FIXED.items():
    L.append(f"| {n} | {neu_f1(p):.3f} |")
open(os.path.join(REV, "results", "E8_per_seed_tests.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
