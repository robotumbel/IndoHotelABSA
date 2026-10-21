# -*- coding: utf-8 -*-
"""Score zero-shot LLM predictions on test-400 (adjudicated gold) and blind-100
(adjudicated gold vs blind majority), next to the Gemini draft and the encoders.

Usage: python scripts/score_llm_blind.py preds_llm/qwen25_7b_zeroshot.jsonl [more ...]
Writes results/E9_llm_blind.md
"""
import json, os, random, sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
POLS = ["POSITIF", "NEGATIF", "NETRAL"]


def load(p):
    return {o["review_id"]: o for o in (json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip())}


def cells(labels):
    m = {a: "NA" for a in ASPECTS}
    for l in labels or []:
        if l.get("aspect") in m and l.get("polarity") in POLS:
            m[l["aspect"]] = l["polarity"]
    return m


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def jt(g, p, r, ign):
    t = fp = fn = 0
    for a in ASPECTS:
        if (r, a) in ign: continue
        if g[r][a] != "NA" and g[r][a] == p[r][a]: t += 1
        else:
            fp += p[r][a] != "NA"; fn += g[r][a] != "NA"
    return t, fp, fn


def joint(g, p, ids, ign=frozenset()):
    s = [jt(g, p, r, ign) for r in ids]
    return f1(sum(x[0] for x in s), sum(x[1] for x in s), sum(x[2] for x in s))


def acd(g, p, ids, ign=frozenset()):
    out = []
    for a in ASPECTS:
        tp = fp = fn = 0
        for r in ids:
            if (r, a) in ign: continue
            gm, pm = g[r][a] != "NA", p[r][a] != "NA"
            tp += gm and pm; fp += pm and not gm; fn += gm and not pm
        out.append(f1(tp, fp, fn))
    return sum(out) / len(out)


test = load(os.path.join(ROOT, "test_gold.jsonl"))
gold = {r: cells(o["labels"]) for r, o in test.items()}
IDS = sorted(gold)
ids_b = [int(x) for x in open(os.path.join(BLIND, "blind_subset_ids.txt")) if x.strip() and not x.startswith("#")]
bl = [{r: cells(o["labels"]) for r, o in load(os.path.join(BLIND, BLIND_FILE.format(i=i))).items()} for i in (1, 2, 3)]
bgold, ties = {}, set()
for r in ids_b:
    bgold[r] = {}
    for a in ASPECTS:
        lab, n = Counter(b[r][a] for b in bl).most_common(1)[0]
        if n == 1: ties.add((r, a)); lab = "NA"
        bgold[r][a] = lab

systems = {"Gemini draft (silver)": {r: cells(o["labels"]) for r, o in load(os.path.join(ROOT, "pred_gemini_fewshot.jsonl")).items()},
           "IndoBERT (seed 42)": {r: cells(o["labels"]) for r, o in load(os.path.join(REV, "preds_seeds", "indobert_42.jsonl")).items()},
           "HI-ABSA (seed 42)": {r: cells(o["labels"]) for r, o in load(os.path.join(REV, "preds_seeds", "hiabsa_42.jsonl")).items()}}
compliance = {}
for path in sys.argv[1:]:
    raw = load(os.path.join(REV, path) if not os.path.isabs(path) else path)
    name = os.path.basename(path).replace(".jsonl", "")
    systems[f"{name} (strict parser)"] = {r: cells(o["labels"]) for r, o in raw.items()}
    systems[f"{name} (lenient parser)"] = {r: cells(o.get("labels_lenient")) for r, o in raw.items()}
    arr = sum(o["raw"].lstrip("`json \n").startswith("[") for o in raw.values())
    compliance[name] = {"n": len(raw), "strict_nonempty": sum(bool(o["labels"]) for o in raw.values()),
                        "lenient_nonempty": sum(bool(o.get("labels_lenient")) for o in raw.values()),
                        "starts_with_array": arr}

rng = random.Random(0); B = 2000
boots = [[rng.choice(ids_b) for _ in ids_b] for _ in range(B)]


def gap_ci(a, b, g):
    obs = joint(g, systems[a], ids_b, ties) - joint(g, systems[b], ids_b, ties)
    d = sorted(joint(g, systems[a], s, ties) - joint(g, systems[b], s, ties) for s in boots)
    return obs, d[int(.025 * B)], d[int(.975 * B)]


L = ["# E9 — Zero-shot LLM not used for drafting, on test-400 and blind-100", "",
     "Encoders shown with seed-42 predictions (5-seed means come with E5).", "",
     "| System | Joint-F1 test-400 | ACD-mF1 test-400 | blind-100 vs gold | blind-100 vs blind | Δ | ACD-mF1 vs blind |",
     "|---|---:|---:|---:|---:|---:|---:|"]
for n, p in systems.items():
    a, b = joint(gold, p, ids_b, ties), joint(bgold, p, ids_b, ties)
    L.append(f"| {n} | {joint(gold, p, IDS):.3f} | {acd(gold, p, IDS):.3f} | {a:.3f} | {b:.3f} | {b - a:+.3f} | {acd(bgold, p, ids_b, ties):.3f} |")
L += ["", "Format compliance of the open-LLM outputs:", ""]
for n, c in compliance.items():
    L.append(f"- {n}: {c['n']} reviews; output begins with a JSON array in {c['starts_with_array']}; "
             f"strict parser non-empty {c['strict_nonempty']}; lenient parser non-empty {c['lenient_nonempty']}.")
L += ["", "Review-level bootstrap on blind-100 (2,000 resamples), Joint-F1 gap:", "",
      "| Comparison | vs adjudicated gold | vs blind majority |", "|---|---|---|"]
for n in [k for k in systems if "lenient" in k]:
    for ref in ["Gemini draft (silver)", "IndoBERT (seed 42)"]:
        g1, g2 = gap_ci(n, ref, gold), gap_ci(n, ref, bgold)
        L.append(f"| {n} − {ref} | {g1[0]:+.3f} [{g1[1]:+.3f}, {g1[2]:+.3f}] | {g2[0]:+.3f} [{g2[1]:+.3f}, {g2[2]:+.3f}] |")
if sys.argv[1:]:   # when imported by blind100_seeds.py (no files) do not overwrite the report
    os.makedirs(os.path.join(REV, "results"), exist_ok=True)
    open(os.path.join(REV, "results", "E9_llm_blind.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
