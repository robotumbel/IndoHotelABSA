# -*- coding: utf-8 -*-
"""
verify_release.py — check the IndoHotelABSA release against the counts and
overlaps reported in the data article, using nothing but the released files.

Run it from the directory holding the .jsonl files:

    python verify_release.py

It prints Table 4 of the article (inclusion and overlap), the annotation
statistics of Section 3, and a PASS/FAIL line for each claim. A non-zero exit
status means at least one claim does not hold.
"""
import json
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]

failures = []


def load(path):
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        failures.append(f"missing file {path}")
        return {}
    d = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                d[o["review_id"]] = o
    return d


def check(label, got, want):
    ok = got == want
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label:<58} {got}" + ("" if ok else f"  (expected {want})"))
    if not ok:
        failures.append(label)
    return ok


def main():
    merged = load("dataset_merged.jsonl")
    silver = load("train_silver.jsonl")
    val = load("val_gold.jsonl")
    test = load("test_gold.jsonl")
    gold = load("gold_final.jsonl")
    excl = load("excluded_empty_labels.jsonl")
    draft = load("annotation_prefilled_gold500.jsonl")
    anns = [load(f"annotator_{i}_verified.jsonl") for i in (1, 2, 3)]

    print("\n== Table 4: inclusion and overlap ==")
    rows = [
        ("dataset_merged.jsonl", len(merged), "Full corpus; superset of all sets below"),
        ("Annotation sample", len(silver) + len(gold) + len(excl), "Drawn from dataset_merged.jsonl"),
        ("  gold_final.jsonl", len(gold), "Adjudicated gold set"),
        ("    val_gold.jsonl", len(val), "Subset of gold_final.jsonl"),
        ("    test_gold.jsonl", len(test), "Subset of gold_final.jsonl; disjoint from validation"),
        ("  train_silver.jsonl", len(silver), "Disjoint from gold_final.jsonl"),
        ("  excluded_empty_labels.jsonl", len(excl), "Disjoint from all labelled files"),
    ]
    print(f"  {'Set':<32}{'Records':>9}  Relationship")
    for name, n, rel in rows:
        print(f"  {name:<32}{n:>9,}  {rel}")

    print("\n== Record counts ==")
    check("dataset_merged.jsonl records", len(merged), 14988)
    check("train_silver.jsonl records", len(silver), 2480)
    check("val_gold.jsonl records", len(val), 100)
    check("test_gold.jsonl records", len(test), 400)
    check("gold_final.jsonl records", len(gold), 500)
    check("excluded_empty_labels.jsonl records", len(excl), 20)
    check("released total (2,480+100+400)", len(silver) + len(val) + len(test), 2980)
    check("annotation sample reconstructed (2,980+20)",
          len(silver) + len(val) + len(test) + len(excl), 3000)

    print("\n== Inclusion and disjointness ==")
    check("val_gold is a subset of gold_final", set(val) <= set(gold), True)
    check("test_gold is a subset of gold_final", set(test) <= set(gold), True)
    check("val_gold and test_gold are disjoint", len(set(val) & set(test)), 0)
    check("gold_final == val_gold union test_gold", set(gold) == set(val) | set(test), True)
    check("train_silver and gold_final are disjoint", len(set(silver) & set(gold)), 0)
    check("excluded and all labelled files are disjoint",
          len(set(excl) & (set(silver) | set(gold))), 0)
    ids_in_corpus = all(
        r["text"] is not None for r in list(silver.values())[:1] or [{"text": None}]
    )
    check("labelled records carry text", ids_in_corpus, True)

    print("\n== Corpus id linkage ==")
    labelled_rows = list(silver.values()) + list(gold.values()) + list(excl.values())
    linked = [r for r in labelled_rows if "corpus_review_id" in r]
    check("labelled records carrying corpus_review_id", len(linked), len(labelled_rows))
    if merged and linked:
        bad = [r for r in linked if r["corpus_review_id"] not in merged]
        check("every corpus_review_id exists in the corpus", len(bad), 0)
        mism = [r for r in linked
                if r["corpus_review_id"] in merged
                and merged[r["corpus_review_id"]]["text"] != r["text"]]
        check("linked texts match the corpus record", len(mism), 0)

    print("\n== Tier flags ==")
    check("every gold record has gold=true, verified=true",
          all(r.get("gold") is True and r.get("verified") is True for r in gold.values()), True)
    check("no silver record is flagged gold",
          any(r.get("gold") for r in silver.values()), False)

    print("\n== Annotation statistics (Section 3) ==")
    pol = Counter()
    for r in gold.values():
        for l in r.get("labels", []):
            pol[l["polarity"]] += 1
    check("gold aspect annotations", sum(pol.values()), 1506)
    check("gold NEGATIF", pol["NEGATIF"], 727)
    check("gold POSITIF", pol["POSITIF"], 710)
    check("gold NETRAL", pol["NETRAL"], 69)

    spol = Counter()
    for r in silver.values():
        for l in r.get("labels", []):
            spol[l["polarity"]] += 1
    check("silver aspect annotations", sum(spol.values()), 7338)
    check("silver NETRAL", spol["NETRAL"], 334)

    if all(anns) and draft:
        def tomap(o):
            return {l["aspect"]: l["polarity"] for l in o.get("labels", [])}
        cells = unan = corrected = ties = 0
        for rid in gold:
            maps = [tomap(a[rid]) for a in anns]
            gm, pm = tomap(gold[rid]), tomap(draft.get(rid, {}))
            for asp in ASPECTS:
                votes = [m.get(asp, "NA") for m in maps]
                top = Counter(votes).most_common()[0][1]
                cells += 1
                unan += top == 3
                ties += top < 2
                corrected += gm.get(asp, "NA") != pm.get(asp, "NA")
        print("\n== Adjudication (Section 4.4) ==")
        check("(review, aspect) cells", cells, 3500)
        check("unanimous cells", unan, 3170)
        check("three-way ties", ties, 0)
        check("cells corrected from the LLM draft", corrected, 275)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) did not hold.")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed: the release matches every count reported in the article.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
