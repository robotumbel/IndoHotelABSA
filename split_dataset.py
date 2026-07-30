"""
split_dataset.py — Bagi dataset teranotasi menjadi train/val/test (70/15/15),
stratified berdasarkan kelompok sentimen dominan tiap review.

Input : file JSONL teranotasi FINAL (mis. annotation_gold.jsonl)
Output: train.jsonl, val.jsonl, test.jsonl + statistik

Pemakaian:
  python split_dataset.py --infile annotation_gold.jsonl
"""
import argparse
import json
import random
from collections import Counter

random.seed(42)


def dominant_bucket(labels):
    """Kelompok stratifikasi: berdasarkan polaritas mayoritas review."""
    pol = [l.get("polarity") for l in labels]
    if not pol:
        return "none"
    c = Counter(pol)
    neg, pos = c.get("NEGATIF", 0), c.get("POSITIF", 0)
    if neg > pos:
        return "neg"
    if pos > neg:
        return "pos"
    return "mix"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="annotation_gold.jsonl")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--only_verified", action="store_true",
                    help="hanya pakai review dengan verified=true")
    args = ap.parse_args()

    rows = []
    with open(args.infile, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if "review_id" not in o:
                continue
            if args.only_verified and not o.get("verified"):
                continue
            rows.append(o)

    print(f"Total review dipakai: {len(rows)}")

    # stratifikasi per bucket
    groups = {}
    for r in rows:
        groups.setdefault(dominant_bucket(r.get("labels", [])), []).append(r)

    train, val, test = [], [], []
    for g, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_tr = int(n * args.train)
        n_va = int(n * args.val)
        train += items[:n_tr]
        val += items[n_tr:n_tr + n_va]
        test += items[n_tr + n_va:]

    random.shuffle(train); random.shuffle(val); random.shuffle(test)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        with open(f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # statistik aspek
    def aspect_count(part):
        c = Counter()
        for r in part:
            for l in r.get("labels", []):
                c[l["aspect"]] += 1
        return c

    print(f"\nSplit: train={len(train)}  val={len(val)}  test={len(test)}")
    print("\nDistribusi bucket sentimen:")
    for g, items in groups.items():
        print(f"  {g:<6}: {len(items)}")
    print("\nAnotasi aspek di train:")
    for a, n in aspect_count(train).most_common():
        print(f"  {a:<22} {n}")
    print("\nFile: train.jsonl, val.jsonl, test.jsonl siap untuk eksperimen.")


if __name__ == "__main__":
    main()
