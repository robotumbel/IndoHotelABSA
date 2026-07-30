"""
sample_for_annotation.py — Ambil subset SEIMBANG dari dataset_merged.jsonl
untuk dianotasi manual. Stratifikasi berdasarkan rating (agar cukup contoh
negatif/netral) dan disebar merata antar-kota.

Input : dataset_merged.jsonl
Output: annotation_sample.jsonl  (siap dibagi ke anotator)

Pemakaian:
  python sample_for_annotation.py --n 3000
"""
import argparse
import json
import random
from collections import defaultdict

random.seed(42)

# Target proporsi per kelompok rating (agar sentimen seimbang untuk ABSA)
#   negatif (1-2), netral (3), positif (4-5)
PROPORSI = {"negatif": 0.40, "netral": 0.20, "positif": 0.40}


def bucket(rating):
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return "positif"          # rating hilang -> anggap positif (mayoritas)
    if r <= 2:
        return "negatif"
    if r == 3:
        return "netral"
    return "positif"


def spread_by_city(items, k):
    """Ambil k item, disebar merata antar-kota (round-robin)."""
    by_city = defaultdict(list)
    for it in items:
        by_city[it["city"]].append(it)
    for c in by_city:
        random.shuffle(by_city[c])
    order = list(by_city.keys())
    random.shuffle(order)
    picked, idx = [], 0
    while len(picked) < k and any(by_city[c] for c in order):
        c = order[idx % len(order)]
        if by_city[c]:
            picked.append(by_city[c].pop())
        idx += 1
    return picked[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="dataset_merged.jsonl")
    ap.add_argument("--out", default="annotation_sample.jsonl")
    ap.add_argument("--n", type=int, default=3000, help="jumlah review yang diambil")
    args = ap.parse_args()

    rows = []
    with open(args.infile, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    groups = defaultdict(list)
    for r in rows:
        groups[bucket(r.get("rating"))].append(r)

    print("Tersedia per kelompok:")
    for g in ["negatif", "netral", "positif"]:
        print(f"  {g:<8}: {len(groups[g]):,}")

    sample = []
    for g, prop in PROPORSI.items():
        target = int(args.n * prop)
        avail = groups[g]
        take = min(target, len(avail))
        sample.extend(spread_by_city(avail, take))
        if take < target:
            print(f"  [info] {g}: hanya {take} tersedia (target {target})")

    random.shuffle(sample)
    # nomori ulang & sisakan kolom yang relevan untuk anotasi
    out = []
    for i, r in enumerate(sample, 1):
        out.append({
            "review_id": i,
            "hotel": r.get("hotel", ""),
            "city": r.get("city", ""),
            "rating": r.get("rating"),
            "text": r["text"],
            "labels": []            # <- diisi anotator: [{"aspect":..,"polarity":..}]
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ringkasan
    from collections import Counter
    b = Counter(bucket(r["rating"]) for r in out)
    print(f"\nSampel anotasi: {len(out):,} review -> {args.out}")
    print(f"  negatif={b['negatif']}  netral={b['netral']}  positif={b['positif']}")
    print(f"  kota berbeda: {len({r['city'] for r in out})}")
    print("\nLangkah berikut: bagi file ini ke anotator (lihat Panduan Anotasi),")
    print("isi field 'labels' untuk tiap review.")


if __name__ == "__main__":
    main()
