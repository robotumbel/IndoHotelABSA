"""
merge_dataset.py — Gabung semua file scrape per-kota, dedup, bersihkan,
dan hasilkan statistik + file gabungan siap-anotasi.

Input : scrape_results/reviews_*.jsonl
Output: dataset_merged.jsonl   (review unik, ber-id, + kolom kota)
        dataset_stats.txt       (ringkasan statistik)

Pemakaian: python merge_dataset.py
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict

SRC_DIR = "scrape_results"
OUT_FILE = "dataset_merged.jsonl"
STATS_FILE = "dataset_stats.txt"

MIN_LEN = 20          # buang review terlalu pendek (karakter)
MAX_LEN = 2000        # buang yang ekstrem panjang (kemungkinan noise)


def norm_key(text):
    """Kunci dedup: lowercase + rapatkan spasi + potong."""
    return re.sub(r"\s+", " ", text.lower()).strip()[:200]


def city_from_filename(path):
    base = os.path.basename(path)
    m = re.match(r"reviews_(.+)\.jsonl", base)
    return m.group(1) if m else "unknown"


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "reviews_*.jsonl")))
    print(f"Menemukan {len(files)} file di {SRC_DIR}/")

    seen = set()
    merged = []
    per_city_raw = Counter()
    per_city_kept = Counter()
    rating_dist = Counter()
    dup_count = 0
    short_count = 0

    for path in files:
        city = city_from_filename(path)
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                per_city_raw[city] += 1
                text = (r.get("text") or "").strip()
                if len(text) < MIN_LEN or len(text) > MAX_LEN:
                    short_count += 1
                    continue
                k = norm_key(text)
                if k in seen:
                    dup_count += 1
                    continue
                seen.add(k)
                rating = r.get("rating")
                rating_dist[rating] += 1
                merged.append({
                    "review_id": len(merged) + 1,
                    "hotel": r.get("hotel", ""),
                    "city": city,
                    "rating": rating,
                    "text": text,
                    "source": r.get("source", "google_places"),
                })
                per_city_kept[city] += 1

    # tulis dataset gabungan
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # panjang teks
    lengths = [len(r["text"]) for r in merged]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    # statistik
    lines = []
    lines.append("=" * 55)
    lines.append("STATISTIK DATASET IndoHotelABSA (mentah, pra-anotasi)")
    lines.append("=" * 55)
    lines.append(f"File sumber          : {len(files)} kota")
    lines.append(f"Review mentah total  : {sum(per_city_raw.values()):,}")
    lines.append(f"Dibuang (pendek/pjg) : {short_count:,}")
    lines.append(f"Duplikat dibuang     : {dup_count:,}")
    lines.append(f"REVIEW UNIK FINAL    : {len(merged):,}")
    lines.append(f"Rata-rata panjang    : {avg_len:.0f} karakter")
    lines.append("")
    lines.append("Distribusi rating:")
    for rt in sorted(rating_dist, key=lambda x: (x is None, x)):
        lines.append(f"  {rt}: {rating_dist[rt]:,}")
    lines.append("")
    lines.append("Review unik per kota (top 20):")
    for city, n in per_city_kept.most_common(20):
        lines.append(f"  {city:<22} {n:>5}")
    lines.append("")
    lines.append(f"File gabungan: {OUT_FILE}")

    report = "\n".join(lines)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nStatistik disimpan -> {STATS_FILE}")


if __name__ == "__main__":
    main()
