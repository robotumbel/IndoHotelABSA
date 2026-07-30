"""
compute_kappa.py — Inter-Annotator Agreement (IAA) untuk IndoHotelABSA.

Menghitung Cohen's Kappa (2 anotator) atau Fleiss' Kappa (>=3 anotator)
pada label ABSA. Tiap sel (review x aspek) diperlakukan sebagai satu kategori
dalam {NA, POSITIF, NEGATIF, NETRAL}.

Input : >=2 file JSONL dari anotator berbeda (review_id sama, field 'labels').
Pemakaian:
  python compute_kappa.py anotatorA.jsonl anotatorB.jsonl
  python compute_kappa.py A.jsonl B.jsonl C.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
CATS = ["NA", "POSITIF", "NEGATIF", "NETRAL"]


def load(path):
    d = {}
    with open(path, encoding="utf-8-sig") as f:
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
            m = {a: "NA" for a in ASPECTS}
            for lab in o.get("labels", []):
                if lab.get("aspect") in m:
                    m[lab["aspect"]] = lab.get("polarity", "NA")
            d[o["review_id"]] = m
    return d


def cohen_kappa(pairs):
    """pairs: list of (labelA, labelB). Cohen's Kappa."""
    n = len(pairs)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    # marginal
    ca = defaultdict(int); cb = defaultdict(int)
    for a, b in pairs:
        ca[a] += 1; cb[b] += 1
    pe = sum((ca[c]/n) * (cb[c]/n) for c in CATS)
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def fleiss_kappa(rows):
    """rows: list of dict cat->count (jumlah anotator yang pilih tiap kategori)."""
    N = len(rows)
    if N == 0:
        return 0.0
    n = sum(rows[0].values())            # jumlah anotator per item
    if n <= 1:
        return 0.0
    # P_i per item
    Pi = []
    for r in rows:
        s = sum(v*v for v in r.values())
        Pi.append((s - n) / (n * (n - 1)))
    P_bar = sum(Pi) / N
    # p_j per kategori
    pj = {c: sum(r.get(c, 0) for r in rows) / (N * n) for c in CATS}
    Pe = sum(v*v for v in pj.values())
    return (P_bar - Pe) / (1 - Pe) if (1 - Pe) else 1.0


def interpret(k):
    if k < 0.0: return "Poor (di bawah kebetulan)"
    if k < 0.20: return "Slight"
    if k < 0.40: return "Fair"
    if k < 0.60: return "Moderate"
    if k < 0.80: return "Substantial (target minimal)"
    return "Almost perfect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="2+ file JSONL anotator")
    args = ap.parse_args()
    assert len(args.files) >= 2, "Butuh minimal 2 file anotator"

    anns = [load(f) for f in args.files]
    common = set(anns[0])
    for a in anns[1:]:
        common &= set(a)
    common = sorted(common)
    print(f"Anotator: {len(anns)} | review beririsan: {len(common)}")
    if not common:
        print("Tidak ada review_id yang sama antar-anotator."); return

    # --- keseluruhan + per-aspek ---
    if len(anns) == 2:
        A, B = anns
        allpairs = []
        print("\nCohen's Kappa per aspek:")
        for asp in ASPECTS:
            pairs = [(A[r][asp], B[r][asp]) for r in common]
            k = cohen_kappa(pairs)
            allpairs += pairs
            print(f"  {asp:<22} {k:.3f}")
        overall = cohen_kappa(allpairs)
        print(f"\n  {'KESELURUHAN':<22} {overall:.3f}  -> {interpret(overall)}")
    else:
        print("\nFleiss' Kappa per aspek:")
        all_rows = []
        for asp in ASPECTS:
            rows = []
            for r in common:
                cnt = {c: 0 for c in CATS}
                for a in anns:
                    cnt[a[r][asp]] += 1
                rows.append(cnt)
            k = fleiss_kappa(rows)
            all_rows += rows
            print(f"  {asp:<22} {k:.3f}")
        overall = fleiss_kappa(all_rows)
        print(f"\n  {'KESELURUHAN':<22} {overall:.3f}  -> {interpret(overall)}")

    print("\nCatatan: target κ > 0,70 untuk dataset berkualitas Q1.")


if __name__ == "__main__":
    main()
