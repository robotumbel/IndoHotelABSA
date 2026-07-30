# -*- coding: utf-8 -*-
"""
make_corpus_figures.py — Figur statistik korpus IndoHotelABSA (dari data nyata).
Menghasilkan PDF siap-LaTeX: distribusi rating, panjang review, cakupan wilayah.
"""
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

INFILE = "dataset_merged.jsonl"
NAVY = "#143a5c"

# Peta kota -> wilayah (untuk agregasi cakupan)
REGION = {
    # Sumatera
    "banda_aceh":"Sumatra","medan":"Sumatra","padang":"Sumatra","bukittinggi":"Sumatra",
    "pekanbaru":"Sumatra","batam":"Sumatra","tanjung_pinang":"Sumatra","jambi":"Sumatra",
    "palembang":"Sumatra","bengkulu":"Sumatra","bandar_lampung":"Sumatra","pangkal_pinang":"Sumatra",
    # Jawa
    "jakarta":"Java","bogor":"Java","depok":"Java","tangerang":"Java","bekasi":"Java",
    "bandung":"Java","cirebon":"Java","sukabumi":"Java","semarang":"Java","solo":"Java",
    "magelang":"Java","yogyakarta":"Java","surabaya":"Java","malang":"Java","batu":"Java",
    "banyuwangi":"Java","serang":"Java",
    # Bali & Nusa Tenggara
    "denpasar":"Bali & Nusa Tenggara","ubud":"Bali & Nusa Tenggara","kuta":"Bali & Nusa Tenggara",
    "seminyak":"Bali & Nusa Tenggara","nusa_dua":"Bali & Nusa Tenggara","mataram":"Bali & Nusa Tenggara",
    "senggigi":"Bali & Nusa Tenggara","labuan_bajo":"Bali & Nusa Tenggara","kupang":"Bali & Nusa Tenggara",
    # Kalimantan
    "pontianak":"Kalimantan","palangkaraya":"Kalimantan","banjarmasin":"Kalimantan",
    "balikpapan":"Kalimantan","samarinda":"Kalimantan","tarakan":"Kalimantan",
    # Sulawesi
    "manado":"Sulawesi","palu":"Sulawesi","makassar":"Sulawesi","kendari":"Sulawesi",
    "gorontalo":"Sulawesi","mamuju":"Sulawesi",
    # Maluku & Papua
    "ambon":"Maluku & Papua","ternate":"Maluku & Papua","jayapura":"Maluku & Papua",
    "sorong":"Maluku & Papua","manokwari":"Maluku & Papua",
}
REGION_ORDER = ["Sumatra","Java","Bali & Nusa Tenggara","Kalimantan","Sulawesi","Maluku & Papua"]

# ── muat data ────────────────────────────────────────────────
rows = []
with open(INFILE, encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
print(f"Total review: {len(rows)}")

ratings = [r.get("rating") for r in rows if r.get("rating") is not None]
lengths = [len(r.get("text", "")) for r in rows]
regions = Counter(REGION.get(r.get("city", ""), "Other") for r in rows)

# ── Fig 1: distribusi rating ────────────────────────────────
rc = Counter(int(x) for x in ratings)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
xs = [1, 2, 3, 4, 5]
ys = [rc.get(x, 0) for x in xs]
bars = ax.bar(xs, ys, color=NAVY, width=0.65)
for b, y in zip(bars, ys):
    ax.text(b.get_x()+b.get_width()/2, y+80, f"{y:,}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Star rating"); ax.set_ylabel("Number of reviews")
ax.set_xticks(xs); ax.set_ylim(0, max(ys)*1.15)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_rating_dist.pdf"); plt.close()
print("fig_rating_dist.pdf")

# ── Fig 2: distribusi panjang review ────────────────────────
fig, ax = plt.subplots(figsize=(4.2, 3.0))
capped = [min(l, 1000) for l in lengths]
ax.hist(capped, bins=40, color=NAVY, alpha=0.85)
mean_len = sum(lengths)/len(lengths)
ax.axvline(mean_len, color="#b25a00", linestyle="--", linewidth=1.5,
           label=f"mean = {mean_len:.0f}")
ax.set_xlabel("Review length (characters, capped at 1000)")
ax.set_ylabel("Number of reviews"); ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_length_dist.pdf"); plt.close()
print("fig_length_dist.pdf")

# ── Fig 3: cakupan per-wilayah ──────────────────────────────
fig, ax = plt.subplots(figsize=(4.6, 3.0))
labels = [r for r in REGION_ORDER if regions.get(r)]
vals = [regions[r] for r in labels]
bars = ax.barh(labels[::-1], vals[::-1], color=NAVY)
for b, v in zip(bars, vals[::-1]):
    ax.text(v+40, b.get_y()+b.get_height()/2, f"{v:,}", va="center", fontsize=8)
ax.set_xlabel("Number of reviews")
ax.set_xlim(0, max(vals)*1.15)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_region_coverage.pdf"); plt.close()
print("fig_region_coverage.pdf")

# ── ringkasan angka untuk teks paper ────────────────────────
print("\n=== Angka untuk paper ===")
print(f"Total review        : {len(rows):,}")
print(f"Mean length         : {mean_len:.0f} karakter")
print(f"Median length       : {sorted(lengths)[len(lengths)//2]}")
print(f"Rating distribution : {dict(sorted(rc.items()))}")
print(f"Region coverage     : {dict(regions)}")
print(f"Jumlah wilayah      : {len([r for r in REGION_ORDER if regions.get(r)])}")
