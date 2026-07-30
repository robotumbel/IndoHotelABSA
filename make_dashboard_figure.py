# -*- coding: utf-8 -*-
"""
make_dashboard_figure.py — Artefak explainability UTAMA: dashboard CRM per-aspek.
Menginstansiasi Eq. net-sentiment s_a dan severity dari prediksi HI-ABSA nyata
pada test set -> fig_dashboard.pdf (dua panel: net sentiment + peringkat severitas).
"""
import sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")

ASPECTS_ID = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
              "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
ASPECTS_EN = ["Location", "Cleanliness", "Service", "Room & Fac.",
              "Price", "Food & Bev.", "Public Fac."]
NAVY = "#143a5c"; RED = "#b0202a"; GREEN = "#1b7a3a"; ORANGE = "#b25a00"; GREY = "#9a9a9a"


def load(p):
    d = {}
    for line in open(p, encoding="utf-8-sig"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        d[o["review_id"]] = o.get("labels", [])
    return d


pred = load("pred_hiabsa_full.jsonl")

# hitung n+, n-, n0 per aspek (Eq. net sentiment)
pos = {a: 0 for a in ASPECTS_ID}; neg = dict(pos); neu = dict(pos)
for labs in pred.values():
    for l in labs:
        a, p = l.get("aspect"), l.get("polarity")
        if a not in pos:
            continue
        if p == "POSITIF": pos[a] += 1
        elif p == "NEGATIF": neg[a] += 1
        elif p == "NETRAL": neu[a] += 1

s = {}; sev = {}
for a in ASPECTS_ID:
    tot = pos[a] + neg[a] + neu[a]
    s[a] = (pos[a] - neg[a]) / tot if tot else 0.0
    sev[a] = (1 - s[a]) * math.log(1 + neg[a])

order = sorted(ASPECTS_ID, key=lambda a: s[a])          # negatif dulu
en = {ai: ei for ai, ei in zip(ASPECTS_ID, ASPECTS_EN)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1))

# panel 1: net sentiment per aspek
labels = [en[a] for a in order]
vals = [s[a] for a in order]
cols = [RED if v < 0 else (GREEN if v > 0.15 else ORANGE) for v in vals]
ax1.barh(labels, vals, color=cols)
ax1.axvline(0, color="k", lw=0.8)
for i, v in enumerate(vals):
    ax1.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}",
             va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax1.set_xlim(-1, 1); ax1.set_xlabel("Net sentiment $s_a$")
ax1.set_title("(a) Per-aspect net sentiment", fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)

# panel 2: severity ranking (prioritas aksi CRM)
order2 = sorted(ASPECTS_ID, key=lambda a: sev[a], reverse=True)
labels2 = [en[a] for a in order2]
vals2 = [sev[a] for a in order2]
ax2.barh(labels2[::-1], vals2[::-1], color=NAVY)
for i, v in enumerate(vals2[::-1]):
    ax2.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8)
ax2.set_xlabel(r"Severity $(1-s_a)\log(1+n_a^-)$")
ax2.set_title("(b) CRM action priority", fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("fig_dashboard.pdf", bbox_inches="tight")
print("fig_dashboard.pdf ditulis.")
print("\nNet sentiment per aspek (dari prediksi HI-ABSA, test set):")
for a in order:
    print(f"  {en[a]:<14} s={s[a]:+.2f}  (pos={pos[a]}, neg={neg[a]}, neu={neu[a]})  severity={sev[a]:.2f}")
