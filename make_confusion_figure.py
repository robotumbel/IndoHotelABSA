# -*- coding: utf-8 -*-
"""Confusion matrix 4-arah HI-ABSA (N/A, POS, NEG, NEU) -> fig_confusion.pdf.
Warna dinormalisasi per-baris (recall) agar kelas langka (NEU) tetap terbaca;
anotasi memakai jumlah mentah."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = np.array(json.load(open("cm_data.json")))            # (4,4) rows=true, cols=pred
labels = ["N/A", "POS", "NEG", "NEU"]
row_sum = M.sum(1, keepdims=True)
norm = M / np.clip(row_sum, 1, None)                     # recall per baris

fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.set_title("HI-ABSA confusion matrix (gold test cells)", fontsize=10)
for i in range(4):
    for j in range(4):
        c = "white" if norm[i, j] > 0.55 else "#222222"
        ax.text(j, i, f"{M[i, j]}\n{norm[i, j]*100:.0f}%", ha="center", va="center",
                fontsize=9, color=c)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("row-normalised (recall)", fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
ax.grid(which="minor", color="w", linewidth=1.2); ax.tick_params(which="minor", length=0)
plt.tight_layout()
plt.savefig("fig_confusion.pdf", bbox_inches="tight")
print("fig_confusion.pdf ditulis.")
print("NEU recall:", f"{norm[3,3]*100:.0f}%", "| NEU->POS:", M[3,1], "of", row_sum[3,0])
