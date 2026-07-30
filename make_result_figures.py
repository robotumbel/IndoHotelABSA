# -*- coding: utf-8 -*-
"""Grafik hasil eksperimen IndoHotelABSA (dari angka nyata) -> PDF utk IEEE."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

NAVY = "#143a5c"; ORANGE = "#b25a00"; GREEN = "#1b5e20"; GREY = "#8a8a8a"

# ── data nyata (Joint-F1, ACD-mF1, ACP-Acc) ─────────────────
models = ["Lexicon", "mBERT", "XLM-R", "IndoBERTweet", "IndoBERT",
          "HI-ABSA", "LLM-LoRA\n(0.5B)", "LLM zero-shot\n(Gemini)"]
joint  = [0.428, 0.670, 0.670, 0.710, 0.756, 0.751, 0.684, 0.872]
acd    = [0.794, 0.771, 0.760, 0.802, 0.850, 0.853, 0.745, 0.933]
acp    = [0.526, 0.840, 0.859, 0.858, 0.872, 0.866, 0.881, 0.926]

colors = [GREY, NAVY, NAVY, NAVY, NAVY, ORANGE, GREEN, GREEN]

# ── Fig 1: Joint-F1 comparison (bar) ────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 3.2))
x = np.arange(len(models))
bars = ax.bar(x, joint, color=colors, width=0.66)
for b, v in zip(bars, joint):
    ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8, rotation=20, ha="right")
ax.set_ylabel("Joint-F1"); ax.set_ylim(0, 1.0)
ax.axhline(0.756, color="k", ls="--", lw=0.8, alpha=0.5)
ax.text(0.1, 0.77, "best encoder", fontsize=7, alpha=0.7)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_model_joint.pdf"); plt.close()
print("fig_model_joint.pdf")

# ── Fig 2: multi-metric grouped bar ─────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 3.2))
w = 0.27
ax.bar(x-w, acd, w, label="ACD macro-F1", color=NAVY)
ax.bar(x,   acp, w, label="ACP accuracy", color=ORANGE)
ax.bar(x+w, joint, w, label="Joint-F1", color=GREEN)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8, rotation=20, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8, ncol=3, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_model_multi.pdf"); plt.close()
print("fig_model_multi.pdf")

# ── Fig 3: per-aspect F1 (HI-ABSA) ──────────────────────────
asp = ["Service", "Room & Fac.", "Price", "Food & Bev.", "Location",
       "Cleanliness", "Public Fac."]
f1a = [0.922, 0.912, 0.908, 0.872, 0.800, 0.793, 0.767]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
bars = ax.barh(asp[::-1], f1a[::-1], color=NAVY)
for b, v in zip(bars, f1a[::-1]):
    ax.text(v+0.008, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("Detection F1"); ax.set_xlim(0, 1.0)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_per_aspect.pdf"); plt.close()
print("fig_per_aspect.pdf")

# ── Fig 4: error-type breakdown (HI-ABSA vs Gemini) ─────────
cats = ["Missed\n(FN)", "Spurious\n(FP)", "Polarity\nflip"]
hi   = [111, 226, 147]
gem  = [93, 45, 83]
fig, ax = plt.subplots(figsize=(5.0, 3.0))
xx = np.arange(len(cats)); w = 0.36
ax.bar(xx-w/2, hi,  w, label="HI-ABSA", color=ORANGE)
ax.bar(xx+w/2, gem, w, label="LLM zero-shot", color=GREEN)
for i, (h, g) in enumerate(zip(hi, gem)):
    ax.text(i-w/2, h+4, str(h), ha="center", fontsize=8)
    ax.text(i+w/2, g+4, str(g), ha="center", fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("Error cells (of 2800)"); ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig_error_breakdown.pdf"); plt.close()
print("fig_error_breakdown.pdf")

print("\n4 grafik hasil dibuat.")
