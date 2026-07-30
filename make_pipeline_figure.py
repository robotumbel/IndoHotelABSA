# -*- coding: utf-8 -*-
"""Flowchart pipeline IndoHotelABSA (desain serpentine, bersih) -> fig_pipeline.pdf.
Baris-1 kiri->kanan (1-5), turun vertikal, baris-2 kanan->kiri (6-10).
Tanpa panah diagonal; diwarnai per fase."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY = "#1f4e79"; ORANGE = "#b5561a"; GREEN = "#1e7a3c"; PURPLE = "#5b2a86"
FILL = {"data": "#e8f0f8", "anno": "#fbeadb", "model": "#e4f2e9", "app": "#eee4f6"}
EDGE = {"data": NAVY, "anno": ORANGE, "model": GREEN, "app": PURPLE}
ARR = "#8a8a8a"

stages = [
    ("1. Data Collection", "Google Places API\n55 cities, 6 regions", "data"),
    ("2. Preprocessing", "dedup, length filter,\nPII removal", "data"),
    ("3. Sampling", "class-balanced\n3,000 reviews", "data"),
    ("4. Pre-annotation", "LLM silver drafts\n(unverified)", "anno"),
    ("5. Annotation", "3 annotators verify\ngold, κ = 0.894", "anno"),
    ("6. Dataset Split", "2,480 silver train\n100 val / 400 test", "model"),
    ("7. Training", "encoders, HI-ABSA,\nLLM (LoRA/zero-shot)", "model"),
    ("8. Testing", "400-review\ngold test set", "model"),
    ("9. Evaluation", "ACD/ACP/Joint,\nbootstrap, McNemar", "model"),
    ("10. CRM Framework", "dashboard +\ndual-signal ICDA", "app"),
]

# ── geometri ───────────────────────────────────────────────
BW, BH = 17.0, 12.0
MX, GAP = 3.0, 2.0
PITCH = BW + GAP                      # 19
cols_x = [MX + c * PITCH for c in range(5)]   # 3,22,41,60,79
Y_TOP, Y_BOT = 26.0, 4.0
# kolom per-stage: baris1 col0..4 ; baris2 col4..0 (serpentine)
col_of = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
row_y = [Y_TOP] * 5 + [Y_BOT] * 5

fig, ax = plt.subplots(figsize=(10.0, 5.0))
ax.set_xlim(0, 100); ax.set_ylim(-2, 44); ax.axis("off")

cx, cy = [], []
for k, (title, sub, phase) in enumerate(stages):
    x = cols_x[col_of[k]]; y = row_y[k]
    ax.add_patch(FancyBboxPatch((x, y), BW, BH,
                 boxstyle="round,pad=0.35,rounding_size=1.6",
                 linewidth=1.6, edgecolor=EDGE[phase], facecolor=FILL[phase]))
    ax.text(x + BW / 2, y + BH - 3.4, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=EDGE[phase])
    ax.text(x + BW / 2, y + 3.9, sub, ha="center", va="center",
            fontsize=7.7, color="#333333")
    cx.append(x + BW / 2); cy.append(y + BH / 2)

def arrow(p1, p2, rad=0.0):
    style = f"arc3,rad={rad}" if rad else "arc3"
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                 linewidth=1.7, color=ARR, connectionstyle=style,
                 shrinkA=0, shrinkB=0))

# baris 1: kiri->kanan (0->1->2->3->4)
for k in range(4):
    x = cols_x[col_of[k]]; nx = cols_x[col_of[k + 1]]
    arrow((x + BW, Y_TOP + BH / 2), (nx, Y_TOP + BH / 2))
# konektor vertikal 5 -> 6 (kolom 4)
xc = cols_x[4] + BW / 2
arrow((xc, Y_TOP), (xc, Y_BOT + BH))
# baris 2: kanan->kiri (5->6->7->8->9)
for k in range(5, 9):
    x = cols_x[col_of[k]]; nx = cols_x[col_of[k + 1]]
    arrow((x, Y_BOT + BH / 2), (nx + BW, Y_BOT + BH / 2))

# legenda fase
handles = [mp.Patch(facecolor=FILL[p], edgecolor=EDGE[p], label=lab) for p, lab in
           [("data", "Data construction"), ("anno", "Annotation"),
            ("model", "Modelling & evaluation"), ("app", "Application")]]
ax.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.5,
          frameon=False, bbox_to_anchor=(0.5, 1.06), handlelength=1.2,
          columnspacing=1.6)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig("fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.05)
print("fig_pipeline.pdf ditulis (serpentine).")
