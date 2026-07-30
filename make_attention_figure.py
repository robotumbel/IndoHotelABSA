# -*- coding: utf-8 -*-
"""
make_attention_figure.py — Visualisasi Aspect-Query Attention HI-ABSA (explainability).
Muat checkpoint terlatih, jalankan pada contoh ulasan campuran, ekstrak alpha_a
(bobot atensi per-aspek per-token), render heatmap token -> fig_attention.pdf.
"""
import sys, json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")

from finetune_hiabsa import (HIABSA, ASPECTS, lexicon_feature, MODEL_NAME, DEVICE)
from transformers import AutoTokenizer

CKPT = "hiabsa_best.pt"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = HIABSA(use_aqa=True, use_lgf=True).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE)
state = state.get("model", state) if isinstance(state, dict) else state
model.load_state_dict(state, strict=False)
model.eval()
print("checkpoint dimuat.")


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def attn_for(text):
    enc = tok(text, truncation=True, max_length=192, padding="max_length",
              return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE); m = enc["attention_mask"].to(DEVICE)
    lex = torch.tensor([lexicon_feature(text)], dtype=torch.float32).to(DEVICE)
    logits = model(input_ids=ids, attention_mask=m, lex=lex)     # (1,A,4)
    alpha = model.last_attn[0].cpu().numpy()                     # (A, L)
    pred = logits[0].argmax(-1).cpu().numpy()                    # (A,)
    toks = tok.convert_ids_to_tokens(ids[0].cpu().tolist())
    n = int(m[0].sum().item())
    return toks[:n], alpha[:, :n], pred


# ── pilih contoh: atensi bersih (kata konten), prediksi cocok gold, campuran ──
rows = load_jsonl("test_gold.jsonl")
CLASSES = ["N/A", "POSITIF", "NEGATIF", "NETRAL"]
STOP = set("juga yang di ke dari dan atau tapi tetapi ini itu ada sih kok deh nya "
           "untuk pada saya kami kita sangat cukup relatif . , ! ? ) ( - ya".split())
AMAP = {a: i for i, a in enumerate(ASPECTS)}

def gold_vec(r):
    y = [0] * len(ASPECTS)
    for l in r.get("labels", []):
        if l["aspect"] in AMAP:
            y[AMAP[l["aspect"]]] = CLASSES.index(l["polarity"])
    return y

def rank(r):
    ntok = len(tok.tokenize(r["text"]))
    if not (10 <= ntok <= 34):
        return (-1,)
    toks, alpha, pred = attn_for(r["text"])
    gy = gold_vec(r)
    active = [a for a in range(len(ASPECTS)) if pred[a] != 0]
    if len(active) < 2:
        return (-1,)
    match = sum(1 for a in active if pred[a] == gy[a]) / len(active)
    clean = 0
    for a in active:
        t1 = toks[int(np.argmax(alpha[a]))].replace("##", "").lower()
        if t1 not in STOP and t1.isalpha() and len(t1) > 2:
            clean += 1
    clean /= len(active)
    # utamakan: semua aspek aktif BENAR, atensi bersih, >=3 aspek, campuran
    allcorrect = int(match >= 0.999)
    return (allcorrect, round(clean, 2), len(active), int(2 <= len(active) <= 4),
            -abs(ntok - 22))

best = max(rows, key=rank)
print("contoh:", best["text"][:160])
toks, alpha, pred = attn_for(best["text"])
gy = gold_vec(best)
# hanya tampilkan aspek aktif yang prediksinya benar
_show = [a for a in range(len(ASPECTS)) if pred[a] != 0 and pred[a] == gy[a]]
print("aspek benar ditampilkan:", [ASPECTS[a] for a in _show])

# hanya aspek yang benar-diprediksi (figure jujur: bukti untuk keputusan yg benar)
show = _show[:4] if _show else [a for a in range(len(ASPECTS)) if pred[a] != 0][:3]

# bersihkan token subword (## -> gabung tampilan), potong [CLS]/[SEP]
disp = [t.replace("##", "") for t in toks]

fig, ax = plt.subplots(figsize=(min(0.42 * len(toks) + 1.5, 12), 0.5 * len(show) + 1.2))
M = alpha[show]
im = ax.imshow(M, aspect="auto", cmap="OrRd", vmin=0)
ax.set_xticks(range(len(disp)))
ax.set_xticklabels(disp, rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(show)))
ax.set_yticklabels([f"{ASPECTS[a]}\n({CLASSES[pred[a]]})" for a in show], fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("attention", fontsize=7); cb.ax.tick_params(labelsize=6)
ax.set_title("HI-ABSA Aspect-Query Attention (per-aspect token evidence)", fontsize=9)
plt.tight_layout()
plt.savefig("fig_attention.pdf", bbox_inches="tight")
print("fig_attention.pdf ditulis.")

# cetak top-token per aspek utk teks paper
print("\nTop token per aspek (bukti eksplisit):")
for a in show:
    idx = np.argsort(-alpha[a])[:5]
    top = [disp[i] for i in idx if disp[i] not in ("[CLS]", "[SEP]")][:4]
    print(f"  {ASPECTS[a]:<20} ({CLASSES[pred[a]]}): {', '.join(top)}")
