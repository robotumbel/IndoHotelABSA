# -*- coding: utf-8 -*-
"""Builds kaggle/large_encoders_absa.ipynb (E11, R2.2) and the dataset zip to upload.
Run: python kaggle/build_notebook.py   (from revision_R1/)
"""
import json, os, shutil, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE

MD0 = """# IndoHotelABSA — large encoders (revision R2.2)

Fine-tunes **IndoBERT-large-p1** and **XLM-RoBERTa-large** with the same formulation as the base encoders in the paper
(shared encoder, [CLS], 7 four-way heads, class weights N/A 0.3 / others 1.0, 4 epochs, AdamW wd 1e-4, 10% linear
warm-up, gradient clipping 1.0, max length 192, best epoch by validation cell accuracy), 5 seeds each.

Differences from the base protocol, forced by GPU memory and stability of large models:
mixed precision (fp16 autocast), micro-batch 8 × 2 gradient-accumulation steps (effective batch 16), learning rate 1e-5.

**Before running:** Settings → Accelerator = GPU T4 x2 (or P100), Internet = On.
Add the dataset `indohotelabsa-splits` (private) with `train_silver.jsonl`, `val_gold.jsonl`, `test_gold.jsonl`.
Then **Save Version → Save & Run All (Commit)** and download `/kaggle/working/preds_large.zip` from the Output tab."""

CODE_SETUP = r'''import glob, json, os, random, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found under /kaggle/input - did you add the dataset?"
    return hits[0]

TRAIN, VAL, TEST = find("train_silver.jsonl"), find("val_gold.jsonl"), find("test_gold.jsonl")
OUT = "/kaggle/working/preds_large"; os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda")
print(torch.cuda.get_device_name(0), TRAIN)

MODELS = {"indobert_large": "indobenchmark/indobert-large-p1", "xlmr_large": "xlm-roberta-large"}
SEEDS = [42, 123, 7, 2024, 99]
EPOCHS, LR, MICRO, ACCUM, MAXLEN = 4, 1e-5, 8, 2, 192

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
CLASSES = ["N/A", "POSITIF", "NEGATIF", "NETRAL"]
C2I = {c: i for i, c in enumerate(CLASSES)}

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]

def targets(labels):
    y = [0] * 7; am = {a: i for i, a in enumerate(ASPECTS)}
    for l in labels or []:
        if l["aspect"] in am and l["polarity"] in C2I:
            y[am[l["aspect"]]] = C2I[l["polarity"]]
    return y

class DS(Dataset):
    def __init__(self, rows, tok): self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        e = self.tok(r["text"], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
        d = {k: v.squeeze(0) for k, v in e.items() if k in ("input_ids", "attention_mask")}
        d["labels"] = torch.tensor(targets(r.get("labels")), dtype=torch.long); d["review_id"] = r["review_id"]
        return d

class Model(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name); h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1); self.heads = nn.ModuleList([nn.Linear(h, 4) for _ in range(7)])
    def forward(self, ids, mask):
        cls = self.drop(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        return torch.stack([h(cls) for h in self.heads], 1)
'''

CODE_RUN = r'''@torch.no_grad()
def cell_acc(m, dl):
    m.eval(); c = t = 0
    for b in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            p = m(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)).argmax(-1)
        y = b["labels"].to(DEVICE); c += (p == y).sum().item(); t += y.numel()
    return c / t

@torch.no_grad()
def predict(m, dl, path):
    m.eval(); rows = []
    for b in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            P = m(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)).argmax(-1).cpu().tolist()
        for rid, pr in zip(b["review_id"].tolist(), P):
            rows.append({"review_id": rid, "labels": [{"aspect": ASPECTS[a], "polarity": CLASSES[c]}
                                                      for a, c in enumerate(pr) if c]})
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

log = open(f"{OUT}/train_log.txt", "a")
for mid, name in MODELS.items():
    tok = AutoTokenizer.from_pretrained(name)
    tr_rows, va_rows, te_rows = load(TRAIN), load(VAL), load(TEST)
    for seed in SEEDS:
        out = f"{OUT}/{mid}_{seed}.jsonl"
        if os.path.exists(out): print("skip", out); continue
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        tr = DataLoader(DS(tr_rows, tok), batch_size=MICRO, shuffle=True)
        va = DataLoader(DS(va_rows, tok), batch_size=32); te = DataLoader(DS(te_rows, tok), batch_size=32)
        m = Model(name).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
        steps = (len(tr) // ACCUM) * EPOCHS
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        lossf = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 1, 1, 1.], device=DEVICE))
        scaler = torch.amp.GradScaler("cuda"); best = -1; t0 = time.time()
        for ep in range(1, EPOCHS + 1):
            m.train(); run = 0
            for i, b in enumerate(tr):
                with torch.autocast("cuda", dtype=torch.float16):
                    lo = m(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
                    loss = lossf(lo.float().reshape(-1, 4), b["labels"].to(DEVICE).reshape(-1)) / ACCUM
                scaler.scale(loss).backward(); run += loss.item() * ACCUM
                if (i + 1) % ACCUM == 0:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
            acc = cell_acc(m, va)
            msg = f"{mid} seed={seed} epoch={ep} loss={run/len(tr):.4f} val_cell_acc={acc:.4f}"
            print(msg); log.write(msg + "\n"); log.flush()
            if acc > best:
                best = acc; torch.save(m.state_dict(), "/kaggle/working/best.pt")
        m.load_state_dict(torch.load("/kaggle/working/best.pt")); predict(m, te, out)
        msg = f"{mid} seed={seed} done best_val={best:.4f} {(time.time()-t0)/60:.1f} min"
        print(msg); log.write(msg + "\n"); log.flush()
        del m, opt; torch.cuda.empty_cache()
if os.path.exists("/kaggle/working/best.pt"): os.remove("/kaggle/working/best.pt")
'''

CODE_ZIP = r'''import shutil, transformers
json.dump({"transformers": transformers.__version__, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "lr": LR, "micro_batch": MICRO, "accum": ACCUM,
           "epochs": EPOCHS, "fp16_autocast": True}, open(f"{OUT}/env.json", "w"), indent=1)
shutil.make_archive("/kaggle/working/preds_large", "zip", OUT)
print(sorted(os.listdir(OUT)))
'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"cells": [cell("markdown", MD0), cell("code", CODE_SETUP), cell("code", CODE_RUN), cell("code", CODE_ZIP)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "large_encoders_absa.ipynb"), "w", encoding="utf-8"), indent=1)

with zipfile.ZipFile(os.path.join(HERE, "indohotelabsa-splits.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["train_silver.jsonl", "val_gold.jsonl", "test_gold.jsonl"]:
        z.write(os.path.join(ROOT, f), f)
print("built", os.listdir(HERE))
