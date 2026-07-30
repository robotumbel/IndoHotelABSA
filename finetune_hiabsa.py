"""
finetune_hiabsa.py — HI-ABSA (Hospitality-Informed ABSA).

Arsitektur (sesuai Persamaan 2-7 di paper):
  Encoder (IndoBERT) -> Aspect-Query Attention (AQA) -> Lexicon-Gated Fusion (LGF)
  -> 7 head klasifikasi 4-arah {N/A, POSITIF, NEGATIF, NETRAL}.

Ablasi:
  --no_aqa : ganti AQA dgn representasi [CLS] bersama (baseline shared-CLS)
  --no_lgf : matikan Lexicon-Gated Fusion

Format data = gold JSONL (train/val/test.jsonl). Prediksi ditulis dalam format
yang sama agar bisa dievaluasi oleh evaluate_absa.py.

Butuh: pip install torch transformers
Pemakaian:
  python finetune_hiabsa.py --train train.jsonl --val val.jsonl --test test.jsonl
"""
import argparse
import json
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from lexicon_baseline import ASPECT_KEYWORDS, POS, NEG   # reuse leksikon

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
CLASSES = ["N/A", "POSITIF", "NEGATIF", "NETRAL"]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}
ID2CLS = {i: c for c, i in CLS2ID.items()}
A = len(ASPECTS)

MODEL_NAME = "indobenchmark/indobert-base-p1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
def load_gold(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_targets(labels):
    y = [0] * A
    amap = {a: i for i, a in enumerate(ASPECTS)}
    for lab in labels:
        a, p = lab.get("aspect"), lab.get("polarity")
        if a in amap and p in CLS2ID:
            y[amap[a]] = CLS2ID[p]
    return y


def lexicon_feature(text):
    """l_a = [pos_count, neg_count] per aspek (Persamaan LGF)."""
    t = text.lower()
    feats = []
    for asp in ASPECTS:
        kws = ASPECT_KEYWORDS.get(asp, [])
        # aktif hanya bila aspek disebut (ada cue word); jika tidak, sinyal nol
        mentioned = any(k in t for k in kws)
        if mentioned:
            pos = float(sum(t.count(w) for w in POS))
            neg = float(sum(t.count(w) for w in NEG))
        else:
            pos = neg = 0.0
        feats.append([pos, neg])
    return feats                                    # (A, 2)


class ABSADataset(Dataset):
    def __init__(self, rows, tok, max_len=192, has_labels=True):
        self.rows, self.tok, self.max_len, self.has_labels = rows, tok, max_len, has_labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["lex"] = torch.tensor(lexicon_feature(r["text"]), dtype=torch.float32)  # (A,2)
        item["review_id"] = r.get("review_id", idx)
        if self.has_labels:
            item["labels"] = torch.tensor(encode_targets(r.get("labels", [])), dtype=torch.long)
        return item


# ─────────────────────────────────────────────────────────────
class HIABSA(nn.Module):
    def __init__(self, model_name=MODEL_NAME, use_aqa=True, use_lgf=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        d = self.encoder.config.hidden_size
        self.d = d
        self.use_aqa, self.use_lgf = use_aqa, use_lgf

        # Aspect-Query Attention (Persamaan 3-4)
        self.aspect_q = nn.Parameter(torch.randn(A, d) * 0.02)   # 7 query aspek
        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)

        # Lexicon-Gated Fusion (Persamaan 5-6)
        self.Wl = nn.Linear(2, d)                    # proyeksi fitur leksikon
        self.wg = nn.Linear(d + 2, 1)                # gerbang
        self.drop = nn.Dropout(0.1)

        # 7 head klasifikasi 4-arah (Persamaan 7)
        self.heads = nn.ModuleList([nn.Linear(d, len(CLASSES)) for _ in range(A)])
        self.last_attn = None                        # simpan alpha utk interpretabilitas

    def forward(self, input_ids, attention_mask, lex, **kw):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state                     # (B, L, d)
        B = H.size(0)

        if self.use_aqa:
            q = self.WQ(self.aspect_q)                # (A, d)
            K = self.WK(H)                            # (B, L, d)
            V = self.WV(H)                            # (B, L, d)
            scores = torch.einsum("ad,bld->bal", q, K) / math.sqrt(self.d)   # (B, A, L)
            mask = (attention_mask == 0).unsqueeze(1)                         # (B,1,L)
            scores = scores.masked_fill(mask, float("-inf"))
            alpha = torch.softmax(scores, dim=-1)     # (B, A, L)
            self.last_attn = alpha.detach()
            c = torch.einsum("bal,bld->bad", alpha, V)                        # (B, A, d)
        else:
            # ablasi: representasi [CLS] bersama untuk semua aspek
            cls = H[:, 0]                             # (B, d)
            c = cls.unsqueeze(1).expand(B, A, self.d)

        c = self.drop(c)

        if self.use_lgf:
            g = torch.sigmoid(self.wg(torch.cat([c, lex], dim=-1)))           # (B, A, 1)
            c = c + g * self.Wl(lex)                  # (B, A, d)

        # klasifikasi per-aspek
        logits = torch.stack([self.heads[a](c[:, a]) for a in range(A)], dim=1)  # (B, A, 4)
        return logits


# ─────────────────────────────────────────────────────────────
def train(model, tr, va, epochs=4, lr=2e-5, warmup=0.1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = len(tr) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * warmup), total)
    weight = torch.tensor([0.3, 1.0, 1.0, 1.0], device=DEVICE)   # bobot kelas, N/A kecil
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0
        for b in tr:
            ids, m = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
            lex, y = b["lex"].to(DEVICE), b["labels"].to(DEVICE)
            logits = model(ids, m, lex)
            loss = loss_fn(logits.reshape(-1, len(CLASSES)), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += loss.item()
        acc = evaluate_acc(model, va)
        print(f"Epoch {ep}: loss={run/len(tr):.4f} | val_aspect_acc={acc:.4f}")
        if acc > best:
            best = acc; torch.save(model.state_dict(), "hiabsa_best.pt")
    print(f"Best val_acc={best:.4f} -> hiabsa_best.pt")


@torch.no_grad()
def evaluate_acc(model, loader):
    model.eval(); correct = total = 0
    for b in loader:
        ids, m = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
        lex, y = b["lex"].to(DEVICE), b["labels"].to(DEVICE)
        pred = model(ids, m, lex).argmax(-1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / total if total else 0.0


@torch.no_grad()
def predict_to_jsonl(model, loader, out_path):
    model.eval(); results = []
    for b in loader:
        ids, m = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
        lex = b["lex"].to(DEVICE)
        rids = b["review_id"]
        preds = model(ids, m, lex).argmax(-1).cpu().tolist()      # (B, A)
        rid_list = rids.tolist() if torch.is_tensor(rids) else rids
        for rid, prow in zip(rid_list, preds):
            labels = [{"aspect": ASPECTS[a], "polarity": ID2CLS[c]}
                      for a, c in enumerate(prow) if c != 0]
            results.append({"review_id": rid, "labels": labels})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Prediksi -> {out_path}")
    print(f"Evaluasi: python evaluate_absa.py --gold test.jsonl --pred {out_path}")


# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--no_aqa", action="store_true", help="ablasi: matikan Aspect-Query Attention")
    ap.add_argument("--no_lgf", action="store_true", help="ablasi: matikan Lexicon-Gated Fusion")
    ap.add_argument("--pred_out", default="pred_hiabsa.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tr = DataLoader(ABSADataset(load_gold(args.train), tok), batch_size=args.batch, shuffle=True)
    va = DataLoader(ABSADataset(load_gold(args.val), tok), batch_size=args.batch)
    te = DataLoader(ABSADataset(load_gold(args.test), tok), batch_size=args.batch)

    model = HIABSA(use_aqa=not args.no_aqa, use_lgf=not args.no_lgf).to(DEVICE)
    variant = "HI-ABSA" + ("" if not args.no_aqa else " -AQA") + ("" if not args.no_lgf else " -LGF")
    print(f"Device: {DEVICE} | varian: {variant} | train: {len(tr.dataset)}")
    train(model, tr, va, epochs=args.epochs, lr=args.lr)

    model.load_state_dict(torch.load("hiabsa_best.pt", map_location=DEVICE))
    predict_to_jsonl(model, te, args.pred_out)


if __name__ == "__main__":
    main()
