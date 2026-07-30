"""
finetune_indobert_absa.py — Fine-tuning IndoBERT untuk Aspect-Based
Sentiment Analysis (IndoHotelABSA).

Formulasi: untuk setiap review, prediksi 7 aspek, masing-masing ke salah satu
dari 4 kelas: {N/A (tidak disebut), POSITIF, NEGATIF, NETRAL}.
Satu encoder IndoBERT + 7 classification head (masing-masing 4-way).
Ini menangani ACD (aspek disebut bila != N/A) dan ACP (polaritas) sekaligus.

Format data (gold JSONL, hasil anotasi):
  {"review_id":1,"text":"...","labels":[{"aspect":"Kebersihan","polarity":"POSITIF"}]}

Prediksi ditulis dalam format yang sama (dengan 'labels') agar bisa langsung
dievaluasi oleh evaluate_absa.py.

Butuh: pip install torch transformers scikit-learn
Pemakaian:
  python finetune_indobert_absa.py --train train.jsonl --val val.jsonl --test test.jsonl
"""
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ---- Skema label ----
ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
CLASSES = ["N/A", "POSITIF", "NEGATIF", "NETRAL"]      # indeks 0..3
CLS2ID = {c: i for i, c in enumerate(CLASSES)}
ID2CLS = {i: c for c, i in CLS2ID.items()}

MODEL_NAME = "indobenchmark/indobert-base-p1"          # IndoBERT (Wilie et al. 2020)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------
def load_gold(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_targets(labels):
    """labels -> vektor 7 int (kelas per aspek). Aspek tak-disebut = N/A(0)."""
    y = [0] * len(ASPECTS)
    amap = {a: i for i, a in enumerate(ASPECTS)}
    for lab in labels:
        a, p = lab["aspect"], lab["polarity"]
        if a in amap and p in CLS2ID:
            y[amap[a]] = CLS2ID[p]
    return y


class ABSADataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=192, has_labels=True):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.has_labels:
            item["labels"] = torch.tensor(encode_targets(r.get("labels", [])),
                                          dtype=torch.long)
        item["review_id"] = r.get("review_id", idx)
        return item


# ---------------------------------------------------------------
class IndoBERTABSA(nn.Module):
    """Encoder IndoBERT + 7 head klasifikasi (masing-masing 4 kelas)."""
    def __init__(self, model_name=MODEL_NAME, n_aspects=len(ASPECTS), n_classes=len(CLASSES)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(hidden, n_classes) for _ in range(n_aspects)])

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])       # token [CLS]
        # (B, n_aspects, n_classes)
        logits = torch.stack([h(cls) for h in self.heads], dim=1)
        return logits


# ---------------------------------------------------------------
def train(model, train_loader, val_loader, epochs=4, lr=2e-5, warmup=0.1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * warmup), total)
    # class weight opsional: N/A biasanya dominan -> beri bobot lebih kecil
    weight = torch.tensor([0.3, 1.0, 1.0, 1.0], device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    best_val = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)                    # (B, 7)
            logits = model(ids, mask)                         # (B, 7, 4)
            loss = loss_fn(logits.reshape(-1, len(CLASSES)), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item()
        acc = evaluate_acc(model, val_loader)
        print(f"Epoch {ep}: train_loss={run/len(train_loader):.4f} | val_aspect_acc={acc:.4f}")
        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), "indobert_absa_best.pt")
    print(f"Model terbaik disimpan (val_acc={best_val:.4f}) -> indobert_absa_best.pt")


@torch.no_grad()
def evaluate_acc(model, loader):
    """Akurasi kasar tingkat-aspek (untuk pemilihan checkpoint)."""
    model.eval()
    correct = total = 0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        pred = model(ids, mask).argmax(-1)                    # (B, 7)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


@torch.no_grad()
def predict_to_jsonl(model, loader, out_path):
    """Tulis prediksi dalam format gold (agar bisa dievaluasi evaluate_absa.py)."""
    model.eval()
    results = []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        rids = batch["review_id"]
        preds = model(ids, mask).argmax(-1).cpu().tolist()    # (B, 7)
        for rid, prow in zip(rids.tolist() if torch.is_tensor(rids) else rids, preds):
            labels = []
            for ai, cls_id in enumerate(prow):
                if cls_id != 0:                               # bukan N/A
                    labels.append({"aspect": ASPECTS[ai], "polarity": ID2CLS[cls_id]})
            results.append({"review_id": rid, "labels": labels})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Prediksi ditulis -> {out_path}")
    print("Evaluasi: python evaluate_absa.py --gold test.jsonl --pred " + out_path)


# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--pred_out", default="pred_indobert.jsonl")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="encoder HF: indobenchmark/indobert-base-p1 (IndoBERT), "
                         "indolem/indobertweet-base-uncased (IndoBERTweet), "
                         "bert-base-multilingual-cased (mBERT), "
                         "xlm-roberta-base (XLM-R)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    tr = DataLoader(ABSADataset(load_gold(args.train), tok), batch_size=args.batch, shuffle=True)
    va = DataLoader(ABSADataset(load_gold(args.val), tok), batch_size=args.batch)
    te = DataLoader(ABSADataset(load_gold(args.test), tok), batch_size=args.batch)

    model = IndoBERTABSA(model_name=args.model).to(DEVICE)
    print(f"Device: {DEVICE} | encoder: {args.model} | contoh train: {len(tr.dataset)}")
    train(model, tr, va, epochs=args.epochs, lr=args.lr)

    model.load_state_dict(torch.load("indobert_absa_best.pt", map_location=DEVICE))
    predict_to_jsonl(model, te, args.pred_out)


if __name__ == "__main__":
    main()
