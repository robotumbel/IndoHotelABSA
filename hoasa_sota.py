# -*- coding: utf-8 -*-
"""
hoasa_sota.py — Validasi SOTA pada benchmark mapan HoASA (IndoNLU).
Latih IndoBERT (encoder+10 head) dan HI-ABSA (AQA+10 head) pada HoASA,
evaluasi dengan metrik IndoNLU (macro-F1 per-aspek, dirata-rata) untuk
head-to-head dengan angka publikasi. Pemakaian:
  python hoasa_sota.py --model indobert
  python hoasa_sota.py --model hiabsa
"""
import argparse, csv, math, sys
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
sys.stdout.reconfigure(encoding="utf-8")

ASPECTS = ["ac","air_panas","bau","general","kebersihan","linen","service",
           "sunrise_meal","tv","wifi"]
CLASSES = ["neut","pos","neg","neg_pos"]          # neut = mayoritas/background
C2I = {c:i for i,c in enumerate(CLASSES)}
A = len(ASPECTS)
MODEL_NAME = "indobenchmark/indobert-base-p1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    hdr = rows[0]
    out = []
    for r in rows[1:]:
        text = r[0]
        labs = [C2I.get(r[1+i], 0) for i in range(A)]
        out.append((text, labs))
    return out


class HoasaDS(Dataset):
    def __init__(self, rows, tok, max_len=192):
        self.rows, self.tok, self.max_len = rows, tok, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        text, labs = self.rows[i]
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(labs, dtype=torch.long)
        return item


class MultiAspect(nn.Module):
    def __init__(self, use_aqa=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        d = self.encoder.config.hidden_size; self.d = d
        self.use_aqa = use_aqa
        if use_aqa:
            self.aspect_q = nn.Parameter(torch.randn(A, d) * 0.02)
            self.WQ = nn.Linear(d, d, bias=False)
            self.WK = nn.Linear(d, d, bias=False)
            self.WV = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(d, len(CLASSES)) for _ in range(A)])
    def forward(self, input_ids, attention_mask, **kw):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B = H.size(0)
        if self.use_aqa:
            q = self.WQ(self.aspect_q); K = self.WK(H); V = self.WV(H)
            scores = torch.einsum("ad,bld->bal", q, K) / math.sqrt(self.d)
            scores = scores.masked_fill((attention_mask==0).unsqueeze(1), float("-inf"))
            alpha = torch.softmax(scores, -1)
            c = torch.einsum("bal,bld->bad", alpha, V)
        else:
            c = H[:,0].unsqueeze(1).expand(B, A, self.d)
        c = self.drop(c)
        return torch.stack([self.heads[a](c[:,a]) for a in range(A)], dim=1)  # (B,A,4)


def run_eval(model, dl):
    model.eval(); P=[[] for _ in range(A)]; T=[[] for _ in range(A)]
    with torch.no_grad():
        for b in dl:
            ids=b["input_ids"].to(DEVICE); m=b["attention_mask"].to(DEVICE)
            logits=model(input_ids=ids, attention_mask=m)
            pred=logits.argmax(-1).cpu()
            for a in range(A):
                P[a]+=pred[:,a].tolist(); T[a]+=b["labels"][:,a].tolist()
    # metrik IndoNLU: macro-F1 per aspek, dirata-rata
    f1s=[f1_score(T[a],P[a],average="macro",zero_division=0) for a in range(A)]
    accs=[accuracy_score(T[a],P[a]) for a in range(A)]
    # micro macro-F1 (flatten semua sel)
    allT=[x for a in range(A) for x in T[a]]; allP=[x for a in range(A) for x in P[a]]
    return sum(f1s)/A, sum(accs)/A, f1_score(allT,allP,average="macro",zero_division=0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", choices=["indobert","hiabsa"], default="indobert")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    tr=load_csv("hoasa/train_preprocess.csv")
    va=load_csv("hoasa/valid_preprocess.csv")
    te=load_csv("hoasa/test_preprocess.csv")
    trdl=DataLoader(HoasaDS(tr,tok),batch_size=args.batch,shuffle=True)
    vadl=DataLoader(HoasaDS(va,tok),batch_size=32)
    tedl=DataLoader(HoasaDS(te,tok),batch_size=32)

    model=MultiAspect(use_aqa=(args.model=="hiabsa")).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    total=len(trdl)*args.epochs
    sched=get_linear_schedule_with_warmup(opt,int(total*0.1),total)
    lossf=nn.CrossEntropyLoss()
    print(f"[{args.model}] device={DEVICE} train={len(tr)} test={len(te)}")
    best=0; best_test=None
    for ep in range(1,args.epochs+1):
        model.train()
        for b in trdl:
            ids=b["input_ids"].to(DEVICE); m=b["attention_mask"].to(DEVICE)
            y=b["labels"].to(DEVICE)
            logits=model(input_ids=ids,attention_mask=m)
            loss=sum(lossf(logits[:,a],y[:,a]) for a in range(A))/A
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step(); opt.zero_grad()
        vf1,_,_=run_eval(model,vadl)
        if vf1>=best:
            best=vf1; best_test=run_eval(model,tedl)
        print(f"  ep{ep}: val macroF1={vf1:.4f}")
    f1,acc,microf1=best_test
    print(f"\n=== HoASA TEST [{args.model}] ===")
    print(f"macro-F1 (per-aspek, dirata-rata, metrik IndoNLU): {f1:.4f}")
    print(f"akurasi rata-rata per-aspek: {acc:.4f}")
    print(f"macro-F1 (flatten semua sel): {microf1:.4f}")


if __name__=="__main__":
    main()
