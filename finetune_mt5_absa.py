# -*- coding: utf-8 -*-
"""#7 Baseline generatif seq2seq (mT5) untuk ABSA hotel Indonesia.
Input = instruksi + review; target = JSON [{aspect,polarity}]. Latih pada silver,
generate pada gold test -> parse -> pred JSONL kompatibel evaluate_absa.py.
Pemakaian: python finetune_mt5_absa.py --model google/mt5-small
"""
import argparse, json, re, random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          get_linear_schedule_with_warmup)

ASPECTS = ["Lokasi","Kebersihan","Pelayanan","Kamar & Fasilitas","Harga",
           "Makanan & Minuman","Fasilitas Pendukung"]
POLARITIES = ["POSITIF","NEGATIF","NETRAL"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INSTR = ("Ekstrak aspek dan sentimen ulasan hotel. "
         f"Aspek: {ASPECTS}. Sentimen: {POLARITIES}. "
         "Keluarkan JSON [{{\"aspect\":..,\"polarity\":..}}]. Ulasan: {text}")


def load_gold(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]

def target_json(labels):
    arr = [{"aspect": l["aspect"], "polarity": l["polarity"]} for l in labels
           if l.get("aspect") in ASPECTS and l.get("polarity") in POLARITIES]
    return json.dumps(arr, ensure_ascii=False)

def parse_output(text):
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m: return []
    try: arr = json.loads(m.group(0))
    except Exception: return []
    out, seen = [], set()
    for o in arr:
        if isinstance(o, dict):
            a, p = o.get("aspect"), o.get("polarity")
            if a in ASPECTS and p in POLARITIES and a not in seen:
                out.append({"aspect": a, "polarity": p}); seen.add(a)
    return out


class DS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tok(INSTR.replace("{text}", r["text"]), truncation=True,
                     max_length=256, padding="max_length", return_tensors="pt")
        y = self.tok(target_json(r.get("labels", [])), truncation=True,
                     max_length=96, padding="max_length", return_tensors="pt")
        lab = y["input_ids"].squeeze(0)
        lab[lab == self.tok.pad_token_id] = -100
        return {"input_ids": x["input_ids"].squeeze(0),
                "attention_mask": x["attention_mask"].squeeze(0), "labels": lab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_silver.jsonl")
    ap.add_argument("--val", default="val_gold.jsonl")
    ap.add_argument("--test", default="test_gold.jsonl")
    ap.add_argument("--model", default="google/mt5-small")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred_out", default="pred_mt5.jsonl")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(DEVICE)
    tr = DataLoader(DS(load_gold(args.train), tok), batch_size=args.batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(tr)*args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(total*0.1), total)
    print(f"Device {DEVICE} | {args.model} | train {len(tr.dataset)}", flush=True)
    model.train()
    for ep in range(1, args.epochs+1):
        run = 0.0
        for b in tr:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            out = model(**b); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(); run += out.loss.item()
        print(f"epoch {ep}: loss={run/len(tr):.4f}", flush=True)

    model.eval(); rows = load_gold(args.test); res = []
    with torch.no_grad():
        for s in range(0, len(rows), 16):
            chunk = rows[s:s+16]
            enc = tok([INSTR.replace("{text}", r["text"]) for r in chunk],
                      return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
            gen = model.generate(**enc, max_new_tokens=96, num_beams=1)
            for r, g in zip(chunk, gen):
                txt = tok.decode(g, skip_special_tokens=True)
                res.append({"review_id": r.get("review_id"), "labels": parse_output(txt)})
            print(f"  {min(s+16,len(rows))}/{len(rows)}", flush=True)
    with open(args.pred_out, "w", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"pred -> {args.pred_out}", flush=True)
    print(f"eval: python evaluate_absa.py --gold test_gold.jsonl --pred {args.pred_out}", flush=True)


if __name__ == "__main__":
    main()
