"""
finetune_llm_lora.py — Baseline: LLM fine-tuned dengan LoRA/QLoRA untuk ABSA.

Pendekatan: supervised fine-tuning (SFT) generatif. Input = instruksi + review,
target = JSON daftar {aspect, polarity}. Loss hanya dihitung pada token target
(prompt di-mask). Inferensi = generate -> parse JSON -> tulis JSONL yang
kompatibel dengan evaluate_absa.py.

Butuh: pip install torch transformers peft
       (opsional GPU 4-bit: pip install bitsandbytes  + flag --load_4bit)

Pemakaian:
  python finetune_llm_lora.py --train train.jsonl --val val.jsonl --test test.jsonl \
         --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
POLARITIES = ["POSITIF", "NEGATIF", "NETRAL"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INSTRUCTION = (
    "Tugas: Aspect-Based Sentiment Analysis ulasan hotel Indonesia.\n"
    f"Aspek valid: {ASPECTS}. Sentimen: {POLARITIES}.\n"
    "Keluarkan HANYA JSON daftar objek {\"aspect\":..,\"polarity\":..} "
    "untuk aspek yang disebut.\n\n"
    "Ulasan: \"{text}\"\nJawaban: "
)


def load_gold(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def target_json(labels):
    arr = [{"aspect": l["aspect"], "polarity": l["polarity"]}
           for l in labels if l.get("aspect") in ASPECTS and l.get("polarity") in POLARITIES]
    return json.dumps(arr, ensure_ascii=False)


def parse_output(text):
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out, seen = [], set()
    for o in arr:
        if not isinstance(o, dict):
            continue
        a, p = o.get("aspect"), o.get("polarity")
        if a in ASPECTS and p in POLARITIES and a not in seen:
            out.append({"aspect": a, "polarity": p}); seen.add(a)
    return out


class SFTDataset(Dataset):
    """Bangun (prompt, target) -> input_ids + labels (prompt di-mask -100)."""
    def __init__(self, rows, tok, max_len=384):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = INSTRUCTION.replace("{text}", r["text"])
        target = target_json(r.get("labels", [])) + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + t_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels)}


def train(model, tr, epochs, lr):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total = len(tr) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)
    model.train()
    for ep in range(1, epochs + 1):
        run = 0.0
        for b in tr:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            out = model(**b)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += out.loss.item()
        print(f"Epoch {ep}: loss={run/len(tr):.4f}")


@torch.no_grad()
def predict(model, tok, rows, out_path, max_new=96, gen_batch=16):
    model.eval(); results = []
    tok.padding_side = "left"                     # wajib utk generate ter-batch (decoder-only)
    for s in range(0, len(rows), gen_batch):
        chunk = rows[s:s + gen_batch]
        prompts = [INSTRUCTION.replace("{text}", r["text"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=384).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        for r, out_ids in zip(chunk, gen):
            text = tok.decode(out_ids[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"review_id": r.get("review_id", len(results) + 1),
                            "labels": parse_output(text)})
        print(f"  {min(s + gen_batch, len(rows))}/{len(rows)}")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Prediksi -> {out_path}")
    print(f"Evaluasi: python evaluate_absa.py --gold test_gold.jsonl --pred {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="LLM causal (ganti ke LLM Indonesia/Llama untuk hasil terbaik)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA 4-bit (butuh GPU + bitsandbytes)")
    ap.add_argument("--pred_out", default="pred_llm_lora.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = {}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        kw["device_map"] = "auto"
    elif DEVICE.type == "cuda":
        kw["torch_dtype"] = torch.float16           # fp16 di GPU -> lebih cepat & hemat VRAM
    base = AutoModelForCausalLM.from_pretrained(args.model, **kw)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(base, lora)
    if not args.load_4bit:
        model.to(DEVICE)
    model.print_trainable_parameters()

    tr_rows = load_gold(args.train)
    tr = DataLoader(SFTDataset(tr_rows, tok), batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    print(f"Device: {DEVICE} | model: {args.model} | train: {len(tr_rows)}")
    train(model, tr, args.epochs, args.lr)

    predict(model, tok, load_gold(args.test), args.pred_out)


if __name__ == "__main__":
    main()
