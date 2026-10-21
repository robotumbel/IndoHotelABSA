# -*- coding: utf-8 -*-
"""E9/E10 (R2.1, R2.2): zero-shot ABSA with an open LLM that never produced any draft.

The prompt, label schema and post-processing are imported unchanged from
preannotate_gemini.py, so the only difference from the Gemini silver drafts is the
model. The prompt is sent as the single user turn of the model's chat template
(no system prompt is added by us; Qwen2.5's template inserts its default one).
Decoding is greedy (do_sample=False), max_new_tokens=256.

Every raw completion is stored next to the parsed labels for audit. Resumable.

Usage (from paper2_sentiment/):
  python submission_NLP_Journal/revision_R1/scripts/llm_zeroshot_local.py \
      --model Qwen/Qwen2.5-7B-Instruct --load_4bit \
      --out submission_NLP_Journal/revision_R1/preds_llm/qwen25_7b_zeroshot.jsonl
Then score with:  --pred <out> in the analysis scripts (labels field is standard).
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
sys.path.insert(0, ROOT)
from preannotate_gemini import PROMPT, parse_labels  # identical prompt + parser to the silver drafts

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_lenient(raw):
    """Sensitivity check only: also accept bare objects that are not wrapped in a list.
    The primary 'labels' field uses the unchanged Gemini parser."""
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", raw or ""):
        try:
            objs.append(json.loads(m.group(0)))
        except Exception:
            pass
    return parse_labels(json.dumps(objs, ensure_ascii=False))


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--infile", default=os.path.join(ROOT, "test_gold.jsonl"))
ap.add_argument("--out", required=True)
ap.add_argument("--load_4bit", action="store_true")
ap.add_argument("--max_new_tokens", type=int, default=256)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--embed_on_cpu", action="store_true",
                help="keep the (unquantised) input embedding on CPU so a 7B 4-bit model fits a 6 GB GPU")
args = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

done = set()
if os.path.exists(args.out):
    with open(args.out, encoding="utf-8") as f:
        done = {json.loads(l)["review_id"] for l in f if l.strip()}
rows = [json.loads(l) for l in open(args.infile, encoding="utf-8-sig") if l.strip()]
if args.limit:
    rows = rows[:args.limit]
todo = [r for r in rows if r["review_id"] not in done]
print(f"{len(rows)} reviews, {len(done)} done, {len(todo)} to go", flush=True)
if not todo:
    sys.exit(0)

tok = AutoTokenizer.from_pretrained(args.model)
kw = {"device_map": "auto"} if torch.cuda.is_available() else {}
if args.load_4bit:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=args.embed_on_cpu)
    if args.embed_on_cpu:
        kw["device_map"] = {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0,
                            "model.rotary_emb": 0, "lm_head": 0}
elif torch.cuda.is_available():
    kw["dtype"] = torch.float16
model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
dev = torch.device("cpu") if args.embed_on_cpu else next(model.parameters()).device
rev = getattr(model.config, "_commit_hash", None)
meta = {"model": args.model, "revision": rev, "load_4bit": args.load_4bit, "embed_on_cpu": args.embed_on_cpu, "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens, "transformers": __import__("transformers").__version__,
        "torch": torch.__version__, "started_utc": datetime.now(timezone.utc).isoformat()}
with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=1)
print(meta, flush=True)

t0 = time.time()
with open(args.out, "a", encoding="utf-8") as fo:
    for i, r in enumerate(todo, 1):
        prompt = PROMPT.replace("{text}", r["text"])
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        labels = parse_labels(raw)
        fo.write(json.dumps({"review_id": r["review_id"], "labels": labels,
                            "labels_lenient": parse_lenient(raw), "raw": raw},
                            ensure_ascii=False) + "\n"); fo.flush()
        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el/i:.1f}s/review  eta {(len(todo)-i)*el/i/60:.0f} min", flush=True)
print("DONE", flush=True)
