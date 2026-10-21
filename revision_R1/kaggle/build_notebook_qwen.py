# -*- coding: utf-8 -*-
"""Builds kaggle/qwen_scale_zeroshot.ipynb (E10, R2.2): Qwen2.5-14B and -32B zero-shot,
same prompt and parsers as the Gemini silver drafts and the local Qwen2.5-7B run.
Run: python kaggle/build_notebook_qwen.py   (from revision_R1/)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
sys.path.insert(0, ROOT)
from preannotate_gemini import PROMPT  # embed the exact prompt string

MD = """# IndoHotelABSA — Qwen2.5 scale series, zero-shot (revision R2.2)

Qwen2.5-14B-Instruct and Qwen2.5-32B-Instruct, 4-bit NF4 (pre-quantised `unsloth/*-bnb-4bit` checkpoints, pinned
revisions), greedy decoding, max 256 new tokens, the prompt used for the Gemini silver drafts as the single user turn.
Raw completions are stored; `labels` uses the Gemini parser unchanged, `labels_lenient` also accepts bare JSON objects.
Together with the local Qwen2.5-7B run this gives a 7B / 14B / 32B series within one model family.

**Settings:** Accelerator = **GPU T4 x2**, Internet = On, input = dataset `indohotelabsa-splits` (needs `test_gold.jsonl`).
Then **Save Version → Save & Run All (Commit)**; download `/kaggle/working/preds_qwen_scale.zip`.
Expected time (estimate): 14B ≈ 1–2 h, 32B ≈ 3–5 h. Resumable: completed reviews are skipped on a re-run."""

SETUP = '''!pip install -q "bitsandbytes>=0.45"
import glob, json, os, re, time
from datetime import datetime, timezone
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST = glob.glob("/kaggle/input/**/test_gold.jsonl", recursive=True)[0]
OUT = "/kaggle/working/preds_qwen_scale"; os.makedirs(OUT, exist_ok=True)
MODELS = [("qwen25_14b_zeroshot", "unsloth/Qwen2.5-14B-Instruct-bnb-4bit", "63af63299c76dc5f79e0b0801e11304f95760ec2"),
          ("qwen25_32b_zeroshot", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit", "aa79e3472818bdec779075d80928602591d9f2a0")]
ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas", "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
POLARITIES = ["POSITIF", "NEGATIF", "NETRAL"]
PROMPT = ''' + repr(PROMPT) + '''

def parse_labels(raw):          # identical to preannotate_gemini.parse_labels
    if not raw: return []
    m = re.search(r"\\[.*?\\]", raw, re.DOTALL)
    if not m: return []
    try: arr = json.loads(m.group(0))
    except Exception: return []
    out, seen = [], set()
    for o in arr:
        if not isinstance(o, dict): continue
        a, p = o.get("aspect"), o.get("polarity")
        if a in ASPECTS and p in POLARITIES and a not in seen:
            out.append({"aspect": a, "polarity": p}); seen.add(a)
    return out

def parse_lenient(raw):
    objs = []
    for m in re.finditer(r"\\{[^{}]*\\}", raw or ""):
        try: objs.append(json.loads(m.group(0)))
        except Exception: pass
    return parse_labels(json.dumps(objs, ensure_ascii=False))

rows = [json.loads(l) for l in open(TEST, encoding="utf-8-sig") if l.strip()]
print(len(rows), "test reviews;", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
'''

RUN = '''for tag, repo, rev in MODELS:
    path = f"{OUT}/{tag}.jsonl"
    done = {json.loads(l)["review_id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    todo = [r for r in rows if r["review_id"] not in done]
    print(tag, len(done), "done,", len(todo), "to go")
    if not todo: continue
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    n_gpu = torch.cuda.device_count()
    print("GPUs visible:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
    if "32B" in repo and n_gpu < 2:
        print("skip 32B: needs GPU T4 x2"); continue
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, device_map="auto",
                                                 max_memory={i: "14GiB" for i in range(n_gpu)},
                                                 dtype=torch.float16).eval()
    json.dump({"model": repo, "revision": rev, "decoding": "greedy", "max_new_tokens": 256,
               "transformers": transformers.__version__, "torch": torch.__version__,
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "started_utc": datetime.now(timezone.utc).isoformat()}, open(path + ".meta.json", "w"), indent=1)
    dev = model.get_input_embeddings().weight.device
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fo:
        for i, r in enumerate(todo, 1):
            # render the chat template to text, then tokenise: same ids as apply_chat_template(tokenize=True)
            # and independent of whether this transformers version returns a tensor or a BatchEncoding
            text = tok.apply_chat_template([{"role": "user", "content": PROMPT.replace("{text}", r["text"])}],
                                           add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=256, do_sample=False,
                                     temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
            raw = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            fo.write(json.dumps({"review_id": r["review_id"], "labels": parse_labels(raw),
                                 "labels_lenient": parse_lenient(raw), "raw": raw}, ensure_ascii=False) + "\\n"); fo.flush()
            if i % 20 == 0:
                el = time.time() - t0
                print(f"  {tag} {i}/{len(todo)} {el/i:.1f}s/review eta {(len(todo)-i)*el/i/60:.0f} min", flush=True)
    del model; torch.cuda.empty_cache()
'''

ZIP = '''import shutil
shutil.make_archive("/kaggle/working/preds_qwen_scale", "zip", OUT)
print(os.listdir(OUT))
'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code": c.update({"execution_count": None, "outputs": []})
    return c


nb = {"cells": [cell("markdown", MD), cell("code", SETUP), cell("code", RUN), cell("code", ZIP)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "qwen_scale_zeroshot.ipynb"), "w", encoding="utf-8"), indent=1)
print("built qwen_scale_zeroshot.ipynb")
