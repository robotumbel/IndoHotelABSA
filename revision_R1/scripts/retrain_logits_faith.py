# -*- coding: utf-8 -*-
"""Retrain encoders exactly as in E5 (same scripts, seeds, order of RNG calls) and keep
what the earlier runs threw away:

  * full 4-class logits for every val/test cell  -> E8 gold-aspect ACP, E15 neutral calibration
  * faithfulness of token explanations (HI-ABSA and IndoBERT only) -> E12
      - HI-ABSA: aspect-query attention alpha_a; Integrated Gradients (IG) for both models;
        random ranking as control
      - erasure: comprehensiveness / sufficiency at k = 10, 20, 50 % of content tokens (ERASER-style)
      - attention permutation and uniform attention (HI-ABSA): prediction flips, total variation
      - agreement between attention and IG rankings (Spearman, top-3 overlap)
    computed on test cells where the aspect is mentioned in gold and predicted correctly.
Removed tokens are excluded through the attention mask; [CLS]/[SEP] are never removed.
For HI-ABSA the lexicon input is held fixed, so erasure tests the token pathway only.

Output per run: logits_e8/<sys>_<seed>.npz and faith_e12/<sys>_<seed>.json
Sanity check: predictions are compared with preds_seeds/<sys>_<seed>.jsonl.

Usage (from paper2_sentiment/): python .../retrain_logits_faith.py [--systems ...] [--seeds ...] [--limit_faith N]
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, REV, BLIND, DEP, BLIND_FILE
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.argv_saved = sys.argv; sys.argv = [sys.argv[0]]
import finetune_hiabsa as FH
import finetune_indobert_absa as FI
sys.argv = sys.argv_saved

ap = argparse.ArgumentParser()
ap.add_argument("--systems", nargs="*", default=["hiabsa", "indobert", "xlmr", "mbert", "indobertweet"])
ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 7, 2024, 99])
ap.add_argument("--limit_faith", type=int, default=0, help="debug: cap faithfulness cells")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--train", default="train_silver.jsonl")
args = ap.parse_args()

ENC = {"indobert": "indobenchmark/indobert-base-p1", "xlmr": "xlm-roberta-base",
       "mbert": "bert-base-multilingual-cased", "indobertweet": "indolem/indobertweet-base-uncased"}
FAITH = {"hiabsa", "indobert"}
DEV = FH.DEVICE
os.makedirs(os.path.join(REV, "logits_e8"), exist_ok=True)
os.makedirs(os.path.join(REV, "faith_e12"), exist_ok=True)
KS = [0.1, 0.2, 0.5]


# ------------------------------------------------------------------ training (mirrors main())
def train_hiabsa(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok = FH.AutoTokenizer.from_pretrained(FH.MODEL_NAME)
    tr = DataLoader(FH.ABSADataset(FH.load_gold(args.train), tok), batch_size=16, shuffle=True)
    va = DataLoader(FH.ABSADataset(FH.load_gold("val_gold.jsonl"), tok), batch_size=16)
    te = DataLoader(FH.ABSADataset(FH.load_gold("test_gold.jsonl"), tok), batch_size=16)
    m = FH.HIABSA().to(DEV)
    FH.train(m, tr, va, epochs=args.epochs, lr=2e-5)
    m.load_state_dict(torch.load("hiabsa_best.pt", map_location=DEV))
    return m.eval(), tok, va, te


def train_encoder(name, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok = FI.AutoTokenizer.from_pretrained(name)
    tr = DataLoader(FI.ABSADataset(FI.load_gold(args.train), tok), batch_size=16, shuffle=True)
    va = DataLoader(FI.ABSADataset(FI.load_gold("val_gold.jsonl"), tok), batch_size=16)
    te = DataLoader(FI.ABSADataset(FI.load_gold("test_gold.jsonl"), tok), batch_size=16)
    m = FI.IndoBERTABSA(model_name=name).to(DEV)
    FI.train(m, tr, va, epochs=args.epochs, lr=2e-5)
    m.load_state_dict(torch.load("indobert_absa_best.pt", map_location=DEV))
    return m.eval(), tok, va, te


# ------------------------------------------------------------------ forward pieces
def head_from_H(m, H, mask, lex, alpha_override=None):
    """Post-encoder computation. Returns logits (B, A, 4) and alpha (B, A, L) or None."""
    if isinstance(m, FH.HIABSA):
        q = m.WQ(m.aspect_q); K = m.WK(H); V = m.WV(H)
        sc = torch.einsum("ad,bld->bal", q, K) / math.sqrt(m.d)
        sc = sc.masked_fill((mask == 0).unsqueeze(1), float("-inf"))
        alpha = torch.softmax(sc, -1) if alpha_override is None else alpha_override
        c = torch.einsum("bal,bld->bad", alpha, V)
        g = torch.sigmoid(m.wg(torch.cat([c, lex], -1)))
        c = c + g * m.Wl(lex)
        return torch.stack([m.heads[a](c[:, a]) for a in range(FH.A)], 1), alpha
    cls = H[:, 0]
    return torch.stack([h(cls) for h in m.heads], 1), None


def run(m, emb, mask, lex, alpha_override=None):
    H = m.encoder(inputs_embeds=emb, attention_mask=mask).last_hidden_state
    return head_from_H(m, H, mask, lex, alpha_override)


@torch.no_grad()
def all_logits(m, loader):
    ids_all, L = [], []
    for b in loader:
        ids, mask = b["input_ids"].to(DEV), b["attention_mask"].to(DEV)
        if isinstance(m, FH.HIABSA):
            lo = m(ids, mask, b["lex"].to(DEV))
        else:
            lo = m(ids, mask)
        L.append(lo.float().cpu().numpy())
        rid = b["review_id"]; ids_all += rid.tolist() if torch.is_tensor(rid) else list(rid)
    return np.array(ids_all), np.concatenate(L)


# ------------------------------------------------------------------ faithfulness
def ig(m, emb, base, mask, lex, a, c, steps=32):
    """Integrated gradients of logit (aspect a, class c) w.r.t. input embeddings. emb, base: (1, L, d)."""
    al = (torch.arange(steps, device=DEV, dtype=emb.dtype) + 0.5) / steps
    x = (base + al.view(-1, 1, 1) * (emb - base)).detach().requires_grad_(True)
    lo, _ = run(m, x, mask.expand(steps, -1), lex.expand(steps, -1, -1))
    g, = torch.autograd.grad(lo[:, a, c].sum(), x)
    return ((emb - base)[0] * g.mean(0)).sum(-1).detach()          # (L,)


@torch.no_grad()
def prob(m, emb, masks, lex, a, c):
    lo, _ = run(m, emb.expand(masks.size(0), -1, -1), masks, lex.expand(masks.size(0), -1, -1))
    return torch.softmax(lo[:, a].float(), -1)[:, c]


def faithfulness(m, tok, sysid, seed):
    test = FH.load_gold("test_gold.jsonl")
    hi = isinstance(m, FH.HIABSA)
    emb_layer = m.encoder.get_input_embeddings()
    pad_id = tok.pad_token_id
    rng = np.random.default_rng(seed)
    out = {"cells": 0, "erasure": {}, "perm": {"flip": [], "tvd": []}, "uniform": {"flip": [], "tvd": []},
           "agree": {"spearman": [], "top3_overlap": []}}
    methods = (["attention"] if hi else []) + ["ig", "random"]
    for meth in methods:
        out["erasure"][meth] = {f"comp@{k}": [] for k in KS} | {f"suff@{k}": [] for k in KS}
    n_done = 0
    for r in test:
        enc = tok(r["text"], truncation=True, max_length=192, return_tensors="pt")
        ids = enc["input_ids"].to(DEV); L = ids.size(1)
        mask = torch.ones(1, L, dtype=torch.long, device=DEV)
        lex = torch.tensor([FH.lexicon_feature(r["text"])], dtype=torch.float32, device=DEV)
        y = FH.encode_targets(r.get("labels", []))
        with torch.no_grad():
            emb = emb_layer(ids)
            lo, alpha = run(m, emb, mask, lex)
        pred = lo[0].argmax(-1).tolist()
        content = list(range(1, L - 1))                              # exclude [CLS], [SEP]
        if len(content) < 2:
            continue
        base = emb.clone(); base[0, 1:L - 1] = emb_layer(torch.tensor([[pad_id]], device=DEV))[0, 0]
        for a in range(FH.A):
            if y[a] == 0 or pred[a] != y[a]:
                continue
            c = pred[a]
            p_full = torch.softmax(lo[0, a].float(), -1)[c].item()
            scores = {"ig": ig(m, emb, base, mask, lex, a, c)[content].cpu().numpy(),
                      "random": rng.random(len(content))}
            if hi:
                scores["attention"] = alpha[0, a, content].float().cpu().numpy()
            masks = []
            plan = []
            for meth in methods:
                order = np.argsort(-scores[meth])
                for k in KS:
                    n = max(1, int(round(k * len(content))))
                    top = [content[i] for i in order[:n]]
                    mc = torch.ones(L, dtype=torch.long); mc[top] = 0                     # remove top
                    ms = torch.zeros(L, dtype=torch.long); ms[0] = ms[L - 1] = 1; ms[top] = 1  # keep top
                    masks += [mc, ms]; plan += [(meth, f"comp@{k}"), (meth, f"suff@{k}")]
            pr = prob(m, emb, torch.stack(masks).to(DEV), lex, a, c).cpu().numpy()
            for (meth, key), p in zip(plan, pr):
                out["erasure"][meth][key].append(p_full - float(p))
            if hi:
                al = alpha[0, a].clone()
                p0 = torch.softmax(lo[0, a].float(), -1)
                for kind in ("perm", "uniform"):
                    reps = 5 if kind == "perm" else 1
                    for _ in range(reps):
                        new = alpha.clone()
                        if kind == "perm":
                            perm = torch.tensor(rng.permutation(L), device=DEV)
                            new[0, a] = al[perm]
                        else:
                            new[0, a] = 1.0 / L
                        with torch.no_grad():
                            lo2, _ = run(m, emb, mask, lex, alpha_override=new)
                        p2 = torch.softmax(lo2[0, a].float(), -1)
                        out[kind]["flip"].append(int(p2.argmax().item() != c))
                        out[kind]["tvd"].append(0.5 * (p0 - p2).abs().sum().item())
                ra = np.argsort(np.argsort(-scores["attention"])); ri = np.argsort(np.argsort(-scores["ig"]))
                out["agree"]["spearman"].append(float(np.corrcoef(ra, ri)[0, 1]))
                t_a = set(np.argsort(-scores["attention"])[:3]); t_i = set(np.argsort(-scores["ig"])[:3])
                out["agree"]["top3_overlap"].append(len(t_a & t_i) / 3)
            n_done += 1
            if args.limit_faith and n_done >= args.limit_faith:
                break
        if args.limit_faith and n_done >= args.limit_faith:
            break
    out["cells"] = n_done
    return out


# ------------------------------------------------------------------ main loop
def preds_from_logits(rids, lg):
    return {int(r): [{"aspect": FH.ASPECTS[a], "polarity": FH.CLASSES[c]}
                     for a, c in enumerate(lg[i].argmax(-1)) if c] for i, r in enumerate(rids)}


for sysid in args.systems:
    for seed in args.seeds:
        tag = f"{sysid}_{seed}"
        npz = os.path.join(REV, "logits_e8", tag + ".npz")
        fj = os.path.join(REV, "faith_e12", tag + ".json")
        need_faith = sysid in FAITH and not os.path.exists(fj)
        if os.path.exists(npz) and not need_faith:
            print("[skip]", tag, flush=True); continue
        t0 = time.time(); print("[run ]", tag, flush=True)
        m, tok, va, te = train_hiabsa(seed) if sysid == "hiabsa" else train_encoder(ENC[sysid], seed)
        vr, vl = all_logits(m, va); tr_, tl = all_logits(m, te)
        np.savez_compressed(npz, val_ids=vr, val_logits=vl, test_ids=tr_, test_logits=tl)
        ref = os.path.join(REV, "preds_seeds", tag + ".jsonl")
        if os.path.exists(ref):
            old = {o["review_id"]: sorted((l["aspect"], l["polarity"]) for l in o["labels"])
                   for o in map(json.loads, open(ref, encoding="utf-8"))}
            new = preds_from_logits(tr_, tl)
            same = sum(old[r] == sorted((l["aspect"], l["polarity"]) for l in new[r]) for r in new)
            print(f"       reproduces E5 predictions on {same}/{len(new)} reviews", flush=True)
        if sysid in FAITH:
            f = faithfulness(m, tok, sysid, seed)
            json.dump(f, open(fj, "w"), indent=0)
            print(f"       faithfulness cells={f['cells']}", flush=True)
        print(f"       done {(time.time()-t0)/60:.1f} min", flush=True)
        del m; torch.cuda.empty_cache()
print("ALL DONE", flush=True)
