# -*- coding: utf-8 -*-
"""E1 with five-seed predictions: every system on test-400 and on blind-100
(adjudicated gold vs blind majority). Reuses the helpers of score_llm_blind.py.

Usage: python scripts/blind100_seeds.py      Writes results/E1_blind100_seeds.md
"""
import os, statistics, sys
sys.argv = [sys.argv[0]]                       # score_llm_blind reads extra args as files
sys.path.insert(0, os.path.dirname(__file__))
import score_llm_blind as S                    # builds gold, blind majority, ties

REV, ROOT = S.REV, S.ROOT
SEEDS = [42, 123, 7, 2024, 99]
SYS = [("Lexicon baseline", None, "pred_lex_gold.jsonl"), ("mBERT", "mbert", None), ("XLM-RoBERTa", "xlmr", None),
       ("IndoBERTweet", "indobertweet", None), ("IndoBERT", "indobert", None), ("HI-ABSA", "hiabsa", None),
       ("HI-ABSA w/o AQA", "hiabsa_noaqa", None), ("HI-ABSA w/o LGF", "hiabsa_nolgf", None),
       ("HI-ABSA w/o AQA & LGF", "hiabsa_none", None), ("mT5-small", "mt5", None),
       ("LoRA Qwen2.5-0.5B", "lora_qwen05b", None), ("Qwen2.5-7B zero-shot (lenient)", None, "QWEN"),
       ("Gemini draft (silver)", None, "pred_gemini_fewshot.jsonl")]


def runs(sysid, single):
    if sysid:
        return [{r: S.cells(o["labels"]) for r, o in S.load(os.path.join(REV, "preds_seeds", f"{sysid}_{s}.jsonl")).items()}
                for s in SEEDS]
    if single == "QWEN":
        raw = S.load(os.path.join(REV, "preds_llm", "qwen25_7b_zeroshot.jsonl"))
        return [{r: S.cells(o.get("labels_lenient")) for r, o in raw.items()}]
    return [{r: S.cells(o["labels"]) for r, o in S.load(os.path.join(ROOT, single)).items()}]


def ms(xs):
    return f"{statistics.mean(xs):.3f}" + (f"±{statistics.pstdev(xs):.3f}" if len(xs) > 1 else "")


L = ["# E1 (5 seeds) — test-400 and blind-100", "",
     f"Blind-100: {len(S.ids_b)} test reviews annotated from scratch; {len(S.ties)} three-way-tie cells ignored.",
     "Trained systems: mean ± SD over 5 seeds. LLM rows and the lexicon are deterministic single runs.", "",
     "| System | Joint-F1 test-400 | blind-100 vs gold | blind-100 vs blind | Δ (blind − gold) | ACD-mF1 vs blind |",
     "|---|---:|---:|---:|---:|---:|"]
for name, sysid, single in SYS:
    P = runs(sysid, single)
    t = [S.joint(S.gold, p, S.IDS) for p in P]
    g = [S.joint(S.gold, p, S.ids_b, S.ties) for p in P]
    b = [S.joint(S.bgold, p, S.ids_b, S.ties) for p in P]
    a = [S.acd(S.bgold, p, S.ids_b, S.ties) for p in P]
    L.append(f"| {name} | {ms(t)} | {ms(g)} | {ms(b)} | {statistics.mean(b) - statistics.mean(g):+.3f} | {ms(a)} |")
ref = S.joint(S.bgold, S.gold, S.ids_b, S.ties)
L += ["", f"Reference: adjudicated gold scored against the blind majority, Joint-F1 {ref:.3f}."]
open(os.path.join(REV, "results", "E1_blind100_seeds.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
