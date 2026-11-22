# -*- coding: utf-8 -*-
"""Path configuration shared by all revision scripts.

Two layouts are supported:
  * the authors' working tree (paper2_sentiment/ with submission_DataInBrief/ beside it);
  * the public repository, where revision_R1/ sits in the repository root and the
    Mendeley Data v2 files (train_silver.jsonl, val_gold.jsonl, test_gold.jsonl,
    gold_final.jsonl, dataset_merged.jsonl, annotator_*_blind.jsonl,
    blind_subset_ids.txt, adjudication_record.jsonl, ...) are copied into the
    repository root, next to the training scripts. Override with IHA_ROOT if needed.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.abspath(os.path.join(_HERE, ".."))                       # revision_R1/
_work = os.path.abspath(os.path.join(REV, "..", ".."))                  # paper2_sentiment/
if os.environ.get("IHA_ROOT"):
    ROOT = os.path.abspath(os.environ["IHA_ROOT"])
elif os.path.isdir(os.path.join(_work, "submission_DataInBrief")):
    ROOT = _work
else:
    ROOT = os.path.abspath(os.path.join(REV, ".."))                     # repository root

_dib = os.path.join(ROOT, "submission_DataInBrief", "revision_R1")
if os.path.isdir(_dib):
    BLIND = os.path.join(_dib, "blind_check")                           # anotator_{i}_blind.jsonl
    DEP = os.path.join(_dib, "deposit_v2")
    BLIND_FILE = "anotator_{i}_blind.jsonl"
else:
    BLIND = DEP = ROOT                                                  # Mendeley v2 file names
    BLIND_FILE = "annotator_{i}_blind.jsonl"
