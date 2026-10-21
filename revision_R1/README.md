# Revision R1: experiments and analyses

This folder contains everything behind the numbers of the revised companion research
article (benchmark, blind re-annotation audit, ablation, faithfulness, label noise).
It holds code, predictions, and results only. As for the data article, **no review text
is stored in this repository**: the corpus, all label tiers, the annotator files, and
the blind re-annotation files are in Mendeley Data v2
(DOI: [10.17632/9vhzg5wkf9.2](https://doi.org/10.17632/9vhzg5wkf9.2)).

## Setup

1. Download the Mendeley Data v2 files and copy them into the **repository root**, next
   to the training scripts. Needed: `train_silver.jsonl`, `val_gold.jsonl`,
   `test_gold.jsonl`, `gold_final.jsonl`, `dataset_merged.jsonl`,
   `annotator_{1,2,3}_blind.jsonl`, `annotator_{1,2,3}_verified.jsonl`,
   `blind_subset_ids.txt`, `adjudication_record.jsonl`,
   `annotation_prefilled_gold500.jsonl`.
2. `pip install -r requirements.txt`.
3. Run the scripts from the repository root, for example
   `python revision_R1/scripts/analysis_E1_E2_E3_E16.py`. `scripts/paths.py` locates
   the data; set `IHA_ROOT` to use another directory.

The released review text is PII-scrubbed (telephone numbers, e-mails, and URLs were
replaced by placeholders). The models in this folder were trained on the pre-scrub text,
so re-training on the release can differ very slightly from the reported numbers.

## Contents

| Path | What it is |
|---|---|
| `scripts/run_seeds_all.py` | E5: every trained system, five seeds (42, 123, 7, 2024, 99) |
| `scripts/queue_e4_e6.py` | E4: hotel-grouped split; E6: ablation variants |
| `scripts/retrain_logits_faith.py` | Logits for gold-aspect polarity and calibration; faithfulness (erasure, Integrated Gradients, attention intervention) |
| `scripts/e7_label_noise.py` | E7: silver vs. gold supervision, synthetic noise, confidence filtering |
| `scripts/llm_zeroshot_local.py` | Zero-shot open LLMs with the silver-drafting prompt |
| `kaggle/*.ipynb` | Large encoders (IndoBERT-large, XLM-R-large) and Qwen2.5-14B/32B on Kaggle |
| `scripts/analysis_*.py`, `score_*.py`, `per_seed_tests.py`, `blind100_seeds.py`, `final_tables.py` | All tables, tests, and statistics in the article |
| `scripts/make_figures_R1.py`, `make_pipeline_figure_R1.py` | Figures |
| `preds_seeds/`, `preds_large/`, `preds_e4/`, `preds_e7/` | Predictions per system and seed (labels only) |
| `preds_llm/`, `preds_qwen_scale/` | Zero-shot LLM predictions with raw model outputs |
| `logits_e8/`, `faith_e12/` | Validation/test logits; per-cell faithfulness scores |
| `splits/` | Review ids of the hotel-grouped and size-matched training sets, CV folds, out-of-fold disagreements |
| `results/` | Every result table as Markdown and JSON, and LaTeX rows of the main table |
| `../pred_*.jsonl` | Single-run predictions of the submitted version, used for comparison |

`finetune_hiabsa.py` gains `--pool {aqa,cls,mean,attn1,frozenq}` and
`--fusion {gate,add}` for the ablation; the default model is unchanged (identical
initialisation). `finetune_llm_lora.py` gains `--seed`.
