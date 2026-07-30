# IndoHotelABSA

Code for the paper **"IndoHotelABSA: An Explainable Aspect-Based Sentiment Analysis
Framework and Benchmark Dataset for Indonesian Hotel Reviews in Hospitality CRM"**
(*Natural Language Processing Journal*, 2026).

IndoHotelABSA is the first large-scale benchmark for aspect-based sentiment analysis
(ABSA) of **Indonesian hotel reviews** (14,988 reviews, 55 cities, 7 CRM aspects;
500-review human-verified gold set at Fleiss' κ = 0.894). This repository contains
the code to reproduce data construction, all baselines, the proposed **HI-ABSA**
model, the explainable CRM dashboard, and every table/figure in the paper.

- 📄 Paper: *Natural Language Processing Journal* (DOI to be added on acceptance)
- 📦 Dataset: **Mendeley Data**, DOI [10.17632/9vhzg5wkf9.1](https://doi.org/10.17632/9vhzg5wkf9.1)
- 🪪 Code license: MIT · Dataset license: CC BY 4.0

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Tested with Python 3.14, PyTorch 2.11 (CUDA 12.8), transformers 4.57 on an
NVIDIA RTX 4050 (6 GB).

## 2. Get the data

Download the corpus from Mendeley Data
([10.17632/9vhzg5wkf9.1](https://doi.org/10.17632/9vhzg5wkf9.1)) and place the JSONL
files in the repository root:

```
dataset_merged.jsonl   train_silver.jsonl   val_gold.jsonl   test_gold.jsonl   gold_final.jsonl
```

Each labelled record: `{review_id, hotel, city, rating, text, labels:[{aspect, polarity}], verified, gold}`.
Aspects (ID): Lokasi, Kebersihan, Pelayanan, Kamar & Fasilitas, Harga, Makanan & Minuman, Fasilitas Pendukung.
Polarity: POSITIF / NEGATIF / NETRAL. See `docs/annotation_guidelines.md` and `docs/datasheet.md`.

## 3. Reproduce the results

```bash
# Encoders (IndoBERT / IndoBERTweet / mBERT / XLM-R)
python finetune_indobert_absa.py --train train_silver.jsonl --val val_gold.jsonl --test test_gold.jsonl --model indobenchmark/indobert-base-p1 --pred_out pred_indobert.jsonl

# Proposed HI-ABSA (Aspect-Query Attention + Lexicon-Gated Fusion)
python finetune_hiabsa.py --train train_silver.jsonl --val val_gold.jsonl --test test_gold.jsonl --pred_out pred_hiabsa.jsonl
#   ablations: add --no_aqa and/or --no_lgf

# Generative / LLM baselines
python finetune_mt5_absa.py  --model google/mt5-small        # seq2seq generative
python finetune_llm_lora.py  --train train_silver.jsonl --val val_gold.jsonl --test test_gold.jsonl   # LoRA SFT
python llm_fewshot_absa.py                                   # zero-/few-shot LLM (set API key)
python lexicon_baseline.py                                   # rule-based lower bound

# Evaluation, significance, multi-seed, external SOTA (HoASA)
python evaluate_absa.py --gold test_gold.jsonl --pred pred_hiabsa.jsonl
python bootstrap_significance.py pred_hiabsa.jsonl pred_indobert.jsonl
python run_seeds.py            # 5-seed mean±std robustness
python hoasa_sota.py --model indobert   # cross-benchmark validation on HoASA

# Figures (PDF)
python make_result_figures.py && python make_corpus_figures.py
python make_dashboard_figure.py && python make_confusion_figure.py
python make_attention_figure.py && python make_pipeline_figure.py
```

`run_experiments.ps1` runs the full benchmark sequentially (Windows PowerShell).

## 4. Data-construction pipeline (optional, to rebuild the corpus)

```
scrape_reviews.py  ->  merge_dataset.py  ->  sample_for_annotation.py
   ->  preannotate_gemini.py (silver)  ->  build_annotator_apps.py (human verify)
   ->  adjudicate_gold.py  ->  compute_kappa.py  ->  split_dataset.py
```

## 5. Repository structure

| File(s) | Purpose |
|---|---|
| `finetune_hiabsa.py` | Proposed **HI-ABSA** (AQA + LGF), with `--no_aqa`/`--no_lgf` ablations |
| `finetune_indobert_absa.py` | Encoder baselines (IndoBERT/IndoBERTweet/mBERT/XLM-R) |
| `finetune_mt5_absa.py`, `finetune_llm_lora.py`, `llm_fewshot_absa.py` | Generative / LLM baselines |
| `lexicon_baseline.py` | Rule-based lexicon baseline |
| `evaluate_absa.py`, `bootstrap_significance.py`, `run_seeds.py` | Metrics, significance, multi-seed |
| `hoasa_sota.py` | Cross-benchmark validation on HoASA (IndoNLU) |
| `make_*_figure*.py` | Publication figures |
| `scrape_reviews.py` … `split_dataset.py` | Data-construction pipeline |
| `docs/` | Annotation guidelines, datasheet |

## 6. Citation

See `CITATION.cff`. Please cite both the paper and the dataset (Mendeley DOI above).

## 7. Ethics

Reviews were collected via the official Google Places API with all personally
identifying metadata removed at collection; the corpus is released for
non-commercial research under CC BY 4.0. A large language model was used **only** to
produce draft (silver) labels; all gold labels are human-verified.
