# IndoHotelABSA

Code for the paper **"IndoHotelABSA: An Aspect-Based Sentiment Analysis Benchmark for
Indonesian Hotel Reviews, with an Audit of LLM-Assisted Annotation"** (*Natural
Language Processing Journal*, under revision) and the companion data article in
*Data in Brief* 69 (2026) 113209, doi:[10.1016/j.dib.2026.113209](https://doi.org/10.1016/j.dib.2026.113209).

IndoHotelABSA is an aspect-based sentiment analysis (ABSA) resource of **Indonesian
hotel reviews**: 14,988 reviews of 3,075 hotels in 55 cities; 2,480 LLM-drafted (silver)
training reviews and 500 human-verified (gold) reviews for seven aspects, plus 100 test
reviews re-annotated from scratch to audit the LLM-assisted annotation. This repository
contains the code to reproduce data construction, all systems, and every table and
figure. **The experiments of the revised article are in [`revision_R1/`](revision_R1/).**

- 📄 Paper: *Natural Language Processing Journal* (under revision)
- 📦 Dataset: **Mendeley Data**, version 2, DOI [10.17632/9vhzg5wkf9.2](https://doi.org/10.17632/9vhzg5wkf9.2)
- 🪪 Code license: MIT · Data: split licence, see `LICENSE_DATA.md`

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
([10.17632/9vhzg5wkf9.2](https://doi.org/10.17632/9vhzg5wkf9.2)) and place the JSONL
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

# HI-ABSA (Aspect-Query Attention + Lexicon-Gated Fusion)
python finetune_hiabsa.py --train train_silver.jsonl --val val_gold.jsonl --test test_gold.jsonl --pred_out pred_hiabsa.jsonl
#   ablations: --no_aqa, --no_lgf, --pool {aqa,cls,mean,attn1,frozenq}, --fusion {gate,add}

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
| `finetune_hiabsa.py` | **HI-ABSA** (AQA + LGF) and its ablation variants |
| `finetune_indobert_absa.py` | Encoder baselines (IndoBERT/IndoBERTweet/mBERT/XLM-R) |
| `finetune_mt5_absa.py`, `finetune_llm_lora.py`, `llm_fewshot_absa.py` | Generative / LLM baselines |
| `lexicon_baseline.py` | Rule-based lexicon baseline |
| `evaluate_absa.py`, `bootstrap_significance.py`, `run_seeds.py` | Metrics, significance, multi-seed |
| `hoasa_sota.py` | Cross-benchmark validation on HoASA (IndoNLU) |
| `make_*_figure*.py` | Publication figures |
| `scrape_reviews.py` … `split_dataset.py` | Data-construction pipeline |
| `docs/` | Annotation guidelines, datasheet |
| `revision_R1/` | Experiments of the revised article: five-seed benchmark, blind re-annotation audit, ablation, faithfulness, label noise, large encoders, zero-shot LLMs |

## 6. Citation

See `CITATION.cff`. Please cite both the paper and the dataset (Mendeley DOI above).

## 7. Ethics

Reviews were collected via the official Google Places API; reviewer-identifying
metadata were never requested, and telephone numbers, e-mails, and URLs in review text
were replaced by placeholders. The annotations and code are released under CC BY 4.0
and MIT; the review text and hotel names are third-party content redistributed for
non-commercial research only and are not relicensed (`LICENSE_DATA.md`, 30-day takedown).
A large language model was used to produce draft (silver) labels; all evaluation labels
are human-verified, and 100 test reviews were re-annotated without any model output.
