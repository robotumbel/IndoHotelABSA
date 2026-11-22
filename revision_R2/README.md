# Second revision (R2) additions

Files added for the second revision of the IndoHotelABSA benchmark paper.

| Path | Content |
|---|---|
| `splits/hotel_grouped_v1/` | Official hotel-grouped split (recommended default evaluation); see its README |
| `data/silver_provenance.csv` | Per-review provenance of the 3,000 Gemini drafts (date, SDK, script, prompt SHA-256, decoding, partition). The serving checkpoint was not logged and is marked `unknown`; candidates are `gemini-2.5-flash` and `gemini-2.0-flash` |
| `scripts/lexicon_v2.py` | Corrected lexicon (v2) for the lexicon baseline. The v1 lexicon in `lexicon_baseline.py` and the lexicon-gated fusion (LGF) in HI-ABSA are deprecated |
| `scripts/build_hotel_grouped_split.py` | Rebuilds the grouped split and `train_silver_hotel_grouped.jsonl` from the Mendeley Data v2 files |
| `scripts/queue_r2_grouped.py`, `scripts/queue_r2_randdrop.py` | Retraining of the five trained encoders on the grouped split and on a size-matched random subsample, five seeds each |
| `scripts/score_r2_grouped.py` | Scores the runs (writes `results/R2_grouped_all.md`) |
| `preds_r2/` | Per-seed test predictions: `grouped_*` and `randdrop_*` for mBERT, XLM-RoBERTa, IndoBERTweet (IndoBERT and HI-ABSA runs are in `revision_R1/`) |
| `results/pred_lexicon_v1_test.jsonl`, `results/pred_lexicon_v2_test.jsonl` | Lexicon baseline predictions on test-400 with v1 and v2 |

The review text is not stored in this repository. The training file for the grouped split is produced from `train_silver.jsonl` (Mendeley Data v2, DOI 10.17632/9vhzg5wkf9.2) by the build script.
