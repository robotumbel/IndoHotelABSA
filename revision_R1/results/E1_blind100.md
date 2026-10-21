# E1 — Systems scored on the 100 blind-annotated test reviews

Blind subset: 100 reviews, all in the 400-review test set; 3 three-way-tie cells in the blind round are ignored (697 cells scored).
Sanity check: the 'Gemini' predictions equal the silver drafts on 400/400 test reviews.
Predictions are the single-seed runs behind the submitted Table 12 (E5 will replace them).

| System | Joint-F1 test-400 (gold) | Joint-F1 blind-100 vs gold | Joint-F1 blind-100 vs blind | Δ (blind − gold) | ACD-mF1 vs blind | 4-class cell mF1 vs blind |
|---|---:|---:|---:|---:|---:|---:|
| Lexicon baseline | 0.428 | 0.409 | 0.431 | +0.023 | 0.796 | 0.514 |
| mBERT | 0.670 | 0.694 | 0.691 | -0.003 | 0.794 | 0.560 |
| XLM-RoBERTa | 0.670 | 0.677 | 0.702 | +0.025 | 0.823 | 0.562 |
| IndoBERTweet | 0.710 | 0.729 | 0.725 | -0.004 | 0.837 | 0.593 |
| IndoBERT | 0.756 | 0.762 | 0.740 | -0.022 | 0.856 | 0.620 |
| HI-ABSA | 0.751 | 0.766 | 0.739 | -0.028 | 0.861 | 0.608 |
| mT5-small | 0.668 | 0.657 | 0.634 | -0.024 | 0.731 | 0.535 |
| LoRA Qwen2.5-0.5B | 0.684 | 0.693 | 0.633 | -0.060 | 0.692 | 0.562 |
| Gemini draft (silver) | 0.872 | 0.883 | 0.757 | -0.126 | 0.848 | 0.669 |

Reference: adjudicated gold scored against the blind majority on the same cells: Joint-F1 0.755, ACD-mF1 0.846, cell mF1 0.668.

Review-level paired bootstrap (2,000 resamples of the 100 reviews), Joint-F1 gap Gemini draft − encoder:

| Encoder | vs adjudicated gold: gap [95% CI] | vs blind majority: gap [95% CI] |
|---|---|---|
| IndoBERT | +0.121 [+0.068, +0.172] | +0.016 [-0.025, +0.057] |
| HI-ABSA | +0.117 [+0.068, +0.162] | +0.018 [-0.025, +0.059] |
