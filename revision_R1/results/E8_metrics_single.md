# E8 — Full metric set on the 400-review gold test set (single predictions)

ACP-pred = polarity over cells where gold and prediction both mark the aspect (submitted protocol; the item set differs by system).
Mentioned-mF1 = polarity macro-F1 over every gold-mentioned cell, predicting NA counts as an error (same items for every system).
Cell4-mF1 = 4-class macro-F1 over all 7 × 400 cells. The gold-aspect setting (forced polarity on gold aspects) needs new inference and is not here.


| System | ACD_mF1 | ACP_pred_Acc | ACP_pred_mF1 | Mentioned_mF1 | Cell4_mF1 | Joint_P | Joint_R | Joint_F1 | ACP-pred items | runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lexicon baseline | 0.794 | 0.526 | 0.465 | 0.417 | 0.500 | 0.422 | 0.434 | 0.428 | 998 | 1 |
| mBERT | 0.771 | 0.840 | 0.586 | 0.552 | 0.554 | 0.608 | 0.747 | 0.670 | 1076 | 1 |
| XLM-RoBERTa | 0.760 | 0.859 | 0.598 | 0.570 | 0.544 | 0.586 | 0.782 | 0.670 | 1101 | 1 |
| IndoBERTweet | 0.802 | 0.858 | 0.599 | 0.561 | 0.584 | 0.668 | 0.757 | 0.710 | 1067 | 1 |
| IndoBERT | 0.850 | 0.872 | 0.630 | 0.596 | 0.633 | 0.731 | 0.784 | 0.756 | 1088 | 1 |
| HI-ABSA | 0.853 | 0.866 | 0.604 | 0.575 | 0.615 | 0.718 | 0.787 | 0.751 | 1099 | 1 |
| HI-ABSA w/o AQA | 0.856 | 0.869 | 0.627 | 0.596 | 0.635 | 0.731 | 0.787 | 0.758 | 1096 | 1 |
| HI-ABSA w/o LGF | 0.862 | 0.857 | 0.597 | 0.570 | 0.616 | 0.719 | 0.782 | 0.749 | 1104 | 1 |
| HI-ABSA w/o AQA & LGF | 0.856 | 0.869 | 0.617 | 0.586 | 0.627 | 0.729 | 0.787 | 0.757 | 1096 | 1 |
| mT5-small | 0.770 | 0.837 | 0.600 | 0.518 | 0.573 | 0.702 | 0.638 | 0.668 | 922 | 1 |
| LoRA Qwen2.5-0.5B | 0.745 | 0.881 | 0.643 | 0.521 | 0.590 | 0.794 | 0.600 | 0.684 | 824 | 1 |
| Gemini draft (silver) | 0.933 | 0.926 | 0.793 | 0.765 | 0.799 | 0.890 | 0.855 | 0.872 | 1117 | 1 |

## Neutral F1 with 95% review-level bootstrap CI (first run)

Gold test has 55 neutral cells.

| System | F1-NEU (ACP-pred) [95% CI] | F1-NEU (mentioned) [95% CI] |
|---|---|---|
| IndoBERT | 0.105 [0.000, 0.222] | 0.097 [0.000, 0.200] |
| HI-ABSA | 0.036 [0.000, 0.115] | 0.034 [0.000, 0.109] |
| LoRA Qwen2.5-0.5B | 0.120 [0.000, 0.245] | 0.094 [0.000, 0.197] |
| Gemini draft (silver) | 0.481 [0.362, 0.600] | 0.477 [0.358, 0.595] |

## Review-level paired tests of HI-ABSA against each system (Joint-F1, first run)

Bootstrap and permutation both resample whole reviews, so the 7 cells of a review stay together (replaces the per-cell McNemar test of submitted Table 11). Holm correction over the comparisons.

| HI-ABSA vs | Δ Joint-F1 | 95% CI | permutation p | Holm p |
|---|---:|---|---:|---:|
| Lexicon baseline | +0.323 | [+0.289, +0.357] | 0.0002 | 0.0022 |
| mBERT | +0.081 | [+0.065, +0.099] | 0.0002 | 0.0022 |
| XLM-RoBERTa | +0.081 | [+0.063, +0.098] | 0.0002 | 0.0022 |
| IndoBERTweet | +0.041 | [+0.026, +0.056] | 0.0002 | 0.0022 |
| IndoBERT | -0.005 | [-0.019, +0.008] | 0.4027 | 1.0000 |
| HI-ABSA w/o AQA | -0.007 | [-0.019, +0.006] | 0.3131 | 1.0000 |
| HI-ABSA w/o LGF | +0.002 | [-0.009, +0.013] | 0.7628 | 1.0000 |
| HI-ABSA w/o AQA & LGF | -0.006 | [-0.018, +0.007] | 0.3393 | 1.0000 |
| mT5-small | +0.083 | [+0.059, +0.107] | 0.0002 | 0.0022 |
| LoRA Qwen2.5-0.5B | +0.067 | [+0.040, +0.096] | 0.0002 | 0.0022 |
| Gemini draft (silver) | -0.121 | [-0.144, -0.099] | 0.0002 | 0.0022 |
