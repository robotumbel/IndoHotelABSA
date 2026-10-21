# E8 — Full metric set on the 400-review gold test set (seeds predictions)

ACP-pred = polarity over cells where gold and prediction both mark the aspect (submitted protocol; the item set differs by system).
Mentioned-mF1 = polarity macro-F1 over every gold-mentioned cell, predicting NA counts as an error (same items for every system).
Cell4-mF1 = 4-class macro-F1 over all 7 × 400 cells. The gold-aspect setting (forced polarity on gold aspects) needs new inference and is not here.
Values are mean ± SD over seeds (n runs in last column).

| System | ACD_mF1 | ACP_pred_Acc | ACP_pred_mF1 | Mentioned_mF1 | Cell4_mF1 | Joint_P | Joint_R | Joint_F1 | ACP-pred items | runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lexicon baseline | 0.794 | 0.526 | 0.465 | 0.417 | 0.500 | 0.422 | 0.434 | 0.428 | 998 | 1 |
| mBERT | 0.783±0.005 | 0.850±0.002 | 0.602±0.014 | 0.568±0.011 | 0.571±0.007 | 0.626±0.008 | 0.759±0.004 | 0.686±0.005 | 1080 | 5 |
| XLM-RoBERTa | 0.812±0.007 | 0.866±0.001 | 0.603±0.001 | 0.577±0.002 | 0.582±0.007 | 0.648±0.014 | 0.795±0.005 | 0.714±0.007 | 1111 | 5 |
| IndoBERTweet | 0.819±0.002 | 0.860±0.005 | 0.602±0.007 | 0.566±0.006 | 0.597±0.005 | 0.690±0.008 | 0.762±0.004 | 0.724±0.006 | 1071 | 5 |
| IndoBERT | 0.859±0.003 | 0.870±0.004 | 0.639±0.016 | 0.604±0.015 | 0.644±0.010 | 0.741±0.005 | 0.782±0.008 | 0.761±0.005 | 1087 | 5 |
| HI-ABSA | 0.863±0.003 | 0.870±0.004 | 0.608±0.006 | 0.579±0.006 | 0.626±0.005 | 0.739±0.005 | 0.790±0.005 | 0.764±0.004 | 1099 | 5 |
| HI-ABSA w/o AQA | 0.861±0.004 | 0.868±0.002 | 0.630±0.017 | 0.598±0.017 | 0.640±0.012 | 0.741±0.007 | 0.783±0.003 | 0.761±0.005 | 1091 | 5 |
| HI-ABSA w/o LGF | 0.863±0.003 | 0.872±0.002 | 0.610±0.005 | 0.581±0.004 | 0.626±0.003 | 0.739±0.002 | 0.793±0.002 | 0.765±0.001 | 1100 | 5 |
| HI-ABSA w/o AQA & LGF | 0.863±0.005 | 0.869±0.006 | 0.623±0.012 | 0.592±0.011 | 0.636±0.008 | 0.742±0.008 | 0.786±0.004 | 0.763±0.005 | 1094 | 5 |
| mT5-small | 0.779±0.014 | 0.845±0.007 | 0.594±0.010 | 0.518±0.008 | 0.576±0.003 | 0.713±0.008 | 0.655±0.017 | 0.682±0.007 | 937 | 5 |
| LoRA Qwen2.5-0.5B | 0.758±0.008 | 0.880±0.007 | 0.668±0.022 | 0.545±0.012 | 0.609±0.008 | 0.799±0.009 | 0.607±0.007 | 0.690±0.007 | 835 | 5 |
| Gemini draft (silver) | 0.933 | 0.926 | 0.793 | 0.765 | 0.799 | 0.890 | 0.855 | 0.872 | 1117 | 1 |

## Neutral F1 with 95% review-level bootstrap CI (first run)

Gold test has 55 neutral cells.

| System | F1-NEU (ACP-pred) [95% CI] | F1-NEU (mentioned) [95% CI] |
|---|---|---|
| IndoBERT | 0.135 [0.000, 0.172] | 0.125 [0.000, 0.161] |
| HI-ABSA | 0.042 [0.000, 0.118] | 0.040 [0.000, 0.105] |
| LoRA Qwen2.5-0.5B | 0.198 [0.054, 0.348] | 0.156 [0.037, 0.273] |
| Gemini draft (silver) | 0.481 [0.362, 0.600] | 0.477 [0.358, 0.595] |

## Review-level paired tests of HI-ABSA against each system (Joint-F1, first run)

Bootstrap and permutation both resample whole reviews, so the 7 cells of a review stay together (replaces the per-cell McNemar test of submitted Table 11). Holm correction over the comparisons.

| HI-ABSA vs | Δ Joint-F1 | 95% CI | permutation p | Holm p |
|---|---:|---|---:|---:|
| Lexicon baseline | +0.331 | [+0.297, +0.366] | 0.0002 | 0.0022 |
| mBERT | +0.068 | [+0.052, +0.084] | 0.0002 | 0.0022 |
| XLM-RoBERTa | +0.044 | [+0.029, +0.060] | 0.0002 | 0.0022 |
| IndoBERTweet | +0.027 | [+0.010, +0.043] | 0.0014 | 0.0070 |
| IndoBERT | -0.002 | [-0.017, +0.013] | 0.8066 | 1.0000 |
| HI-ABSA w/o AQA | -0.007 | [-0.019, +0.005] | 0.2603 | 1.0000 |
| HI-ABSA w/o LGF | -0.005 | [-0.015, +0.006] | 0.3835 | 1.0000 |
| HI-ABSA w/o AQA & LGF | -0.007 | [-0.020, +0.005] | 0.2911 | 1.0000 |
| mT5-small | +0.080 | [+0.059, +0.100] | 0.0002 | 0.0022 |
| LoRA Qwen2.5-0.5B | +0.067 | [+0.039, +0.096] | 0.0002 | 0.0022 |
| Gemini draft (silver) | -0.112 | [-0.136, -0.090] | 0.0002 | 0.0022 |
