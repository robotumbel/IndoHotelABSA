# E8 — Polarity under gold aspects vs predicted aspects (5 seeds, test-400)

Gold-aspect: every gold-mentioned cell, polarity = argmax over the POS/NEG/NEU logits (N/A logit ignored); the item set is identical for every system. Predicted-aspect: submitted protocol (cells both gold and model mark).
Logits come from the E8 re-training (IndoBERT and HI-ABSA reproduce E5 predictions on 400/400 reviews in every seed; XLM-R, mBERT and IndoBERTweet reproduce 308–400/400, GPU non-determinism), so predicted-aspect numbers may differ slightly from E5 for those three.

| System | gold-aspect Acc | gold-aspect mF1 | gold-aspect F1-NEU | pred-aspect Acc | pred-aspect mF1 | pred-aspect items |
|---|---:|---:|---:|---:|---:|---:|
| IndoBERT | 0.860±0.006 | 0.632±0.012 | 0.132±0.029 | 0.870±0.004 | 0.639±0.016 | 1087 |
| HI-ABSA | 0.862±0.007 | 0.609±0.006 | 0.059±0.017 | 0.870±0.004 | 0.608±0.006 | 1099 |
| XLM-RoBERTa | 0.855±0.003 | 0.595±0.002 | 0.034±0.001 | 0.867±0.003 | 0.604±0.002 | 1107 |
| mBERT | 0.834±0.003 | 0.591±0.012 | 0.064±0.034 | 0.850±0.003 | 0.602±0.014 | 1077 |
| IndoBERTweet | 0.847±0.006 | 0.600±0.011 | 0.065±0.027 | 0.861±0.006 | 0.603±0.008 | 1070 |

Gold-mentioned test cells: 1210 (of which neutral: 55).
