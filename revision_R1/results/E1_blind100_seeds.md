# E1 (5 seeds) — test-400 and blind-100

Blind-100: 100 test reviews annotated from scratch; 3 three-way-tie cells ignored.
Trained systems: mean ± SD over 5 seeds. LLM rows and the lexicon are deterministic single runs.

| System | Joint-F1 test-400 | blind-100 vs gold | blind-100 vs blind | Δ (blind − gold) | ACD-mF1 vs blind |
|---|---:|---:|---:|---:|---:|
| Lexicon baseline | 0.428 | 0.409 | 0.431 | +0.023 | 0.796 |
| mBERT | 0.686±0.005 | 0.717±0.006 | 0.718±0.003 | +0.001 | 0.829±0.007 |
| XLM-RoBERTa | 0.714±0.007 | 0.726±0.011 | 0.741±0.004 | +0.014 | 0.851±0.003 |
| IndoBERTweet | 0.724±0.006 | 0.750±0.006 | 0.723±0.005 | -0.026 | 0.831±0.009 |
| IndoBERT | 0.761±0.005 | 0.776±0.011 | 0.736±0.007 | -0.040 | 0.845±0.005 |
| HI-ABSA | 0.764±0.004 | 0.772±0.005 | 0.737±0.003 | -0.035 | 0.844±0.005 |
| HI-ABSA w/o AQA | 0.761±0.005 | 0.779±0.012 | 0.737±0.006 | -0.043 | 0.844±0.007 |
| HI-ABSA w/o LGF | 0.765±0.001 | 0.773±0.007 | 0.736±0.005 | -0.037 | 0.845±0.004 |
| HI-ABSA w/o AQA & LGF | 0.763±0.005 | 0.777±0.009 | 0.739±0.004 | -0.038 | 0.848±0.006 |
| mT5-small | 0.682±0.007 | 0.696±0.016 | 0.657±0.015 | -0.038 | 0.750±0.020 |
| LoRA Qwen2.5-0.5B | 0.690±0.007 | 0.702±0.005 | 0.633±0.009 | -0.070 | 0.698±0.012 |
| Qwen2.5-7B zero-shot (lenient) | 0.686 | 0.696 | 0.678 | -0.018 | 0.763 |
| Gemini draft (silver) | 0.872 | 0.883 | 0.757 | -0.126 | 0.848 |

Reference: adjudicated gold scored against the blind majority, Joint-F1 0.755.
