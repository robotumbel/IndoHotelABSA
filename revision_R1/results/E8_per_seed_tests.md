# E8 — Review-level paired tests per seed (HI-ABSA vs each system, Joint-F1, test-400)

For each seed, HI-ABSA of that seed is compared with the same-seed run of each trained system (deterministic systems are the same in every seed). Paired bootstrap (2000) and sign-flip permutation (5000) over whole reviews; Holm correction over the 13 comparisons within each seed. Smallest attainable p = 0.0002.

| HI-ABSA vs | Δ mean | Δ range over seeds | 95% CI (widest) | p range | Holm p range | seeds significant (Holm < .05) |
|---|---:|---|---|---|---|---:|
| mBERT | +0.078 | [+0.068, +0.087] | [+0.051, +0.105] | 0.0002–0.0002 | 0.0026–0.0026 | 5/5 |
| XLM-RoBERTa | +0.050 | [+0.044, +0.066] | [+0.027, +0.083] | 0.0002–0.0002 | 0.0026–0.0026 | 5/5 |
| IndoBERTweet | +0.039 | [+0.027, +0.051] | [+0.011, +0.068] | 0.0002–0.0014 | 0.0026–0.0084 | 5/5 |
| IndoBERT | +0.002 | [-0.005, +0.017] | [-0.017, +0.031] | 0.0142–0.9476 | 0.0568–1.0000 | 0/5 |
| HI-ABSA w/o AQA | +0.002 | [-0.007, +0.008] | [-0.019, +0.020] | 0.1722–0.8252 | 0.6887–1.0000 | 0/5 |
| HI-ABSA w/o LGF | -0.001 | [-0.005, +0.005] | [-0.015, +0.013] | 0.2150–0.7510 | 0.6449–1.0000 | 0/5 |
| HI-ABSA w/o AQA & LGF | +0.000 | [-0.007, +0.008] | [-0.019, +0.020] | 0.1868–0.8988 | 0.7055–1.0000 | 0/5 |
| mT5-small | +0.081 | [+0.068, +0.093] | [+0.048, +0.113] | 0.0002–0.0002 | 0.0026–0.0026 | 5/5 |
| LoRA Qwen2.5-0.5B | +0.074 | [+0.067, +0.084] | [+0.040, +0.115] | 0.0002–0.0002 | 0.0026–0.0026 | 5/5 |
| Lexicon baseline | +0.335 | [+0.331, +0.340] | [+0.297, +0.374] | 0.0002–0.0002 | 0.0026–0.0026 | 5/5 |
| Qwen2.5-7B zero-shot (lenient) | +0.078 | [+0.073, +0.082] | [+0.047, +0.108] | 0.0002–0.0002 | 0.0026–0.0026 | 5/5 |
| Qwen2.5-14B zero-shot (lenient) | +0.042 | [+0.038, +0.047] | [+0.012, +0.073] | 0.0004–0.0048 | 0.0026–0.0240 | 5/5 |
| Gemini draft (silver) | -0.108 | [-0.112, -0.104] | [-0.135, -0.081] | 0.0002–0.0002 | 0.0026–0.0026 | 5/5 |

## F1 on the neutral cell class (4-class cells, test-400), mean ± SD over seeds

| System | F1-NEU |
|---|---:|
| mBERT | 0.058 ± 0.034 |
| XLM-RoBERTa | 0.034 ± 0.001 |
| IndoBERTweet | 0.041 ± 0.013 |
| IndoBERT | 0.118 ± 0.039 |
| HI-ABSA w/o AQA | 0.102 ± 0.044 |
| HI-ABSA w/o LGF | 0.040 ± 0.013 |
| HI-ABSA w/o AQA & LGF | 0.080 ± 0.026 |
| mT5-small | 0.038 ± 0.012 |
| LoRA Qwen2.5-0.5B | 0.146 ± 0.040 |
| HI-ABSA | 0.038 ± 0.014 |
| Lexicon baseline | 0.109 |
| Qwen2.5-7B zero-shot (lenient) | 0.110 |
| Qwen2.5-14B zero-shot (lenient) | 0.164 |
| Gemini draft (silver) | 0.452 |
