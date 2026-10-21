# E14a — Test results re-weighted to the corpus rating distribution (single predictions)

Each test review is weighted by (corpus share of its star rating) / (test share). This estimates performance on reviews drawn at the natural rating distribution; it cannot correct for anything other than rating.

| Stars | corpus share | test share | weight |
|---|---:|---:|---:|
| 1 | 0.174 | 0.328 | 0.53 |
| 2 | 0.054 | 0.098 | 0.55 |
| 3 | 0.084 | 0.210 | 0.40 |
| 4 | 0.134 | 0.068 | 1.99 |
| 5 | 0.555 | 0.297 | 1.86 |

Effective sample size after weighting: 275 of 400 reviews.

| System | Joint-F1 balanced | Joint-F1 natural | ACD-mF1 balanced | ACD-mF1 natural | Mentioned-mF1 balanced | Mentioned-mF1 natural |
|---|---:|---:|---:|---:|---:|---:|
| Lexicon baseline | 0.428 | 0.497 | 0.794 | 0.818 | 0.417 | 0.439 |
| mBERT | 0.670 | 0.687 | 0.771 | 0.782 | 0.552 | 0.561 |
| XLM-RoBERTa | 0.670 | 0.679 | 0.760 | 0.754 | 0.570 | 0.586 |
| IndoBERTweet | 0.710 | 0.732 | 0.802 | 0.817 | 0.561 | 0.572 |
| IndoBERT | 0.756 | 0.777 | 0.850 | 0.861 | 0.596 | 0.610 |
| HI-ABSA | 0.751 | 0.768 | 0.853 | 0.862 | 0.575 | 0.583 |
| HI-ABSA w/o AQA | 0.758 | 0.769 | 0.856 | 0.862 | 0.596 | 0.585 |
| HI-ABSA w/o LGF | 0.749 | 0.766 | 0.862 | 0.870 | 0.570 | 0.577 |
| HI-ABSA w/o AQA & LGF | 0.757 | 0.774 | 0.856 | 0.866 | 0.586 | 0.586 |
| mT5-small | 0.668 | 0.690 | 0.770 | 0.777 | 0.518 | 0.523 |
| LoRA Qwen2.5-0.5B | 0.684 | 0.713 | 0.745 | 0.777 | 0.521 | 0.534 |
| Gemini draft (silver) | 0.872 | 0.877 | 0.933 | 0.941 | 0.765 | 0.758 |

## Dashboard net sentiment s_a (Figure 8a) under both distributions

| | Lokasi | Kebersihan | Pelayanan | Kamar & Fasilitas | Harga | Makanan & Minuman | Fasilitas Pendukung |
|---|---:|---:|---:|---:|---:|---:|---:|
| gold, balanced | +0.71 | -0.29 | -0.13 | -0.21 | +0.01 | +0.13 | -0.09 |
| gold, natural | +0.83 | +0.19 | +0.34 | +0.22 | +0.44 | +0.43 | +0.31 |
| HI-ABSA, balanced | +0.96 | -0.20 | -0.16 | -0.18 | +0.07 | +0.16 | -0.05 |
| HI-ABSA, natural | +0.99 | +0.31 | +0.31 | +0.29 | +0.47 | +0.49 | +0.41 |
