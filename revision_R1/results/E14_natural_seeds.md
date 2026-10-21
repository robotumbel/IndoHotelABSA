# E14a — Test results re-weighted to the corpus rating distribution (seeds predictions)

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
| mBERT | 0.686 | 0.706 | 0.783 | 0.796 | 0.568 | 0.576 |
| XLM-RoBERTa | 0.714 | 0.724 | 0.812 | 0.812 | 0.577 | 0.587 |
| IndoBERTweet | 0.724 | 0.740 | 0.819 | 0.829 | 0.566 | 0.573 |
| IndoBERT | 0.761 | 0.776 | 0.859 | 0.869 | 0.604 | 0.600 |
| HI-ABSA | 0.764 | 0.778 | 0.863 | 0.871 | 0.579 | 0.588 |
| HI-ABSA w/o AQA | 0.761 | 0.774 | 0.861 | 0.870 | 0.598 | 0.592 |
| HI-ABSA w/o LGF | 0.765 | 0.783 | 0.863 | 0.874 | 0.581 | 0.592 |
| HI-ABSA w/o AQA & LGF | 0.763 | 0.777 | 0.863 | 0.872 | 0.592 | 0.594 |
| mT5-small | 0.682 | 0.706 | 0.779 | 0.788 | 0.518 | 0.529 |
| LoRA Qwen2.5-0.5B | 0.690 | 0.718 | 0.758 | 0.783 | 0.545 | 0.554 |
| Gemini draft (silver) | 0.872 | 0.877 | 0.933 | 0.941 | 0.765 | 0.758 |

## Dashboard net sentiment s_a (Figure 8a) under both distributions

| | Lokasi | Kebersihan | Pelayanan | Kamar & Fasilitas | Harga | Makanan & Minuman | Fasilitas Pendukung |
|---|---:|---:|---:|---:|---:|---:|---:|
| gold, balanced | +0.71 | -0.29 | -0.13 | -0.21 | +0.01 | +0.13 | -0.09 |
| gold, natural | +0.83 | +0.19 | +0.34 | +0.22 | +0.44 | +0.43 | +0.31 |
| HI-ABSA, balanced | +0.94 | -0.19 | -0.12 | -0.18 | +0.08 | +0.19 | -0.08 |
| HI-ABSA, natural | +0.96 | +0.33 | +0.34 | +0.28 | +0.51 | +0.51 | +0.39 |
