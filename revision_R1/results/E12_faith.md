# E12 — Faithfulness of token explanations (5 seeds, test cells mentioned in gold and predicted correctly)

Comprehensiveness (higher = more faithful): drop in p(predicted class) after removing the top-k% content tokens. Sufficiency (lower = more faithful): drop when only the top-k% are kept. AOPC = mean over k ∈ {10, 20, 50}%. Random = random token ranking (control). Tokens removed through the attention mask; HI-ABSA lexicon input fixed.

| Model | Explanation | cells/seed | comp@10 | comp@20 | comp@50 | AOPC comp | AOPC suff |
|---|---|---:|---:|---:|---:|---:|---:|
| HI-ABSA | attention | 956 | 0.279±0.031 | 0.385±0.035 | 0.555±0.037 | 0.406±0.033 | 0.099±0.008 |
| HI-ABSA | ig | 956 | 0.349±0.034 | 0.443±0.041 | 0.595±0.042 | 0.463±0.039 | -0.026±0.007 |
| HI-ABSA | random | 956 | 0.025±0.003 | 0.049±0.004 | 0.149±0.009 | 0.074±0.005 | 0.341±0.026 |
| IndoBERT | ig | 946 | 0.406±0.019 | 0.503±0.026 | 0.655±0.026 | 0.521±0.024 | 0.023±0.014 |
| IndoBERT | random | 946 | 0.029±0.005 | 0.060±0.005 | 0.200±0.004 | 0.096±0.003 | 0.427±0.006 |

Replacing the aspect-query attention of HI-ABSA (Jain & Wallace-style test):

| Model | Intervention | prediction flip rate | mean total variation of p(class) |
|---|---|---:|---:|
| HI-ABSA | perm attention | 0.001±0.001 | 0.001±0.000 |
| HI-ABSA | uniform attention | 0.001±0.001 | 0.001±0.000 |

Agreement between HI-ABSA attention and IG rankings over content tokens: Spearman 0.237±0.016, top-3 overlap 0.315±0.020.
