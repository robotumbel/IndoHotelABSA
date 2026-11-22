# R2: significance tests for the hotel-grouped split (Joint-F1, test-400)

Seed-matched comparisons; paired bootstrap (2000) for the 95% CI and paired sign-flip permutation (5000) for p, both over whole reviews; Holm correction within each seed and family. Smallest attainable p = 0.0002.

## A. Grouped minus random drop (equal size, 2,130 reviews): effect of hotel separation

| System | Δ mean | Δ range over seeds | 95% CI (widest) | Holm p range | seeds significant (Holm < .05) |
|---|---:|---|---|---|---:|
| mBERT | -0.006 | [-0.021, +0.011] | [-0.034, +0.023] | 0.0030–1.0000 | 2/5 |
| XLM-RoBERTa | +0.005 | [-0.008, +0.022] | [-0.020, +0.035] | 0.0050–1.0000 | 1/5 |
| IndoBERTweet | +0.004 | [-0.002, +0.008] | [-0.013, +0.018] | 0.6303–1.0000 | 0/5 |
| IndoBERT | +0.001 | [-0.011, +0.009] | [-0.022, +0.021] | 0.3219–1.0000 | 0/5 |
| HI-ABSA | -0.002 | [-0.005, +0.006] | [-0.015, +0.019] | 0.6303–1.0000 | 0/5 |

## B. Grouped (2,130) minus standard (2,480): effect of the smaller training set

| System | Δ mean | Δ range over seeds | 95% CI (widest) | Holm p range | seeds significant (Holm < .05) |
|---|---:|---|---|---|---:|
| mBERT | -0.024 | [-0.041, -0.012] | [-0.054, -0.000] | 0.0010–0.2030 | 3/5 |
| XLM-RoBERTa | -0.011 | [-0.025, +0.000] | [-0.037, +0.012] | 0.0010–0.9432 | 2/5 |
| IndoBERTweet | -0.006 | [-0.011, -0.003] | [-0.019, +0.008] | 0.0264–1.0000 | 1/5 |
| IndoBERT | -0.003 | [-0.011, +0.008] | [-0.022, +0.018] | 0.2268–1.0000 | 0/5 |
| HI-ABSA | -0.007 | [-0.013, -0.003] | [-0.024, +0.009] | 0.0480–1.0000 | 1/5 |

## C. HI-ABSA minus each encoder, all trained on the grouped split: is the ranking kept?

| System | Δ mean | Δ range over seeds | 95% CI (widest) | Holm p range | seeds significant (Holm < .05) |
|---|---:|---|---|---|---:|
| mBERT | +0.095 | [+0.087, +0.104] | [+0.068, +0.122] | 0.0008–0.0008 | 5/5 |
| XLM-RoBERTa | +0.054 | [+0.047, +0.059] | [+0.030, +0.076] | 0.0008–0.0008 | 5/5 |
| IndoBERTweet | +0.039 | [+0.029, +0.052] | [+0.011, +0.069] | 0.0008–0.0016 | 5/5 |
| IndoBERT | -0.001 | [-0.011, +0.006] | [-0.024, +0.019] | 0.1372–0.9256 | 0/5 |

## Per-seed detail, family A (grouped minus random drop)

| System | seed 42 | seed 123 | seed 7 | seed 2024 | seed 99 |
|---|---:|---:|---:|---:|---:|
| mBERT | -0.021* | -0.004 | +0.004 | +0.011 | -0.019* |
| XLM-RoBERTa | +0.007 | +0.001 | +0.004 | +0.022* | -0.008 |
| IndoBERTweet | +0.008 | +0.003 | +0.004 | -0.002 | +0.006 |
| IndoBERT | +0.001 | -0.011 | +0.006 | +0.001 | +0.009 |
| HI-ABSA | -0.004 | +0.006 | -0.002 | -0.004 | -0.005 |

\* Holm p < .05.

## Pooled over five seeds (per-review counts summed over seeds), Holm over the five systems

| Comparison | System | Δ (pooled) | 95% CI | p | Holm p |
|---|---|---:|---|---:|---:|
| A grouped − random | mBERT | -0.006 | [-0.013, +0.001] | 0.0798 | 0.3191 |
| A grouped − random | XLM-RoBERTa | +0.005 | [+0.000, +0.011] | 0.0604 | 0.3019 |
| A grouped − random | IndoBERTweet | +0.004 | [-0.001, +0.008] | 0.1370 | 0.4109 |
| A grouped − random | IndoBERT | +0.001 | [-0.005, +0.007] | 0.7171 | 1.0000 |
| A grouped − random | HI-ABSA | -0.002 | [-0.008, +0.005] | 0.6215 | 1.0000 |
| B grouped − standard | mBERT | -0.024 | [-0.031, -0.019] | 0.0002 | 0.0010 |
| B grouped − standard | XLM-RoBERTa | -0.011 | [-0.017, -0.005] | 0.0002 | 0.0010 |
| B grouped − standard | IndoBERTweet | -0.006 | [-0.012, -0.001] | 0.0194 | 0.0582 |
| B grouped − standard | IndoBERT | -0.003 | [-0.009, +0.003] | 0.2733 | 0.2733 |
| B grouped − standard | HI-ABSA | -0.007 | [-0.013, -0.001] | 0.0274 | 0.0582 |
| C HI-ABSA − system (grouped) | mBERT | +0.095 | [+0.081, +0.110] | 0.0002 | 0.0008 |
| C HI-ABSA − system (grouped) | XLM-RoBERTa | +0.054 | [+0.042, +0.067] | 0.0002 | 0.0008 |
| C HI-ABSA − system (grouped) | IndoBERTweet | +0.039 | [+0.027, +0.052] | 0.0002 | 0.0008 |
| C HI-ABSA − system (grouped) | IndoBERT | -0.001 | [-0.009, +0.007] | 0.8136 | 0.8136 |
