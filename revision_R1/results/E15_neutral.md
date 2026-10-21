# E15 — Neutral calibration pilot: additive bias on the NEU logit (5 seeds)

Bias b chosen on the 100-review validation set (only 14 neutral cells) by maximising 4-class cell macro-F1, grid −1 … 6 step 0.25, then applied unchanged to test-400. 'Oracle' = best b on test itself, an upper bound only.

| System | b (val, per seed) | F1-NEU before → after | Mentioned-mF1 before → after | Joint-F1 before → after | oracle F1-NEU |
|---|---|---|---|---|---:|
| IndoBERT | 0.25, 0, -0.25, 2, 1 | 0.125±0.041 → 0.148±0.071 | 0.604±0.015 → 0.610±0.019 | 0.761±0.005 → 0.752±0.018 | 0.273±0.023 |
| HI-ABSA | 1, 0, 0.5, 1.5, 1.25 | 0.040±0.014 → 0.137±0.075 | 0.579±0.006 → 0.610±0.023 | 0.764±0.004 → 0.755±0.008 | 0.285±0.029 |
