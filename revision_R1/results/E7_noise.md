# E7 — Silver vs gold supervision and label noise (IndoBERT-base)

## (A) 5-fold CV over the 500 gold reviews (seed 42; test fold 100, val fold 100, gold-train 300)

Pooled over the five test folds (all 500 gold reviews scored once); blind columns use the 100 blind-annotated reviews.

| Training data | Joint-F1 (500, pooled) | per-fold SD | blind-100 vs gold | blind-100 vs blind |
|---|---:|---:|---:|---:|
| silver only (2,480) | 0.767 | 0.022 | 0.773 | 0.738 |
| gold only (300) | 0.594 | 0.017 | 0.599 | 0.613 |
| silver + gold (2,780) | 0.776 | 0.015 | 0.784 | 0.751 |

## (B) Synthetic noise on the silver labels and (C) confidence filtering (seeds 42/123/7, test-400)

| Training data | reviews | Joint-F1 test-400 | blind-100 vs blind |
|---|---:|---:|---:|
| silver, no added noise (E5) | 2,480 | 0.763±0.001 | 0.741±0.006 |
| +5% cells flipped (853) | 2,480 | 0.750±0.004 | 0.746±0.005 |
| +10% cells flipped (1,697) | 2,480 | 0.737±0.005 | 0.744±0.007 |
| +20% cells flipped (3,540) | 2,480 | 0.689±0.007 | 0.713±0.010 |
| filtered, ≤1 OOF disagreement | 1,730 | 0.745±0.003 | 0.740±0.004 |
| filtered, 0 OOF disagreements | 822 | 0.710±0.006 | 0.708±0.010 |
