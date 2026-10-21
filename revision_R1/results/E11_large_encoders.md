# E11 — Large encoders (Kaggle, Tesla T4) vs base encoders, 5 seeds

Large: fp16 autocast, micro-batch 8 × 2 accumulation, lr 1e-5, transformers 5.16.1 (Kaggle). Base: fp32, batch 16, lr 2e-5, transformers 4.57.6 (local RTX 4050). Everything else identical.

| System | ACD-mF1 | Mentioned-mF1 | Joint-F1 test-400 | blind-100 vs gold | blind-100 vs blind |
|---|---:|---:|---:|---:|---:|
| XLM-RoBERTa base | 0.812±0.007 | 0.577±0.002 | 0.714±0.007 | 0.726±0.011 | 0.741±0.004 |
| XLM-RoBERTa large | 0.849±0.006 | 0.593±0.006 | 0.759±0.007 | 0.763±0.004 | 0.767±0.011 |
| IndoBERT base | 0.859±0.003 | 0.604±0.015 | 0.761±0.005 | 0.776±0.011 | 0.736±0.007 |
| IndoBERT large | 0.848±0.005 | 0.575±0.002 | 0.749±0.005 | 0.754±0.010 | 0.743±0.008 |
| HI-ABSA (base) | 0.863±0.003 | 0.579±0.006 | 0.764±0.004 | 0.772±0.005 | 0.737±0.003 |

Seed-matched review-level bootstrap of Joint-F1 differences (test-400):

| Comparison | Δ mean | Δ range | seeds with 95% CI excluding 0 |
|---|---:|---|---:|
| IndoBERT large − IndoBERT base | -0.012 | [-0.020, -0.002] | 2/5 |
| XLM-RoBERTa large − XLM-RoBERTa base | +0.045 | [+0.024, +0.057] | 5/5 |
| IndoBERT large − XLM-RoBERTa large | -0.010 | [-0.022, +0.003] | 2/5 |
| IndoBERT large − HI-ABSA (base) | -0.014 | [-0.019, -0.003] | 4/5 |
