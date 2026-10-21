# E4 — Hotel-grouped training vs size-matched random drop (5 seeds)

Test-400 split by whether the review's hotel occurs in the full silver training set: seen 214, unseen 186.
Grouped training removes every val/test hotel (2,130 reviews); random drop keeps 2,130 random reviews.

| Training set | runs | Joint-F1 test-400 | seen-hotel reviews | unseen-hotel reviews | blind-100 vs blind |
|---|---:|---:|---:|---:|---:|
| IndoBERT, full silver (2,480) | 5 | 0.761±0.005 | 0.762±0.005 | 0.760±0.010 | 0.736±0.007 |
| IndoBERT, random drop (2,130) | 5 | 0.757±0.005 | 0.758±0.009 | 0.755±0.005 | 0.738±0.005 |
| IndoBERT, hotel-grouped (2,130) | 5 | 0.758±0.002 | 0.753±0.003 | 0.763±0.003 | 0.730±0.010 |
| HI-ABSA, full silver (2,480) | 5 | 0.764±0.004 | 0.758±0.005 | 0.770±0.003 | 0.737±0.003 |
| HI-ABSA, random drop (2,130) | 5 | 0.758±0.006 | 0.754±0.007 | 0.764±0.007 | 0.734±0.010 |
| HI-ABSA, hotel-grouped (2,130) | 5 | 0.757±0.006 | 0.751±0.005 | 0.764±0.007 | 0.736±0.009 |
