# Official hotel-grouped split (v1)

Use this split to measure generalisation to hotels never seen in training. It is the
recommended default evaluation for IndoHotelABSA; the original random review-level split
(`train_silver.jsonl`, `val_gold.jsonl`, `test_gold.jsonl`) remains available for
comparability with earlier numbers, and results should state which split was used.

| File | Reviews | Content |
|---|---|---|
| `train_ids.txt` | 2130 | silver training reviews whose hotel occurs in neither validation nor test |
| `val_ids.txt` | 100 | gold validation reviews (same as the standard split) |
| `test_ids.txt` | 400 | gold test reviews (same as the standard split) |
| `test_unseen_in_full_train_ids.txt` | 186 | test reviews whose hotel is absent even from the full 2,480-review silver set (subset for the standard split) |

Hotel identity = (lower-cased stripped hotel name, city). Guarantee: no hotel in
`train_ids.txt` occurs in `val_ids.txt` or `test_ids.txt` (checked by the build script).
Cities and regions still occur in all three partitions. 350 silver reviews
were dropped from the standard training set to obtain this guarantee. Rebuild with
`scripts/build_hotel_grouped_split.py`, run from the folder that holds the Mendeley Data v2
files; it also writes `train_silver_hotel_grouped.jsonl`, the same 2130 reviews with text
and labels.
