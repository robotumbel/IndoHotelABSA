# E4 — Overlap between splits and hotel coverage

Hotel key = (lower-cased hotel name, city).

| Level | train | val | test | test units also in train | test reviews whose unit is in train | val units in train | val ∩ test units |
|---|---:|---:|---:|---:|---:|---:|---:|
| hotel | 1740 | 98 | 386 | 208 (54%) | 214/400 (54%) | 51 | 11 |
| city | 55 | 47 | 55 | 55 (100%) | 400/400 (100%) | 47 | 47 |
| region | 6 | 6 | 6 | 6 (100%) | 400/400 (100%) | 6 | 6 |

## Reviews per hotel, full corpus (R2.8)

Hotels (name × city): 3075; reviews: 14988; mean 4.87, median 5, min 1, max 5.
The Places API returns at most five reviews per place; fewer than five remain where the place had fewer reviews or reviews were removed by the length filter or deduplication.

| reviews per hotel | hotels |
|---|---:|
| 1 | 16 |
| 2 | 16 |
| 3 | 38 |
| 4 | 199 |
| 5 | 2806 |

Labelled sample (train+val+test = 2980 reviews): 1961 hotels, mean 1.52, median 1, max 5 reviews per hotel.

## Files prepared for the grouped-split experiment (not yet trained)

- `data/train_silver_hotel_grouped.jsonl`: 2130 reviews — silver training with every hotel that occurs in val or test removed (350 reviews dropped).
- `data/train_silver_random_drop.jsonl`: 2130 reviews — size-matched random subsample of the full silver training set (seed 42), the control for the size reduction.
