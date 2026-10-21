# E3 — Aspect × polarity distributions

## silver train (2,480)

| Aspect | POS | NEG | NEU | mentioned | NEU % of mentioned |
|---|---:|---:|---:|---:|---:|
| Location | 658 | 81 | 55 | 794 | 6.9 |
| Cleanliness | 354 | 423 | 4 | 781 | 0.5 |
| Service/Staff | 744 | 796 | 24 | 1564 | 1.5 |
| Room & Facilities | 854 | 1012 | 61 | 1927 | 3.2 |
| Price/Value | 373 | 295 | 97 | 765 | 12.7 |
| Food & Beverage | 384 | 255 | 64 | 703 | 9.1 |
| Public Facilities | 379 | 396 | 29 | 804 | 3.6 |
| **Total** | 3746 | 3258 | 334 | 7338 | 4.6 |

## gold val (100)

| Aspect | POS | NEG | NEU | mentioned | NEU % of mentioned |
|---|---:|---:|---:|---:|---:|
| Location | 29 | 3 | 2 | 34 | 5.9 |
| Cleanliness | 19 | 16 | 1 | 36 | 2.8 |
| Service/Staff | 30 | 31 | 1 | 62 | 1.6 |
| Room & Facilities | 36 | 34 | 4 | 74 | 5.4 |
| Price/Value | 14 | 12 | 5 | 31 | 16.1 |
| Food & Beverage | 17 | 7 | 1 | 25 | 4.0 |
| Public Facilities | 18 | 16 | 0 | 34 | 0.0 |
| **Total** | 163 | 119 | 14 | 296 | 4.7 |

## gold test (400)

| Aspect | POS | NEG | NEU | mentioned | NEU % of mentioned |
|---|---:|---:|---:|---:|---:|
| Location | 100 | 13 | 9 | 122 | 7.4 |
| Cleanliness | 63 | 115 | 0 | 178 | 0.0 |
| Service/Staff | 119 | 154 | 4 | 277 | 1.4 |
| Room & Facilities | 112 | 176 | 15 | 303 | 5.0 |
| Price/Value | 53 | 52 | 9 | 114 | 7.9 |
| Food & Beverage | 52 | 39 | 9 | 100 | 9.0 |
| Public Facilities | 48 | 59 | 9 | 116 | 7.8 |
| **Total** | 547 | 608 | 55 | 1210 | 4.5 |

## gold all (500)

| Aspect | POS | NEG | NEU | mentioned | NEU % of mentioned |
|---|---:|---:|---:|---:|---:|
| Location | 129 | 16 | 11 | 156 | 7.1 |
| Cleanliness | 82 | 131 | 1 | 214 | 0.5 |
| Service/Staff | 149 | 185 | 5 | 339 | 1.5 |
| Room & Facilities | 148 | 210 | 19 | 377 | 5.0 |
| Price/Value | 67 | 64 | 14 | 145 | 9.7 |
| Food & Beverage | 69 | 46 | 10 | 125 | 8.0 |
| Public Facilities | 66 | 75 | 9 | 150 | 6.0 |
| **Total** | 710 | 727 | 69 | 1506 | 4.6 |

## Blind-100: blind majority vs adjudicated gold (same 697 cells)

| Aspect | blind POS/NEG/NEU | gold POS/NEG/NEU |
|---|---|---|
| Location | 19/2/5 | 22/2/1 |
| Cleanliness | 19/36/3 | 12/35/0 |
| Service/Staff | 24/44/3 | 26/42/0 |
| Room & Facilities | 21/49/10 | 19/49/4 |
| Price/Value | 9/22/4 | 9/17/5 |
| Food & Beverage | 14/15/5 | 14/12/2 |
| Public Facilities | 16/52/6 | 13/18/1 |

## Star-rating bucket × aspect polarity (gold 500)

| Rating bucket | reviews | POS | NEG | NEU | reviews with both POS and NEG aspects |
|---|---:|---:|---:|---:|---:|
| 1-2★ | 206 | 42 | 491 | 7 | 30 |
| 3★ | 105 | 142 | 177 | 32 | 58 |
| 4-5★ | 189 | 526 | 59 | 30 | 33 |
