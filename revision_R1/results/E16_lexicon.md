# E16 — Coverage of the Lexicon-Gated Fusion signal

Lexicon: 27 positive, 21 negative terms; cue words per aspect: Location 14, Cleanliness 11, Service/Staff 12, Room & Facilities 12, Price/Value 8, Food & Beverage 11, Public Facilities 8.
Matching is substring on the lower-cased review; ℓ_a = whole-review POS/NEG counts when any cue of aspect a occurs.

## silver train (2480 reviews; label source = silver)

Reviews with any non-zero ℓ: 2180 (87.9%).

| Aspect | cue hit (reviews) | of which substring-only | ℓ≠0 when aspect mentioned | cue absent when mentioned | ℓ≠0 when aspect NA |
|---|---:|---:|---:|---:|---:|
| Location | 929 | 81 | 662/794 (83%) | 110/794 | 231/1686 (14%) |
| Cleanliness | 1136 | 93 | 709/781 (91%) | 71/781 | 424/1699 (25%) |
| Service/Staff | 1155 | 161 | 1071/1564 (68%) | 452/1564 | 42/916 (5%) |
| Room & Facilities | 1992 | 150 | 1701/1927 (88%) | 136/1927 | 181/553 (33%) |
| Price/Value | 652 | 57 | 580/765 (76%) | 160/765 | 43/1715 (3%) |
| Food & Beverage | 1041 | 303 | 623/703 (89%) | 63/703 | 384/1777 (22%) |
| Public Facilities | 811 | 108 | 583/804 (73%) | 203/804 | 194/1676 (12%) |

## gold test (400 reviews; label source = gold)

Reviews with any non-zero ℓ: 363 (90.8%).

| Aspect | cue hit (reviews) | of which substring-only | ℓ≠0 when aspect mentioned | cue absent when mentioned | ℓ≠0 when aspect NA |
|---|---:|---:|---:|---:|---:|
| Location | 139 | 12 | 89/122 (73%) | 30/122 | 47/278 (17%) |
| Cleanliness | 192 | 13 | 155/178 (87%) | 23/178 | 37/222 (17%) |
| Service/Staff | 192 | 28 | 181/277 (65%) | 90/277 | 5/123 (4%) |
| Room & Facilities | 323 | 23 | 282/303 (93%) | 15/303 | 27/97 (28%) |
| Price/Value | 105 | 8 | 98/114 (86%) | 15/114 | 6/286 (2%) |
| Food & Beverage | 161 | 60 | 88/100 (88%) | 11/100 | 70/300 (23%) |
| Public Facilities | 131 | 18 | 87/116 (75%) | 28/116 | 41/284 (14%) |

Cue words matching only inside other words (gold test, reviews). This mixes legitimate affixed forms (kamarnya, makanan, kebersihan, staff) with false hits (merasa -> rasa, 'ac' inside unrelated words); it is an upper bound on false triggers, not a count of them.

- Food & Beverage / 'rasa': 42
- Service/Staff / 'staf': 38
- Food & Beverage / 'makan': 37
- Room & Facilities / 'ac': 37
- Cleanliness / 'bersih': 34
- Room & Facilities / 'kamar': 29
- Food & Beverage / 'menu': 29
- Service/Staff / 'pelayanan': 19
- Location / 'lokasi': 15
- Location / 'jalan': 13
- Food & Beverage / 'enak': 12
- Public Facilities / 'parkir': 11
- Room & Facilities / 'nyaman': 9
- Cleanliness / 'debu': 8
- Room & Facilities / 'kasur': 8
