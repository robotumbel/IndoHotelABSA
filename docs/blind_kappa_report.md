# Blinded re-annotation - result

100 reviews, 700 cells per condition.

## 1. Agreement

| Condition | Fleiss' kappa | Unanimous |
|---|---:|---:|
| Blinded (labels cleared) | 0.975 | 97.7% |
| Verification (draft visible) | 0.908 | 91.9% |
| Difference | +0.067 | +5.9% |

| Aspect | Blinded | Verified | Diff |
|---|---:|---:|---:|
| Lokasi | 0.967 | 0.811 | +0.156 |
| Kebersihan | 0.980 | 0.886 | +0.094 |
| Pelayanan | 0.940 | 0.969 | -0.029 |
| Kamar & Fasilitas | 0.980 | 0.896 | +0.084 |
| Harga | 0.974 | 0.902 | +0.073 |
| Makanan & Minuman | 0.987 | 0.955 | +0.032 |
| Fasilitas Pendukung | 0.974 | 0.824 | +0.150 |

## 2. Independence of the returned files

- Blinded pairwise raw agreement: A1-A2 98.1%, A1-A3 98.6%, A2-A3 98.3%
- Verification pairwise raw agreement: A1-A2 92.4%, A1-A3 99.3%, A2-A3 92.0%
- Same aspect set on 94/100 reviews, identical click order on 66 of those.
- Label order follows the on-screen order in [88, 89, 86] reviews (blinded) vs [30, 28, 31] (verification) - consistent with working down an empty form, and inconsistent with copied files.

## 3. Blinded vs gold

- Blinded majority == adjudicated gold: 564/697 (80.9%)
- Blinded majority == LLM draft: 568/697 (81.5%)

## 4. Direction of divergence

- Aspect mentions: blinded 378 vs gold 303 (+24.8%)
- Added by blinded round: 87/133 (65%)
- Dropped: 12/133 (9%)
- Polarity changed: 34/133 (26%)

| gold -> blinded | cells |
|---|---:|
| NA -> NEGATIF | 60 |
| NA -> POSITIF | 17 |
| NEGATIF -> NETRAL | 13 |
| NA -> NETRAL | 10 |
| POSITIF -> NETRAL | 8 |
| POSITIF -> NA | 7 |
| NEGATIF -> POSITIF | 4 |
| NEGATIF -> NA | 4 |
| NETRAL -> NEGATIF | 4 |
| NETRAL -> POSITIF | 3 |

## Paragraph for the manuscript

To measure the effect of the non-blinded protocol rather than only caveat it, 100 of the 500 gold reviews were re-annotated by the same three annotators with every label cleared and no model output shown. Over these 700 cells, blinded agreement is Fleiss' kappa = 0.975 (97.7% of cells unanimous), against kappa = 0.908 (91.9%) for the same reviews under the original protocol. Agreement is therefore higher without the draft, so the reported kappa is not an artefact of three annotators accepting one shared suggestion. The returned files were checked for independence: the three chose the same aspect set on 94 of 100 reviews but recorded them in the same order on only 66 of those, and label order follows the on-screen aspect order in 88, 89, and 86 reviews respectively, as expected when each annotator works down an empty form, against 30, 28, and 31 in the draft-anchored round. The blinded labels nevertheless differ from the adjudicated gold in 133 of 697 cells (19.1%), and the divergence is directional: the blinded round marks 378 aspect mentions against 303 in the gold (+24.8%), and 65% of the differing cells are aspects the gold left unmarked, predominantly negative ones. The visible draft thus appears to have depressed aspect recall rather than inflated agreement: annotators working from an empty form detect mentions that the draft omitted and that the verification pass did not restore. Users should treat the gold aspect inventory as conservative, particularly for negative mentions. Because the same annotators had seen these reviews three weeks earlier, recall of the earlier session cannot be excluded. It would, however, pull the blinded labels towards the gold, whereas the observed divergence is away from it. The blinded label files are released with the dataset.
