# E2 — Agreement beyond overall Fleiss' kappa

Unit: one (review, aspect) cell, four values {NA, POS, NEG, NEU}; 3 annotators.

| Statistic | Verification, 500 reviews | Verification, blind-100 reviews | Blind round, same 100 |
|---|---:|---:|---:|
| cells | 3500 | 700 | 700 |
| Fleiss κ, 4-class, all cells | 0.894 | 0.908 | 0.975 |
| Krippendorff α, 4-class, all cells | 0.894 | 0.908 | 0.975 |
| Fleiss κ, detection (mentioned vs NA) | 0.914 | 0.921 | 0.979 |
| cells with ≥1 annotator marking the aspect | 1631 | 325 | 384 |
| Fleiss κ, 4-class, excl. unanimous-NA cells | 0.775 | 0.797 | 0.946 |
| Krippendorff α, 4-class, excl. unanimous-NA cells | 0.775 | 0.797 | 0.946 |
| cells all three mark as mentioned | 1410 | 284 | 373 |
| Fleiss κ, polarity only (3-class), all-three-mention cells | 0.905 | 0.928 | 0.982 |
| Krippendorff α, polarity only | 0.905 | 0.928 | 0.982 |
| unanimous cells | 3170 | 643 | 684 |
| 2-of-3 majority cells | 330 | 57 | 13 |
| three-way ties | 0 | 0 | 3 |

Category-specific Fleiss κ (Fleiss 1971), 4-class, all cells:

| Category | Verification 500 | Verification same 100 | Blind 100 |
|---|---:|---:|---:|
| NA | 0.914 | 0.921 | 0.979 |
| POSITIF | 0.889 | 0.896 | 0.971 |
| NEGATIF | 0.899 | 0.921 | 0.982 |
| NETRAL | 0.632 | 0.686 | 0.931 |

Per-class κ within polarity-only cells:

| Polarity | Verification 500 | Blind 100 |
|---|---:|---:|
| POSITIF | 0.929 | 0.980 |
| NEGATIF | 0.929 | 0.996 |
| NETRAL | 0.636 | 0.947 |

Per aspect (verification, 500 reviews):

| Aspect | κ all cells | κ excl. unanimous NA | cells ≥1 mention |
|---|---:|---:|---:|
| Location | 0.811 | 0.514 | 188 |
| Cleanliness | 0.872 | 0.735 | 244 |
| Service/Staff | 0.906 | 0.841 | 357 |
| Room & Facilities | 0.892 | 0.831 | 388 |
| Price/Value | 0.905 | 0.770 | 153 |
| Food & Beverage | 0.922 | 0.794 | 131 |
| Public Facilities | 0.859 | 0.690 | 170 |

Pairwise inter-annotator confusion (verification 500; counts over ordered annotator pairs, rows = annotator i, cols = annotator j):

| | NA | POSITIF | NEGATIF | NETRAL |
|---|---:|---:|---:|---:|
| NA | 11464 | 230 | 202 | 10 |
| POSITIF | 230 | 3916 | 82 | 68 |
| NEGATIF | 202 | 82 | 4042 | 68 |
| NETRAL | 10 | 68 | 68 | 258 |

Pairwise inter-annotator confusion (blind 100):

| | NA | POSITIF | NEGATIF | NETRAL |
|---|---:|---:|---:|---:|
| NA | 1902 | 7 | 13 | 2 |
| POSITIF | 7 | 728 | 1 | 10 |
| NEGATIF | 13 | 1 | 1300 | 2 |
| NETRAL | 2 | 10 | 2 | 200 |

2-of-3 splits in the verification round, majority ← minority label:

- NA<-POSITIF: 70
- NA<-NEGATIF: 51
- NEGATIF<-NA: 50
- POSITIF<-NA: 45
- POSITIF<-NEGATIF: 24
- NETRAL<-NEGATIF: 21
- POSITIF<-NETRAL: 17
- NETRAL<-POSITIF: 17
- NEGATIF<-POSITIF: 17
- NEGATIF<-NETRAL: 13
- NA<-NETRAL: 4
- NETRAL<-NA: 1
