# IndoHotelABSA — Annotation Guidelines

These are the guidelines used to annotate Indonesian hotel reviews for
aspect-based sentiment analysis (ABSA). The task: for each review, identify which
of the seven aspect categories are **mentioned**, and assign each mentioned aspect
a **polarity** (POSITIF / NEGATIF / NETRAL).

## 1. Task summary

- **Unit of annotation:** the (review, aspect) cell. There are 7 aspects, so each
  review yields 7 decisions.
- Each aspect is labelled as one of: **N/A** (not mentioned), **POSITIF**,
  **NEGATIF**, **NETRAL**.
- Only aspects that are **explicitly or clearly implicitly** referred to are labelled
  with a polarity; all others are N/A (and simply omitted from `labels`).
- A single review may carry **different polarities for different aspects**
  (e.g., positive service but negative cleanliness).

## 2. Aspect categories and scope

| Aspect (Indonesian) | English | Scope / typical cues |
|---------------------|---------|----------------------|
| **Lokasi** | Location | Access, distance to attractions/airport/centre, strategic position, traffic |
| **Kebersihan** | Cleanliness | Cleanliness of room/bathroom/linen; odour; pests; stains; dust |
| **Pelayanan** | Service/Staff | Friendliness, speed, check-in/out, responsiveness, professionalism |
| **Kamar & Fasilitas** | Room & Facilities | Room size/comfort, bed, AC, TV, water heater, in-room amenities |
| **Harga** | Price/Value | Affordability, value for money, worth the price |
| **Makanan & Minuman** | Food & Beverage | Breakfast, restaurant, taste, variety, drinks |
| **Fasilitas Pendukung** | Public Facilities | Wi-Fi, parking, pool, gym, lift, lobby, common areas |

Guidance on overlaps:
- In-room amenities → **Kamar & Fasilitas**; shared/public amenities (Wi-Fi,
  pool, parking, lift) → **Fasilitas Pendukung**.
- "Bathroom is dirty" → **Kebersihan** (not Kamar & Fasilitas), because the salient
  judgement is about cleanliness.
- Staff behaviour → **Pelayanan**; physical breakfast quality → **Makanan &
  Minuman** (even if served by staff).

## 3. Polarity decision rules

| Polarity | Assign when the reviewer expresses… | Examples (ID) |
|----------|--------------------------------------|---------------|
| **POSITIF** | Satisfaction / praise about the aspect | "kamar nyaman", "pelayanan ramah", "lokasi strategis" |
| **NEGATIF** | Dissatisfaction / complaint about the aspect | "AC gak dingin", "kamar mandi bocor", "harga terlalu mahal" |
| **NETRAL** | Aspect mentioned factually with no clear sentiment, or genuinely mixed within the same aspect | "wifi tersedia hanya via hotspot", "sarapan ada" |

Rules:
1. **Negation and contrast** must be resolved from context: "kamarnya bagus tapi
   kotor" → Kamar & Fasilitas = POSITIF, Kebersihan = NEGATIF.
2. **Implicit sentiment** counts: "AC gak dingin" implies a negative Room &
   Facilities judgement even without the word "buruk".
3. Use **NETRAL** sparingly — only for factual mentions or truly balanced
   sentiment about a single aspect, not as a default for uncertainty.
4. Rating stars are context but **do not override** the text: annotate what the
   text says.
5. Informal spelling, slang, and code-mixing are common; interpret meaning, not
   surface form ("gak", "ga", "ngga" = "tidak"/not).

## 4. Two-tier, LLM-assisted, human-verified protocol

1. **Silver tier:** each sampled review is pre-annotated by a large language model
   prompted (in Indonesian) to output aspect–polarity pairs restricted to the seven
   categories. These are **drafts only**, flagged `llm_prefilled`.
2. **Gold tier:** three human annotators independently review the **same** 500
   reviews through an offline verification interface, correcting aspect and polarity
   where necessary. **The human decision is final.**
3. **Adjudication:** disagreements are resolved by **per-cell majority vote** across
   the three annotators (no three-way ties occurred).
4. **Agreement:** inter-annotator agreement is measured with **Fleiss' κ** over all
   (review, aspect) cells: overall κ = **0.894**; per-aspect κ ranges 0.811
   (Lokasi) to 0.922 (Makanan & Minuman).
5. **LLM–human agreement:** 92.1% of LLM-drafted cells were retained unchanged;
   7.9% were corrected.

## 5. Edge cases

- **Aspect mentioned but no evaluation** ("ada kolam renang") → NETRAL for that
  aspect (or N/A if it is a pure listing with no experiential content — annotator
  judgement, documented).
- **Multiple sentiments for one aspect** ("kamar luas tapi bau") → weigh the
  dominant/actionable judgement; if truly balanced, NETRAL.
- **Aspect not mentioned** → omit from `labels` (treated as N/A).
- **Sarcasm / world-knowledge-dependent** cues are annotated conservatively and
  flagged during adjudication.
