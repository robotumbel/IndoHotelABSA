# Datasheet — IndoHotelABSA

A benchmark corpus of **Indonesian hotel reviews** annotated for **Aspect-Based
Sentiment Analysis (ABSA)**. This datasheet follows the *Datasheets for Datasets*
framework (Gebru et al., 2021).

> **Status:** complete. Collection (14,988 unique reviews) and annotation (3,000
> reviews; 500 human-verified gold at Fleiss' κ = 0.894) are finished. Figures
> below reflect the released corpus.

---

## 1. Motivation

- **Purpose.** Aspect-based sentiment analysis of hotel reviews is central to
  hospitality CRM, yet no public, human-annotated ABSA benchmark exists for
  Indonesian — a widely spoken but comparatively low-resource language.
  IndoHotelABSA fills this gap and enables reproducible benchmarking of
  classical, encoder-based, and LLM-based ABSA methods.
- **Intended tasks.** Aspect Category Detection (ACD), Aspect Category Polarity
  (ACP), and joint ABSA, over seven CRM-meaningful aspect categories.
- **Created by.** The authors of the accompanying paper (research use).

## 2. Composition

- **Instances.** Each instance is a single hotel review (Indonesian free text)
  with: `review_id`, `hotel`, `city`, `rating` (1–5 stars), `text`, and
  `labels` (list of `{aspect, polarity}` pairs).
- **Count.** 14,988 unique reviews collected; a class-balanced sample of 3,000 is
  annotated (1,200 negative / 600 neutral / 1,200 positive). The released labels
  comprise 2,480 silver (LLM-drafted) training reviews and a 500-review
  human-verified gold set (100 validation + 400 test) containing 1,506 gold aspect
  annotations (727 negative, 710 positive, 69 neutral).
- **Aspect categories (7).** Location; Cleanliness; Service/Staff;
  Room & Facilities; Price/Value; Food & Beverage; Public Facilities.
- **Polarity labels (3).** Positive; Negative; Neutral.
- **Coverage.** 55 cities across all six major Indonesian regions
  (Java 4,978; Sumatra 3,412; Bali & Nusa Tenggara 2,428; Kalimantan 1,585;
  Sulawesi 1,564; Maluku & Papua 1,021).
- **Rating distribution.** 1★:2,604 · 2★:810 · 3★:1,252 · 4★:2,010 · 5★:8,312.
- **Length.** Mean 348 characters, median 250.
- **Sensitive data.** None retained. Reviewer names and personal identifiers are
  discarded at collection; only review text and non-personal metadata are kept.

## 3. Collection Process

- **Source.** Google Places API (official, terms-compliant), Indonesian-language
  reviews of hotels/guest houses/resorts.
- **Sampling of establishments.** Each of 55 cities queried with multiple
  complementary search expressions (budget/mid/upscale hotels, guest houses,
  resorts); results merged and de-duplicated by place identifier to maximise
  unique-hotel coverage.
- **Timeframe.** Reviews as available on the platform at collection time (2026).
- **De-duplication.** Exact and normalised near-duplicate texts removed; reviews
  <20 or >2,000 characters excluded.

## 4. Preprocessing / Labelling

- **Cleaning.** Whitespace normalisation; length filtering; text-based
  de-duplication.
- **Sampling for annotation.** 3,000 reviews drawn stratified by rating group and
  spread evenly across cities (round-robin) to counter the positive skew.
- **Annotation protocol (LLM-assisted, human-verified).**
  1. *Silver tier:* an LLM produces draft aspect–polarity labels (flagged
     `llm_prefilled: true`, `verified: false`).
  2. *Gold tier:* three independent human annotators verify/correct every label
     via a dedicated offline interface; the human decision is final.
  3. *Agreement:* the full 500-review gold set is annotated by all three; Fleiss'
     κ = 0.894 (almost-perfect; per-aspect 0.811–0.922); disagreements resolved by
     per-cell majority vote (no three-way ties). 92.1% of LLM drafts were retained,
     7.9% corrected.
- **Transparency.** Released records distinguish LLM-drafted from human-verified
  labels. All reported experiments use human-verified gold labels only.

## 5. Uses

- **Recommended.** ABSA benchmarking (ACD/ACP/joint), low-resource NLP research,
  hospitality CRM analytics, cross-lingual transfer studies.
- **Out of scope / caution.** Not intended to identify or profile individual
  reviewers or specific properties; star ratings are platform-provided and may
  carry their own biases; positive-skew of raw ratings should be considered.

## 6. Distribution

- **Availability.** Deposited on **Mendeley Data** (reserved DOI
  10.17632/9vhzg5wkf9.1; active on publication) and released publicly upon paper
  acceptance, alongside a companion data descriptor in *Data in Brief*.
- **Format.** JSON Lines (`.jsonl`), UTF-8.
- **License.** CC BY 4.0 (Creative Commons Attribution 4.0 International).

## 7. Maintenance

- **Maintainer.** Corresponding author (see paper).
- **Updates.** Corrections and expanded annotation may be released as versioned
  updates; version and changelog will accompany the release.
- **Contact.** Via the corresponding author.

## 8. Ethical & Legal Considerations

- Collected via the official API in compliance with platform terms.
- No personal identifiers retained; content anonymised at source.
- Reviews express subjective opinions about businesses; the dataset must not be
  used to harass properties or individuals.
- LLM assistance is disclosed; final labels are human-verified.

---

### File schema (JSONL)

```json
{"review_id": 1, "hotel": "...", "city": "yogyakarta", "rating": 5,
 "text": "Kamarnya bersih, staf ramah, lokasi strategis.",
 "labels": [{"aspect": "Kebersihan", "polarity": "POSITIF"},
            {"aspect": "Pelayanan", "polarity": "POSITIF"},
            {"aspect": "Lokasi", "polarity": "POSITIF"}],
 "verified": true, "gold": true}
```

### Aspect label set (Indonesian ↔ English)

| Indonesian (annotation) | English (paper) |
|---|---|
| Lokasi | Location |
| Kebersihan | Cleanliness |
| Pelayanan | Service/Staff |
| Kamar & Fasilitas | Room & Facilities |
| Harga | Price/Value |
| Makanan & Minuman | Food & Beverage |
| Fasilitas Pendukung | Public Facilities |
