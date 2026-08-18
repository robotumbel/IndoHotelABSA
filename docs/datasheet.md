# Datasheet for IndoHotelABSA

A benchmark corpus of **Indonesian hotel reviews** annotated for **Aspect-Based
Sentiment Analysis (ABSA)**. This datasheet follows the *Datasheets for Datasets*
framework (Gebru et al., 2021).

> **Status:** complete, version 2 (August 2026). Collection (14,988 unique reviews,
> 10 July 2026) and annotation (3,000 LLM-drafted, 500 human-verified gold at
> Fleiss' κ = 0.894) are finished. Figures below reflect the released corpus.

---

## 1. Motivation

- **Purpose.** Aspect-based sentiment analysis of hotel reviews is central to
  hospitality CRM, yet no public ABSA benchmark with an independently verified gold
  tier existed for Indonesian, a widely spoken but comparatively low-resource
  language. IndoHotelABSA fills this gap and enables reproducible benchmarking of
  classical, encoder-based, and LLM-based ABSA methods.
- **Intended tasks.** Aspect Category Detection (ACD), Aspect Category Polarity
  (ACP), and joint ABSA, over seven CRM-meaningful aspect categories.
- **Created by.** The authors of the accompanying paper (research use).

## 2. Composition

- **Instances.** Each instance is a single hotel review (Indonesian free text)
  with: `review_id`, `hotel`, `city`, `rating` (1–5 stars), `text`, and
  `labels` (list of `{aspect, polarity}` pairs).
- **Identifiers.** Two identifier spaces exist. Corpus records are numbered in
  collection order. The annotation sample was renumbered 1..3000 when drawn, so the
  same integer denotes different reviews in the two file families. Every labelled
  record carries `corpus_review_id` linking it back to `dataset_merged.jsonl`, and
  the full correspondence is released as `sample_to_corpus_ids.tsv`. **Join on
  `corpus_review_id`, not `review_id`.**
- **Provenance.** The review texts are **secondary data**: user-generated content
  written by Google Maps reviewers, retrieved through the official Places API. The
  authors' original contribution is the collection design, curation, deduplication,
  sampling, aspect schema, annotation protocol, labels, and splits.
- **Count.** 14,988 unique reviews from 3,075 establishments. A class-balanced
  sample of 3,000 is annotated (1,200 negative / 600 neutral / 1,200 positive).
  The release comprises 2,480 **silver** (LLM-drafted, unverified) training
  reviews, a 500-review **gold** set (100 validation + 400 test) containing 1,506
  gold aspect annotations (727 negative, 710 positive, 69 neutral), and 20 excluded
  records that received an empty aspect set. The silver tier carries 7,338 aspect
  annotations (3,258 negative, 3,746 positive, 334 neutral).
- **Aspect categories (7).** Location, Cleanliness, Service/Staff,
  Room & Facilities, Price/Value, Food & Beverage, Public Facilities.
- **Polarity labels (3).** Positive, Negative, Neutral.
- **Coverage.** 55 cities across all six major Indonesian regions
  (Java 4,978, Sumatra 3,412, Bali & Nusa Tenggara 2,428, Kalimantan 1,585,
  Sulawesi 1,564, Maluku & Papua 1,021).
- **Rating distribution.** 1★:2,604 · 2★:810 · 3★:1,252 · 4★:2,010 · 5★:8,312.
- **Length.** Mean 348 characters, median 250.

### Does the dataset contain data that might be considered confidential or sensitive?

Answered in full rather than briefly, because the honest answer is "not none".

- **Reviewer identity metadata: none.** Reviewer names, profile and photo URLs,
  author attributions, and timestamps were never requested from the API and were
  never stored. Place identifiers were used transiently for establishment
  de-duplication and are not in the release.
- **Contact details inside the review text: removed.** A deterministic pass
  (`scrub_pii.py`) replaced telephone numbers, e-mail addresses, and URLs with
  `[PHONE]`, `[EMAIL]`, `[URL]`. It includes a contextual rule for the
  homoglyph-obfuscated WhatsApp numbers posted in listing-owner solicitations, which
  Unicode normalisation alone does not catch. The pass affected 111 of 14,988
  corpus records (0.74%): 109 phone numbers, 2 URLs, and no e-mail addresses. Within
  the labelled release it affected 20 silver, 1 validation, and 3 test records.
- **Person names inside the review text: retained, deliberately.** Indonesian NER
  over informal, code-mixed review text has a high false-positive rate on hotel,
  place, and brand names, and destructive substitution would damage exactly the
  aspect-bearing spans the dataset exists to support. Names occurring here are
  overwhelmingly staff first names used in praise ("mbak Rina ramah sekali"), which
  are analytically relevant and low-risk. **This is a residual disclosure risk and
  we state it rather than claim the corpus is free of personal data.** The takedown
  procedure in `LICENSE_DATA.md` covers removal requests.
- **Rules tested and rejected as destructive.** A bare-domain URL rule matched
  online travel agency brand names (`tiket.com`, `booking.com`, `trip.com`) that are
  ordinary, sometimes aspect-bearing content. An `@`-handle rule matched 34 spans,
  mostly the Indonesian preposition use of "@" for unit prices (`@200rb`), room
  numbers (`@325`), and staff first names (`@Mba`). Neither removes personal data
  here, and both would delete annotated content.

## 3. Collection Process

- **Source.** Google Places API (official, no scraping), Indonesian-language
  reviews of hotels, guest houses, and resorts.
- **Date.** 10 July 2026, single session, 02:00–05:20 WIB.
- **Endpoints and parameters.** Text Search
  (`/maps/api/place/textsearch/json`) for establishment discovery, Place Details
  (`/maps/api/place/details/json`, `fields=name,review`) for reviews. Both with
  `language=id`, Place Details additionally `reviews_no_translations=true`.
- **Sampling of establishments.** Each of 55 cities queried with eight fixed
  Indonesian search expressions (`hotel di {city}`, `hotel murah di {city}`,
  `hotel bintang 5 di {city}`, `hotel bintang 3 di {city}`, `penginapan di {city}`,
  `guest house di {city}`, `resort di {city}`, `hotel dekat pusat kota {city}`),
  paginated up to three pages, merged and de-duplicated by place identifier, capped
  at 60 establishments per city → 3,075 establishments. Place Details returns at
  most five reviews per establishment, which bounds corpus size.
- **Language handling.** Constrained at request time via `language=id` with
  translations disabled, **not** by a post-hoc language-identification model. A
  small minority of user-written English or regional-language reviews therefore
  remains.
- **Filtering and de-duplication.** Deterministic and threshold-free, no fuzzy or
  near-duplicate matching. Whitespace normalised. Reviews <20 or >2,000 characters
  dropped (169 removed). The deduplication key is the lower-cased normalised text
  truncated to its first 200 characters, with the first occurrence kept (148
  removed). 15,305 raw → 14,988
  unique.

## 4. Preprocessing / Labelling

- **Sampling for annotation.** 3,000 reviews, `random.seed(42)`, stratified by
  rating group at 40% negative (1–2★) / 20% neutral (3★) / 40% positive (4–5★) and
  spread across the 55 cities by round-robin selection.
- **Gold subset selection.** 500 reviews drawn from the 3,000 by rating-stratified
  random subsampling (206 at 1–2★, 105 at 3★, 189 at 4–5★). The realised
  identifiers are released as `gold_subset_ids.txt`.
- **Annotation protocol (two tiers).**
  1. *Silver tier:* Google Gemini (Flash family) produced draft aspect–polarity
     labels for all 3,000 sampled reviews on 16 July 2026, zero-shot, one call per
     review, at API-default decoding parameters, via `preannotate_gemini.py`. The
     3,000 drafts were accumulated over several resumable runs using
     `gemini-2.5-flash` (default, most runs) and `gemini-2.0-flash` (two runs). The
     serving checkpoint was not recorded per record. Records are flagged
     `llm_prefilled: true`, `verified: false`. The verbatim prompt is in
     `annotation_guidelines.md`.
  2. *Gold tier:* **the 500-review subset only** was checked and corrected by three
     independent annotators in an offline, self-contained interface, and adjudicated
     by per-cell majority vote. The human decision is final.
  3. *Blinding:* annotators were **not** blinded. Each aspect row was pre-set to
     the LLM draft and the task was to confirm or correct it. Agreement statistics
     therefore describe a shared correction task, not independent labelling from
     scratch, and are an upper bound on protocol reliability.
  4. *Blinded check:* a random 100-review sub-sample was re-annotated by the same
     three annotators with all labels cleared (18 August 2026). Blinded Fleiss' κ =
     0.975 (97.7% unanimous) against 0.908 (91.9%) for the same reviews verified
     from the draft, so agreement is not inflated by the shared suggestion. The
     blinded labels do, however, differ from the gold in 19.1% of cells, marking
     24.8% more aspect mentions, and 65% of the differing cells are aspects the gold
     left unmarked, mostly negative. The gold aspect inventory is therefore
     conservative. Files and analysis: `annotator_{1,2,3}_blind.jsonl`,
     `blind_kappa_report.md`, `compute_blind_kappa.py`.
  5. *Agreement:* over 3,500 (review, aspect) cells, all three agreed on 3,170
     (90.6%). Fleiss' κ = 0.894 (per-aspect 0.811–0.922), and pairwise Cohen's κ
     0.846–0.987. No three-way ties occurred, so the LLM tie-break provided for in
     the adjudication script was never exercised. 92.1% of drafts were retained,
     7.9% (275 cells) corrected.
- **Reproducibility.** Each annotator's raw labels, the per-cell adjudication
  record, the draft shown to annotators, and the gold identifiers are released,
  `compute_kappa.py` and `adjudicate_gold.py` regenerate the statistics and the gold
  file, and `verify_release.py` re-checks every count claimed in the article.
- **Transparency.** Released records distinguish LLM-drafted from human-verified
  labels record by record. All reported experiments use gold labels only.

## 5. Uses

- **Recommended.** ABSA benchmarking (ACD/ACP/joint), low-resource NLP research,
  LLM-assisted annotation and label-noise research, hospitality CRM analytics,
  cross-lingual transfer studies.
- **Cautions.**
  - The annotated sample is deliberately **rating-balanced, not
    prevalence-representative**, so its aspect and polarity frequencies are not
    population estimates. Re-sample from `dataset_merged.jsonl` if you need those.
  - The neutral class is rare (4.6% of gold and 4.6% of silver aspect annotations),
    per-class neutral metrics rest on few instances and need confidence intervals.
  - Silver labels are unverified model output and should not be treated as ground
    truth.
  - Aspect **detection** in the gold tier is conservative (see §4, blinded check):
    recall measured against this gold is likely understated, particularly for
    negative mentions. Polarity on detected aspects is more dependable.
  - Single-platform data: reviewer demographics and platform moderation may
    introduce selection bias.
- **Out of scope.** Not intended to identify or profile individual reviewers or to
  rank or disparage specific properties. Star ratings are platform-provided and
  carry their own biases.

## 6. Distribution

- **Availability.** Mendeley Data, DOI 10.17632/9vhzg5wkf9.1. Code on GitHub
  (MIT) and archived on Zenodo, DOI 10.5281/zenodo.21740913. A companion data
  descriptor appears in *Data in Brief*.
- **Format.** JSON Lines (`.jsonl`), UTF-8.
- **Licence, split by component.** See `LICENSE_DATA.md` for the full text.
  - *Authors' contribution* (annotations, schema, guidelines, splits, datasheet,
    statistics, code): **CC BY 4.0**.
  - *Review text and hotel names*: **third-party, user-generated content**.
    Copyright remains with the individual review authors, and the content is served
    subject to the source platform's terms. The dataset authors neither hold nor
    claim the right to relicense it, and CC BY 4.0 does **not** apply to it. It is
    redistributed for non-commercial academic research only, with attribution to
    Google Maps and to the review authors, and must not be used as or built into a
    substitute for the Places API.
- **Why the split.** Lawful collection through an official API does not by itself
  confer a right to redistribute and relicense third-party content. Version 1 of
  this record placed the whole deposit under CC BY 4.0, that was an overstatement
  and has been corrected.

## 7. Maintenance

- **Maintainer.** Xaverius Sika (`xaver_ius@unama.ac.id`), Universitas Dinamika
  Bangsa, Jambi, Indonesia.
- **Updates.** Corrections and expanded annotation may be released as versioned
  updates, and the version and changelog accompany the release.
- **Takedown.** Removal requests from Google or from a review author are actioned
  within 30 days, followed by a versioned re-release. See `LICENSE_DATA.md`.

## 8. Ethical & Legal Considerations

- Collected via the official API, no scraping.
- Reviewer identity metadata never stored, in-text contact details removed,
  residual in-text person names disclosed above rather than claimed absent.
- Redistribution rights of the review text are stated explicitly and not
  overclaimed (§6).
- Reviews express subjective opinions about businesses. The dataset must not be
  used to harass properties or individuals.
- LLM assistance is disclosed, including its scope: **all** silver labels are
  machine-drafted, and **only** the 500 gold reviews were human-verified.

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
