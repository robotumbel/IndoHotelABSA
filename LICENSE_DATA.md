# IndoHotelABSA. Licence and rights

This release contains two kinds of material with **different rights status**. Read
both sections before reusing anything.

---

## 1. Authors' contribution. CC BY 4.0

The following are the original work of the authors and are released under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/):

- all aspect–polarity annotations (silver and gold), in every `.jsonl` file,
- the seven-category aspect schema and `annotation_guidelines.md`,
- the train / validation / test split definitions and the gold subset identifiers,
- the per-annotator label files and the adjudication record,
- `datasheet.md`, `README.md`, and all summary statistics,
- all code in this repository and in the companion GitHub repository (MIT).

You may share and adapt this material for any purpose, including commercially,
provided you give appropriate credit.

**How to credit:** cite the data article and the Mendeley Data record (see
`README.md`).

---

## 2. Review text and hotel names, third-party content, NOT relicensed

The `text` and `hotel` fields are **not** the authors' work. They are
user-generated content written by Google Maps reviewers and retrieved through the
official Google Places API. Copyright in each review remains with the person who
wrote it, and the content is served subject to Google's terms.

**The authors neither hold nor claim the right to relicense this material, and
CC BY 4.0 does not apply to it.**

It is redistributed here on the following terms:

1. **Non-commercial academic research use only.** Do not use the review text or
   hotel names in a commercial product or service.
2. **Attribution is required.** Any use must credit Google Maps as the source
   platform and acknowledge that the reviews were written by individual users.
3. **No redistribution as a place-data product.** This corpus is a static research
   dataset of short texts. It provides no lookup, mapping, or place-discovery
   functionality and must not be used as, or built into, a substitute for the
   Places API.
4. **You remain bound by the source platform's terms.** Reuse of this material does
   not grant you any right against Google or against the review authors.

### What was and was not retained

| Field | Status |
|---|---|
| Review text | Retained, with in-text telephone numbers, e-mail addresses, and URLs replaced by `[PHONE]`, `[EMAIL]`, `[URL]` |
| Star rating | Retained |
| Hotel name | Retained |
| City | Retained (derived from the query, not from the API record) |
| Reviewer name, profile URL, photo URL, author attribution | **Never requested, never stored** |
| Review timestamp | **Never requested, never stored** |
| Place ID, coordinates, any other Places field | Used transiently for de-duplication at collection time. **not stored** |

---

## 3. Takedown procedure

We will remove material on request, without argument.

**Contact:** Xaverius Sika, `xaver_ius@unama.ac.id`
(Universitas Dinamika Bangsa, Jambi, Indonesia)

If you are Google, or the author of a review contained in this corpus, and you want
a record removed:

1. Send the review text, or the `review_id`, or a description sufficient to locate
   the record, to the address above.
2. We will remove the record, and any annotation derived from it, within **30 days**
   of receipt.
3. We will publish a new version of the Mendeley Data record with the material
   removed, and note the removal in the version history. Earlier versions are
   immutable by design of the archive. We will ask Mendeley Data to restrict access
   to the superseded version where the platform permits it.

The same procedure applies to a request to remove personal information that our
automated scrubbing pass (`scrub_pii.py`) did not catch.

---

## 4. Disclaimer

The dataset is provided "as is", without warranty of any kind. The authors accept no
liability for any use made of it. Reviews express the opinions of their individual
authors. Inclusion in this corpus is not an endorsement of any statement in them,
nor of any establishment named.
