# -*- coding: utf-8 -*-
"""
remove_records.py — action a takedown request against the IndoHotelABSA release.

`LICENSE_DATA.md` commits the maintainers to removing a record, and every
annotation derived from it, within 30 days of a request from Google or from the
author of a review. This script does that removal consistently across every file
so the release stays internally coherent and `verify_release.py` still describes
it correctly.

Usage
-----
Dry run first (nothing is written; shows exactly what would go):

    python remove_records.py --ids 1444 2087
    python remove_records.py --ids-file takedown_2026-09-01.txt

Then apply:

    python remove_records.py --ids 1444 2087 --apply

Finding the id from the review text, if the requester quoted it rather than
giving an id:

    python remove_records.py --find "AC gak dingin"

What it does
------------
* drops the record from every `.jsonl` file that contains it, including
  `dataset_merged.jsonl`, the split files, the per-annotator files, and
  `annotation_prefilled_gold500.jsonl`;
* drops every `adjudication_record.jsonl` cell belonging to it;
* drops the id from `gold_subset_ids.txt`;
* translates between the two id spaces automatically: ids given on the command
  line are **corpus** ids (`dataset_merged.jsonl`), and the labelled files, which
  were renumbered 1..3000 when the annotation sample was drawn, are filtered via
  `sample_to_corpus_ids.tsv`;
* writes `TAKEDOWN_LOG.md` recording the date, the ids, and the resulting counts,
  which is what you cite in the Mendeley version history.

It deliberately does **not** renumber anything. `review_id` values stay stable, so
a removed id simply becomes absent — anyone comparing against the published article
can see precisely what was withdrawn.

After applying, re-run `verify_release.py`. Counts will no longer match the numbers
printed in the article; that is expected and correct after a takedown, and the
TAKEDOWN_LOG explains the difference.
"""
import argparse
import datetime
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

JSONL_FILES = [
    "dataset_merged.jsonl",
    "train_silver.jsonl",
    "val_gold.jsonl",
    "test_gold.jsonl",
    "gold_final.jsonl",
    "excluded_empty_labels.jsonl",
    "annotation_prefilled_gold500.jsonl",
    "annotator_1_verified.jsonl",
    "annotator_2_verified.jsonl",
    "annotator_3_verified.jsonl",
]
CELL_FILE = "adjudication_record.jsonl"
IDS_FILE = "gold_subset_ids.txt"
MAP_FILE = "sample_to_corpus_ids.tsv"
LOG = "TAKEDOWN_LOG.md"


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find(needle):
    needle = needle.lower()
    hits = []
    for path in ("dataset_merged.jsonl",):
        if not os.path.exists(path):
            continue
        for r in read_jsonl(path):
            if needle in (r.get("text") or "").lower():
                hits.append((r["review_id"], r.get("hotel", ""), r.get("city", ""),
                             (r.get("text") or "")[:120]))
    if not hits:
        print(f"No review contains {needle!r}.")
        return
    print(f"{len(hits)} matching review(s):\n")
    for rid, hotel, city, snippet in hits[:25]:
        print(f"  review_id={rid}  [{hotel} / {city}]")
        print(f"    {snippet}...\n")
    if len(hits) > 25:
        print(f"  ... and {len(hits) - 25} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=[],
                    help="corpus review_id values (as in dataset_merged.jsonl)")
    ap.add_argument("--ids-file", help="file with one review_id per line")
    ap.add_argument("--find", help="locate a review_id by a substring of its text")
    ap.add_argument("--apply", action="store_true",
                    help="actually write the files (default is a dry run)")
    ap.add_argument("--requester", default="(not recorded)",
                    help="who asked, for the takedown log")
    args = ap.parse_args()

    if args.find:
        find(args.find)
        return 0

    ids = set(args.ids)
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.add(int(line))
    if not ids:
        ap.error("give --ids, --ids-file, or --find")

    mode = "APPLY" if args.apply else "DRY RUN (nothing will be written)"
    print(f"== Takedown, {mode} ==")
    print(f"removing {len(ids)} corpus review_id(s): {sorted(ids)}")

    # The labelled files use their own id space: the annotation sample was
    # renumbered 1..3000 when it was drawn, so corpus id 1444 and labelled id 1444
    # are different reviews. Translate before filtering those files.
    sample_ids = set()
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("sample_"):
                    continue
                sid, cid = line.split("\t")
                if int(cid) in ids:
                    sample_ids.add(int(sid))
        if sample_ids:
            print(f"  -> {len(sample_ids)} of these are annotated; "
                  f"labelled review_id(s): {sorted(sample_ids)}")
        else:
            print("  -> none of these appear in the annotated sample")
    else:
        print(f"  WARNING: {MAP_FILE} not found. Labelled files can still be "
              f"filtered via corpus_review_id, but the gold id list cannot.")
    print()

    summary = []
    for path in JSONL_FILES:
        if not os.path.exists(path):
            continue
        rows = read_jsonl(path)
        drop = ids if path == "dataset_merged.jsonl" else sample_ids
        keep = [r for r in rows
                if r.get("review_id") not in drop
                and r.get("corpus_review_id") not in ids]
        gone = len(rows) - len(keep)
        summary.append((path, len(rows), len(keep), gone))
        print(f"  {path:<38} {len(rows):>7,} -> {len(keep):>7,}  ({gone} removed)")
        if args.apply and gone:
            write_jsonl(path, keep)

    if os.path.exists(CELL_FILE):
        rows = read_jsonl(CELL_FILE)
        keep = [r for r in rows
                if r.get("review_id") not in sample_ids
                and r.get("corpus_review_id") not in ids]
        gone = len(rows) - len(keep)
        summary.append((CELL_FILE, len(rows), len(keep), gone))
        print(f"  {CELL_FILE:<38} {len(rows):>7,} -> {len(keep):>7,}  ({gone} cells removed)")
        if args.apply and gone:
            write_jsonl(CELL_FILE, keep)

    if os.path.exists(IDS_FILE):
        with open(IDS_FILE, encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
        head = [l for l in lines if l.startswith("#")]
        body = [l for l in lines if l.strip() and not l.startswith("#")]
        keep = [l for l in body if int(l) not in sample_ids]
        gone = len(body) - len(keep)
        summary.append((IDS_FILE, len(body), len(keep), gone))
        print(f"  {IDS_FILE:<38} {len(body):>7,} -> {len(keep):>7,}  ({gone} removed)")
        if args.apply and gone:
            with open(IDS_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(head + keep) + "\n")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write the files.")
        return 0

    today = datetime.date.today().isoformat()
    entry = [f"\n## {today}\n",
             f"- **Requested by:** {args.requester}",
             f"- **Records removed:** {len(ids)} (`review_id` "
             + ", ".join(str(i) for i in sorted(ids)) + ")",
             "- **Effect on the release:**", ""]
    entry.append("| File | Before | After | Removed |")
    entry.append("|---|---:|---:|---:|")
    for path, before, after, gone in summary:
        entry.append(f"| `{path}` | {before:,} | {after:,} | {gone} |")
    entry.append("")
    entry.append("Counts in the associated Data in Brief article describe the release "
                 "as published and are not retrospectively amended; this log records "
                 "the difference.")

    header = ("# Takedown log — IndoHotelABSA\n\n"
              "Records removed under the takedown procedure in `LICENSE_DATA.md`.\n"
              "Each entry corresponds to a version of the Mendeley Data record.\n")
    existing = ""
    if os.path.exists(LOG):
        existing = open(LOG, encoding="utf-8").read()
        if existing.startswith("# Takedown log"):
            header = ""
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(header + existing + "\n".join(entry) + "\n")

    print(f"\nApplied. Wrote {LOG}.")
    print("Next: re-run verify_release.py, then upload a new Mendeley Data version")
    print("citing the takedown log in the version notes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
