# -*- coding: utf-8 -*-
"""
scrub_pii.py — Deterministic in-text PII scrubbing for the IndoHotelABSA release.

Addresses Reviewer 2, Comment 2: metadata-level anonymisation (applied at
collection time) does not remove personal data that reviewers typed into the
review text itself. This pass replaces such spans with typed placeholders.

Targets, in priority order:
  [URL]     explicit http(s):// and www. links
  [EMAIL]   e-mail addresses
  [PHONE]   Indonesian mobile numbers, including spaced/dotted digit groups
            and full-width or styled digit substitutions (folded by NFKC)

Deliberately NOT targeted: @-handles and bare domain names; see the comments
on URL_RE and the @-handle note below for why a rule for each was tested and
rejected as destructive to annotated content.

Person names are deliberately NOT removed; see the datasheet for the rationale
(Indonesian NER over informal code-mixed review text has a high false-positive
rate on hotel, place, and brand names, and destructive substitution would
damage the aspect-bearing spans the dataset exists to support).

Usage:
  python scrub_pii.py --report                 # count only, write nothing
  python scrub_pii.py --infile a.jsonl --out b.jsonl
  python scrub_pii.py --inplace a.jsonl b.jsonl ...
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_FILES = [
    "dataset_merged.jsonl",
    "train_silver.jsonl",
    "val_gold.jsonl",
    "test_gold.jsonl",
    "gold_final.jsonl",
]

# Homoglyph / styled-digit folding: mathematical alphanumerics, full-width
# digits, and the circled/parenthesised forms all normalise to ASCII digits
# under NFKC, which is how listing spam evades a naive \d match.
def fold(text):
    return unicodedata.normalize("NFKC", text)

# Explicit URLs only. A bare-domain rule (e.g. \S+\.com) was tested and
# rejected: in this corpus it matches online travel agency brand names
# (tiket.com, booking.com, trip.com) that are ordinary review content and in
# several cases carry the Price/Value or Service aspect. Redacting them would
# destroy annotated content without removing any personal data.
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Indonesian mobile numbers: 08xxxxxxxxx or +62 8xxxxxxxxx / 62 8xxxxxxxxx,
# 10-13 digits in total. Separators seen in the wild: space, dot, dash,
# middle dot, en/em dash, underscore.
#
# The leading group is deliberately anchored to "08" or "+62 8" rather than a
# bare leading zero: a looser rule matched Indonesian price ranges written with
# dot thousands separators ("000 - 300.000").
SEP = r"[\s.\-·–—_]*"
PHONE_RE = re.compile(
    r"(?<![\w])"
    r"(?:\+?6" + SEP + r"2" + SEP + r"8|08)"
    r"(?:" + SEP + r"\d){7,11}"
    r"(?![\w])"
)

# NOTE ON @-HANDLES: no handle rule is applied. In Indonesian reviews "@" is
# overwhelmingly used as the preposition "at" for unit prices ("@200rb",
# "@Rp.5000"), for room numbers ("@325"), and as informal address to staff by
# first name ("@Mba", "@Faisal"). A blanket @-handle rule matched 34 spans in
# this corpus, of which the large majority were of those three kinds. Social
# handles that remain are covered by the same rationale as in-text person
# names; see the datasheet.


# Obfuscated contact numbers in listing-owner spam. These posts follow a fixed
# template ("... Info kontak untuk booking kamar Wa <number> (Pemilik)") in
# which the digits are written with Yi, Cherokee, and other rare-script
# homoglyphs that NFKC does not fold, so PHONE_RE cannot see them. The rule is
# therefore contextual: after a contact-solicitation cue, a run dominated by
# non-ASCII, non-emoji glyphs is treated as an obfuscated number.
CONTACT_CUE = re.compile(
    r"\b(?:info\s*kontak|booking\s*kamar|chat\s*wa|hubungi|reservasi|pemesanan|wa)\b",
    re.IGNORECASE,
)
# Characters that may appear inside an obfuscated number: rare-script
# homoglyphs, plus the ASCII letters and digits used as digit look-alikes
# (O for 0, l/I for 1) and the usual separators.
OBF_CHAR = re.compile(r"[^\x00-\x7F]|[OolIiSsZzB\d\-.]")


def _is_emoji(ch):
    o = ord(ch)
    return (0x1F000 <= o <= 0x1FAFF) or (0x2600 <= o <= 0x27BF) or (0xFE00 <= o <= 0xFE0F)


def scrub_obfuscated(s, hits):
    """Replace homoglyph-obfuscated contact numbers that follow a contact cue.

    The span is taken from the first to the last non-ASCII glyph inside a
    60-character window after the cue, so that ASCII digit look-alikes sitting
    between the homoglyphs (O for 0, l for 1) are removed with the rest of the
    number rather than left stranded.
    """
    out = s
    for cue in list(CONTACT_CUE.finditer(out)):
        start = cue.end()
        window = out[start:start + 60]
        idx = [i for i, c in enumerate(window)
               if ord(c) > 0x7F and not _is_emoji(c) and not c.isspace()]
        if len(idx) < 6:
            continue
        lo, hi = idx[0], idx[-1] + 1
        # widen to include adjoining ASCII look-alikes and separators
        while lo > 0 and OBF_CHAR.fullmatch(window[lo - 1]):
            lo -= 1
        while hi < len(window) and OBF_CHAR.fullmatch(window[hi]):
            hi += 1
        span = window[lo:hi]
        if span.strip():
            out = out[:start + lo] + "[PHONE]" + out[start + hi:]
            hits["[PHONE]"] += 1
    return out


def scrub(text):
    """Return (scrubbed_text, Counter of placeholder -> hits)."""
    hits = Counter()
    folded = fold(text)

    def sub(pattern, tag, s):
        def _r(m):
            hits[tag] += 1
            return tag
        return pattern.sub(_r, s)

    s = folded
    s = sub(URL_RE, "[URL]", s)
    s = sub(EMAIL_RE, "[EMAIL]", s)
    s = sub(PHONE_RE, "[PHONE]", s)
    s = scrub_obfuscated(s, hits)
    return s, hits


def process(path, out_path=None):
    total = changed = 0
    hits = Counter()
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            total += 1
            new, h = scrub(o.get("text", ""))
            if h:
                changed += 1
                hits.update(h)
                o["text"] = new
                o["pii_scrubbed"] = True
            rows.append(o)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for o in rows:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

    return total, changed, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="count matches across the default release files, write nothing")
    ap.add_argument("--infile")
    ap.add_argument("--out")
    ap.add_argument("--inplace", nargs="*", default=None)
    args = ap.parse_args()

    if args.report or (not args.infile and args.inplace is None):
        grand_t = grand_c = 0
        grand_h = Counter()
        print(f"{'file':<28}{'records':>10}{'affected':>10}  placeholders")
        for p in DEFAULT_FILES:
            try:
                t, c, h = process(p, None)
            except FileNotFoundError:
                print(f"{p:<28}{'--':>10}{'--':>10}  (not found)")
                continue
            grand_t += t
            grand_c += c
            grand_h.update(h)
            detail = ", ".join(f"{k} {v}" for k, v in sorted(h.items())) or "-"
            print(f"{p:<28}{t:>10,}{c:>10,}  {detail}")
        print("-" * 72)
        detail = ", ".join(f"{k} {v}" for k, v in sorted(grand_h.items())) or "-"
        pct = grand_c / grand_t * 100 if grand_t else 0
        print(f"{'TOTAL':<28}{grand_t:>10,}{grand_c:>10,}  {detail}")
        print(f"records affected: {grand_c:,}/{grand_t:,} ({pct:.2f}%)")
        return

    if args.infile:
        t, c, h = process(args.infile, args.out or args.infile)
        print(f"{args.infile}: {c}/{t} records scrubbed; {dict(h)}")
        return

    for p in args.inplace:
        t, c, h = process(p, p)
        print(f"{p}: {c}/{t} records scrubbed; {dict(h)}")


if __name__ == "__main__":
    main()
