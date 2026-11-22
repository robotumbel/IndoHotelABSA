# -*- coding: utf-8 -*-
"""
lexicon_v2.py -- corrected sentiment lexicon for IndoHotelABSA (R2, reviewer 2 point 3).

Fixes relative to lexicon v1 (lexicon_baseline.py, 10 July 2026, still used for all
numbers reported in the paper's main tables):
  1. "dingin" removed from the negative list. Its polarity depends on the aspect
     ("AC dingin" is positive, "makanan dingin" and "air dingin" negative); a word-level
     entry cannot encode that, so it is no longer a cue.
  2. "tidak" removed from the negative list. It is a negation marker, not a sentiment
     word. Negators ("tidak", "tak", "bukan", "belum", "kurang", "gak", "ga", "nggak")
     now flip the polarity of the next sentiment word within a window of two tokens
     ("tidak bersih" -> negative; "tidak bau" -> positive).
  3. Matching stays substring-based, as in v1 (Indonesian suffixes: "kamarnya",
     "bersihnya"); whole-word matching was tried and lowers aspect detection.
  4. "kurang" is a negator, not a standalone negative word ("kurang bersih" -> negative).
  5. "biasa saja" (neutral/mediocre) is a neutral cue and no longer counts as negative.
The aspect cue words are unchanged, so aspect detection is identical to v1.

Remaining limitation (documented, not fixed): this is a word-list, so it cannot resolve
sarcasm, aspect-dependent polarity ("kecil", "murah" for a room vs. food) or scope beyond
the clause. The polarity signal is computed per clause containing an aspect cue (not per
review) in this baseline, but the LGF component of HI-ABSA in v1 used review-level
counts; LGF is therefore deprecated (see paper, Section 4.3).
"""
import argparse, json, re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ASPECT_KEYWORDS = {
    "Lokasi": ["lokasi","dekat","jauh","strategis","stasiun","bandara","pusat kota",
               "pantai","malioboro","mall","macet","jalan","akses","transit"],
    "Kebersihan": ["bersih","kotor","bau","rambut","seprai","wangi","disinfeksi",
                   "higienis","debu","rokok","kebersihan"],
    "Pelayanan": ["staf","pelayanan","resepsionis","ramah","cuek","check-in","check-out",
                  "sopan","lambat","cepat","membantu","senyum"],
    "Kamar & Fasilitas": ["kamar","ac","tv","kasur","air panas","view","luas","sempit",
                          "pengap","nyaman","tempat tidur","handuk"],
    "Harga": ["harga","murah","mahal","terjangkau","worth","promo","kemahalan","value"],
    "Makanan & Minuman": ["sarapan","makan","menu","kopi","restoran","prasmanan","rasa",
                          "hambar","lezat","enak","minuman"],
    "Fasilitas Pendukung": ["wifi","parkir","kolam","gym","lift","spa","fasilitas","lobby"],
}
POS = ["bagus","enak","ramah","bersih","nyaman","strategis","cepat","murah","terjangkau",
       "puas","lezat","mantap","luas","empuk","lengkap","indah","worth","senang","membantu",
       "sopan","wangi","betah","lembut","kencang","praktis","memuaskan","recommended"]
NEG = ["kotor","bau","lambat","mahal","sempit","lemot","cuek","kecewa","hambar","pengap",
       "berisik","susah","macet","kemahalan","mati","tutup","terlambat"]
NEGATORS = {"tidak","tak","bukan","belum","kurang","gak","ga","nggak","enggak"}
NEUTRAL_PHRASES = ["biasa saja"]
_tok = re.compile(r"[a-z0-9\-]+")


def _find(tokens, words):
    """indices of sentiment words (single tokens only; phrases are matched separately)"""
    S = set(words)
    return [i for i, t in enumerate(tokens) if t in S]


def polarity(text):
    t = text.lower()
    tokens = _tok.findall(t)
    pos = neg = 0
    for i, tok in enumerate(tokens):
        p = any(w in tok for w in POS)
        n = any(w in tok for w in NEG)
        if not (p or n):
            continue
        flip = any(tokens[j] in NEGATORS for j in range(max(0, i - 2), i))
        if p != flip:   # positive unflipped or negative flipped
            pos += 1
        else:
            neg += 1
    if pos > neg: return "POSITIF"
    if neg > pos: return "NEGATIF"
    return "NETRAL"


def _has(clause, kw):
    return kw in clause


def predict(text):
    t = text.lower()
    labels = []
    clauses = re.split(r"[.,;!?]", t)
    for asp, kws in ASPECT_KEYWORDS.items():
        hit = None
        for c in clauses:
            if any(_has(c, k) for k in kws):
                hit = c; break
        if hit is not None:
            labels.append({"aspect": asp, "polarity": polarity(hit)})
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--pred_out", default="pred_lexicon_v2.jsonl")
    a = ap.parse_args()
    out = []
    with open(a.test, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append({"review_id": r["review_id"], "labels": predict(r["text"])})
    with open(a.pred_out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"lexicon v2 predictions -> {a.pred_out} ({len(out)} reviews)")


if __name__ == "__main__":
    main()
