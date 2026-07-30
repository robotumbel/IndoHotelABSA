# -*- coding: utf-8 -*-
"""
lexicon_baseline.py — Baseline ABSA berbasis kata-kunci + leksikon sentimen.
Batas bawah (lower bound) tanpa pelatihan. Output JSONL utk evaluate_absa.py.
Pemakaian: python lexicon_baseline.py --test test.jsonl --pred_out pred_lexicon.jsonl
"""
import argparse, json, re

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
       "berisik","susah","macet","kemahalan","mati","tutup","terlambat","dingin","kurang",
       "tidak","biasa saja"]

def polarity(text):
    t = text.lower()
    pos = sum(t.count(w) for w in POS)
    neg = sum(t.count(w) for w in NEG)
    if pos > neg: return "POSITIF"
    if neg > pos: return "NEGATIF"
    return "NETRAL"

def predict(text):
    t = text.lower()
    labels = []
    # cari klausa per tanda baca untuk polaritas lokal
    clauses = re.split(r"[.,;!?]", t)
    for asp, kws in ASPECT_KEYWORDS.items():
        hit_clause = None
        for c in clauses:
            if any(k in c for k in kws):
                hit_clause = c; break
        if hit_clause is not None:
            labels.append({"aspect": asp, "polarity": polarity(hit_clause)})
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--pred_out", default="pred_lexicon.jsonl")
    args = ap.parse_args()
    out = []
    with open(args.test, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            out.append({"review_id": r["review_id"], "labels": predict(r["text"])})
    with open(args.pred_out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Prediksi baseline -> {args.pred_out} ({len(out)} review)")

if __name__ == "__main__":
    main()
