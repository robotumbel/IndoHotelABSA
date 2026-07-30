"""
preannotate_gemini.py — PRE-ANOTASI ABSA dengan Gemini (Google).

=====================================================================
  PENTING (integritas):
  LLM hanya mengisi label AWAL untuk MEMPERCEPAT anotator.
  Keputusan final WAJIB oleh anotator manusia (verifikasi/koreksi).
  Setiap review ditandai "llm_prefilled": true agar transparan.
  Di paper, laporkan jujur: "LLM-assisted pre-annotation, human-verified".
=====================================================================

Fitur: RESUME — bila proses terputus, jalankan ulang; review yang sudah
diproses otomatis dilewati (dibaca dari file output).

Input : annotation_sample.jsonl   (field 'labels' kosong)
Output: annotation_prefilled.jsonl (field 'labels' terisi tebakan LLM)

Butuh: pip install google-genai
Set API key:  $env:GEMINI_API_KEY="kunci_anda"   (PowerShell)
Pemakaian:
  python preannotate_gemini.py --infile annotation_sample.jsonl --out annotation_prefilled.jsonl
"""
import argparse
import json
import os
import re
import time

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
POLARITIES = ["POSITIF", "NEGATIF", "NETRAL"]

PROMPT = (
    "Anda anotator Aspect-Based Sentiment Analysis untuk ulasan hotel berbahasa Indonesia.\n"
    "Untuk ulasan di bawah, keluarkan HANYA JSON: daftar objek "
    '{"aspect": <aspek>, "polarity": <sentimen>}.\n'
    f"Aspek valid (persis): {ASPECTS}.\n"
    f"Sentimen valid: {POLARITIES}.\n"
    "Aturan:\n"
    "- Hanya sertakan aspek yang benar-benar disebut.\n"
    "- Perhatikan negasi (mis. 'tidak bersih' = NEGATIF).\n"
    "- 'Kamar & Fasilitas' = fasilitas DALAM kamar; 'Fasilitas Pendukung' = fasilitas UMUM (wifi, parkir, kolam).\n"
    "- Bila faktual tanpa penilaian, polaritas NETRAL.\n"
    "- Jangan beri penjelasan apa pun di luar JSON.\n\n"
    'Ulasan: "{text}"\n'
    "Jawaban (JSON):"
)


def parse_labels(raw):
    """Ambil JSON pertama dari keluaran LLM; validasi label."""
    if not raw:
        return []
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out, seen = [], set()
    for o in arr:
        if not isinstance(o, dict):
            continue
        a, p = o.get("aspect"), o.get("polarity")
        if a in ASPECTS and p in POLARITIES and a not in seen:
            out.append({"aspect": a, "polarity": p})
            seen.add(a)
    return out


def load_done(out_path):
    """review_id yang sudah ada di file output (untuk RESUME)."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["review_id"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="annotation_sample.jsonl")
    ap.add_argument("--out", default="annotation_prefilled.jsonl")
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="model Gemini (ganti bila perlu, mis. gemini-flash-latest)")
    ap.add_argument("--api_key", default=os.environ.get("GEMINI_API_KEY"))
    ap.add_argument("--sleep", type=float, default=4.5,
                    help="jeda antar-panggilan (detik); free-tier ~15 RPM -> pakai >=4.5")
    ap.add_argument("--limit", type=int, default=0, help="batasi N review (0=semua; untuk uji coba)")
    ap.add_argument("--list_models", action="store_true", help="tampilkan model yang tersedia lalu keluar")
    args = ap.parse_args()

    assert args.api_key, "Set GEMINI_API_KEY atau beri --api_key"

    from google import genai
    client = genai.Client(api_key=args.api_key)

    # --- daftar model yang tersedia (diagnostik) ---
    if args.list_models:
        print("Model yang mendukung generateContent:")
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if (not actions) or ("generateContent" in actions):
                print("  ", m.name)
        return

    # --- pilih model: coba yang diminta, lalu fallback otomatis ---
    # flash-lite didahulukan: limit free-tier lebih tinggi (RPM & RPD)
    candidates = [args.model, "gemini-2.0-flash-lite", "gemini-flash-lite-latest",
                  "gemini-2.0-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
    model_name = None
    for cand in dict.fromkeys(candidates):        # buang duplikat, jaga urutan
        try:
            client.models.generate_content(model=cand, contents="ping")
            model_name = cand
            break
        except Exception as e:
            msg = str(e)
            # 429 = model VALID tapi kuota habis -> tetap pakai, biar loop utama yang retry
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print(f"  [model] '{cand}' valid tapi rate-limited (429) -> dipakai.")
                model_name = cand
                break
            print(f"  [model] '{cand}' tidak bisa dipakai ({msg[:50]})")
    assert model_name, "Tidak ada model Gemini yang bisa dipakai. Jalankan --list_models."
    print(f"Memakai model: {model_name}")
    args.model = model_name

    rows = []
    with open(args.infile, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    done = load_done(args.out)
    todo = [r for r in rows if r["review_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Total {len(rows)} | sudah selesai {len(done)} | akan diproses {len(todo)}")

    fout = open(args.out, "a", encoding="utf-8")
    processed = 0
    consecutive_fail = 0                          # circuit breaker limit harian
    for r in todo:
        # pakai replace (bukan .format) karena PROMPT berisi kurung kurawal JSON literal
        prompt = PROMPT.replace("{text}", r["text"])
        labels = None                            # None = API belum berhasil
        for attempt in range(6):                 # retry pada error/429
            try:
                resp = client.models.generate_content(model=args.model, contents=prompt)
                labels = parse_labels(resp.text)
                break
            except Exception as e:
                msg = str(e)
                # 429 = kuota per-menit habis -> tunggu jauh lebih lama
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    wait = 60
                elif "503" in msg or "UNAVAILABLE" in msg:
                    wait = 15
                else:
                    wait = 5 * (attempt + 1)
                if attempt == 5:
                    print(f"  [skip] review {r['review_id']} gagal ({msg[:45]}); "
                          f"TIDAK ditulis, akan dicoba lagi saat run ulang.")
                else:
                    print(f"  [retry {attempt+1}] {msg[:40]} - tunggu {wait}s")
                    time.sleep(wait)
        if labels is None:
            consecutive_fail += 1
            # 12 gagal beruntun -> kemungkinan limit HARIAN tercapai, stop bersih
            if consecutive_fail >= 12:
                print(f"\n[STOP] {consecutive_fail} review gagal beruntun -> "
                      f"kemungkinan limit harian tercapai. Berhenti bersih.")
                print(f"Lanjutkan besok dengan menjalankan perintah yang sama (resume).")
                break
            continue                             # JANGAN tulis review gagal (biar di-retry)
        consecutive_fail = 0                      # reset bila berhasil
        out = dict(r)
        out["labels"] = labels
        out["llm_prefilled"] = True              # transparansi: label dari LLM
        out["verified"] = False                  # anotator set True setelah cek
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        fout.flush()
        processed += 1
        if processed % 50 == 0:
            print(f"  {processed}/{len(todo)} selesai (total {len(done)+processed}/{len(rows)})")
        time.sleep(args.sleep)

    fout.close()
    print(f"\nSelesai. Pre-anotasi -> {args.out}")
    print("LANGKAH WAJIB: anotator manusia verifikasi/koreksi tiap label,")
    print("lalu set 'verified': true. Hitung Kappa dari label FINAL manusia.")


if __name__ == "__main__":
    main()
