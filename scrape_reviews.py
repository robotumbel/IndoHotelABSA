"""
scrape_reviews.py — Pengumpul review hotel berbahasa Indonesia untuk IndoHotelABSA.

=====================================================================
  PERHATIAN ETIKA & LEGAL (WAJIB DIBACA SEBELUM MENJALANKAN)
=====================================================================
  1. Patuhi Terms of Service tiap platform. Banyak platform melarang
     scraping otomatis. Cara PALING AMAN & LEGAL: pakai API resmi
     (mis. Google Places API) seperti pada MODE A di bawah.
  2. Hormati robots.txt dan pasang rate-limit (jeda antar-permintaan).
  3. ANONIMISASI: buang nama reviewer / data pribadi. Simpan hanya teks
     ulasan + metadata non-pribadi (rating, tanggal, nama hotel).
  4. Untuk publikasi dataset: sebutkan sumber, patuhi lisensi, dan
     pertimbangkan meminta izin bila diperlukan oleh platform.
=====================================================================

MODE A (disarankan): Google Places API — legal, resmi.
    Perlu API key: https://developers.google.com/maps/documentation/places/web-service
    Catatan: endpoint Place Details hanya mengembalikan hingga 5 review
    per tempat, tetapi legal dan stabil. Kumpulkan dari banyak hotel.

MODE B (template umum): scraping HTML dengan requests+BeautifulSoup.
    Hanya untuk platform yang MENGIZINKAN. Sesuaikan selector-nya.
    Gunakan dengan tanggung jawab penuh.

Output: reviews_raw.jsonl  (satu review per baris)
    {"review_id": 1, "hotel": "...", "rating": 4, "text": "...", "source": "..."}
"""
import argparse
import json
import time
import re

# ---------------------------------------------------------------
# Util umum
# ---------------------------------------------------------------
def clean_text(t: str) -> str:
    """Bersihkan whitespace berlebih; pertahankan isi Bahasa Indonesia."""
    t = re.sub(r"\s+", " ", t).strip()
    return t

def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            r.setdefault("review_id", i)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Tersimpan {len(rows)} review -> {path}")


# ---------------------------------------------------------------
# MODE A — Google Places API (resmi/legal)
# ---------------------------------------------------------------
TEXT_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAIL_URL = "https://maps.googleapis.com/maps/api/place/details/json"


def _get_json(url, params, retries=4, timeout=25):
    """HTTP GET tahan-gangguan: retry dengan backoff pada error jaringan."""
    import requests
    for attempt in range(retries):
        try:
            return requests.get(url, params=params, timeout=timeout).json()
        except Exception as e:
            wait = 3 * (attempt + 1)
            if attempt == retries - 1:
                print(f"  [warn] request gagal ({e}); dilewati.")
                return {}
            print(f"  [retry {attempt+1}/{retries}] jaringan bermasalah, tunggu {wait}s...")
            time.sleep(wait)
    return {}

# Variasi query untuk menambah jumlah hotel unik per kota.
# Text Search legacy hanya balas 20/query, jadi variasi = cara paling andal
# menambah cakupan tanpa bergantung pada next_page_token.
QUERY_VARIANTS = [
    "hotel di {kota}",
    "hotel murah di {kota}",
    "hotel bintang 5 di {kota}",
    "hotel bintang 3 di {kota}",
    "penginapan di {kota}",
    "guest house di {kota}",
    "resort di {kota}",
    "hotel dekat pusat kota {kota}",
]


def _search_places(api_key, query):
    """Satu query Text Search -> list place (coba pagination bila tersedia)."""
    def fetch_page(p, is_token):
        for attempt in range(5):
            resp = _get_json(TEXT_URL, p)
            st = resp.get("status")
            if st == "OK":
                return resp
            if st == "INVALID_REQUEST" and is_token:
                time.sleep(2.0 + attempt)
                continue
            return resp
        return resp
    places, params, is_token = [], {"query": query, "language": "id", "key": api_key}, False
    for _ in range(3):
        resp = fetch_page(params, is_token)
        places.extend(resp.get("results", []))
        token = resp.get("next_page_token")
        if not token:
            break
        params = {"pagetoken": token, "key": api_key}
        is_token = True
    return places


def collect_google_places(api_key, query, max_places=20, pause=1.0, expand=False, kota=None):
    """
    Kumpulkan review dari Google Places API.
    Bila expand=True, jalankan beberapa variasi query (QUERY_VARIANTS) dan
    gabungkan hotel unik (dedup by place_id) untuk cakupan jauh lebih luas.
    """
    import requests

    # 1) Kumpulkan tempat unik (dedup by place_id)
    if expand and kota:
        queries = [v.format(kota=kota) for v in QUERY_VARIANTS]
    else:
        queries = [query]

    uniq = {}                                    # place_id -> place
    for q in queries:
        for pl in _search_places(api_key, q):
            pid = pl.get("place_id")
            if pid and pid not in uniq:
                uniq[pid] = pl
        if len(uniq) >= max_places:
            break
        time.sleep(0.5)                          # jeda antar variasi query
    places = list(uniq.values())[:max_places]
    print(f"Ditemukan {len(places)} hotel unik "
          f"({'expand' if expand else 'single'} query untuk: {query})")

    # 2) Ambil review tiap hotel (maks 5 per tempat, batasan API)
    rows = []
    for pl in places:
        pid = pl["place_id"]
        hotel = pl.get("name", "")
        time.sleep(pause)  # rate-limit
        d = _get_json(DETAIL_URL, {
            "place_id": pid, "fields": "name,review",
            "language": "id", "reviews_no_translations": "true", "key": api_key
        })
        for rv in d.get("result", {}).get("reviews", []):
            txt = clean_text(rv.get("text", ""))
            if len(txt) < 15:          # buang review terlalu pendek
                continue
            rows.append({
                "hotel": hotel,
                "rating": rv.get("rating"),
                "text": txt,             # ANONIM: nama reviewer tidak disimpan
                "source": "google_places",
            })
    return rows


# ---------------------------------------------------------------
# MODE B — Template scraping HTML (sesuaikan & pakai bertanggung jawab)
# ---------------------------------------------------------------
def collect_generic_html(urls, pause=2.0):
    """
    Template scraping halaman review. HANYA untuk situs yang mengizinkan.
    Sesuaikan CSS selector ke struktur situs target.
    Butuh: pip install requests beautifulsoup4
    """
    import requests
    from bs4 import BeautifulSoup

    HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact: you@univ.ac.id)"}
    rows = []
    for url in urls:
        time.sleep(pause)  # rate-limit sopan
        try:
            html = requests.get(url, headers=HEADERS, timeout=20).text
        except Exception as e:
            print(f"Gagal {url}: {e}")
            continue
        soup = BeautifulSoup(html, "html.parser")

        # >>> SESUAIKAN SELECTOR INI ke struktur situs target <<<
        #     Contoh generik (ganti sesuai inspeksi elemen situs):
        for card in soup.select("div.review-card"):
            text_el = card.select_one(".review-text")
            rating_el = card.select_one(".review-rating")
            if not text_el:
                continue
            txt = clean_text(text_el.get_text())
            if len(txt) < 15:
                continue
            rows.append({
                "hotel": "",              # isi jika tersedia
                "rating": rating_el.get_text(strip=True) if rating_el else None,
                "text": txt,
                "source": url,
            })
    return rows


# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["google", "html"], required=True)
    ap.add_argument("--out", default="reviews_raw.jsonl")
    ap.add_argument("--api_key", help="Google Places API key (mode google)")
    ap.add_argument("--query", default="hotel di Indonesia",
                    help="query pencarian (mode google)")
    ap.add_argument("--max_places", type=int, default=20)
    ap.add_argument("--expand", action="store_true",
                    help="jalankan beberapa variasi query per kota (hotel unik jauh lebih banyak)")
    ap.add_argument("--kota", help="nama kota untuk variasi query (dipakai bila --expand)")
    ap.add_argument("--urls_file", help="file .txt berisi daftar URL (mode html)")
    args = ap.parse_args()

    if args.mode == "google":
        assert args.api_key, "Mode google butuh --api_key"
        rows = collect_google_places(args.api_key, args.query, args.max_places,
                                     expand=args.expand, kota=args.kota)
    else:
        assert args.urls_file, "Mode html butuh --urls_file"
        with open(args.urls_file, encoding="utf-8") as f:
            urls = [u.strip() for u in f if u.strip()]
        rows = collect_generic_html(urls)

    # dedup sederhana berdasarkan teks
    seen, uniq = set(), []
    for r in rows:
        key = r["text"][:120]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    save_jsonl(uniq, args.out)
    print("Selanjutnya: anotasi manual (lihat Panduan Anotasi) menjadi format "
          "gold JSONL dengan field 'labels'.")


if __name__ == "__main__":
    main()
