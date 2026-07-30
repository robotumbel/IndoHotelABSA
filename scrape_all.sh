#!/usr/bin/env bash
set -euo pipefail

API_KEY="${GOOGLE_PLACES_API_KEY:?Set GOOGLE_PLACES_API_KEY environment variable (do not hardcode keys)}"
OUT_DIR="scrape_results"
MAX_PLACES=60

mkdir -p "$OUT_DIR"

# Format: "Nama Provinsi|nama_file"
provinsi=(
  "Aceh|aceh"
  "Sumatera Utara|sumatera_utara"
  "Sumatera Barat|sumatera_barat"
  "Riau|riau"
  "Jambi|jambi"
  "Sumatera Selatan|sumatera_selatan"
  "Bengkulu|bengkulu"
  "Lampung|lampung"
  "Kepulauan Bangka Belitung|bangka_belitung"
  "Kepulauan Riau|kepulauan_riau"
  "DKI Jakarta|dki_jakarta"
  "Jawa Barat|jawa_barat"
  "Jawa Tengah|jawa_tengah"
  "DI Yogyakarta|diy_yogyakarta"
  "Jawa Timur|jawa_timur"
  "Banten|banten"
  "Bali|bali"
  "Nusa Tenggara Barat|ntb"
  "Nusa Tenggara Timur|ntt"
  "Kalimantan Barat|kalimantan_barat"
  "Kalimantan Tengah|kalimantan_tengah"
  "Kalimantan Selatan|kalimantan_selatan"
  "Kalimantan Timur|kalimantan_timur"
  "Kalimantan Utara|kalimantan_utara"
  "Sulawesi Utara|sulawesi_utara"
  "Sulawesi Tengah|sulawesi_tengah"
  "Sulawesi Selatan|sulawesi_selatan"
  "Sulawesi Tenggara|sulawesi_tenggara"
  "Gorontalo|gorontalo"
  "Sulawesi Barat|sulawesi_barat"
  "Maluku|maluku"
  "Maluku Utara|maluku_utara"
  "Papua|papua"
  "Papua Barat|papua_barat"
  "Papua Selatan|papua_selatan"
  "Papua Tengah|papua_tengah"
  "Papua Pegunungan|papua_pegunungan"
  "Papua Barat Daya|papua_barat_daya"
)

for item in "${provinsi[@]}"; do
  nama="${item%%|*}"
  file="${item##*|}"
  out="$OUT_DIR/reviews_${file}.jsonl"

  if [[ -f "$out" ]]; then
    echo "[SKIP] $nama -> $out sudah ada"
    continue
  fi

  echo "[RUN] $nama -> $out"
  python scrape_reviews.py \
    --mode google \
    --api_key "$API_KEY" \
    --query "hotel di $nama" \
    --max_places "$MAX_PLACES" \
    --out "$out"
done

echo "Selesai."
