# ============================================================
#  scrape_all.ps1 — Kumpul review hotel Indonesia PER-KOTA
#  Query level kota memberi hotel unik jauh lebih banyak
#  daripada level provinsi (maks 60 hotel/query dari Google).
# ============================================================

# >>> GANTI dengan API key BARU Anda (yang lama sudah terekspos di chat, regenerate!) <<<
# Lebih aman: set environment variable, mis:  $env:GOOGLE_PLACES_KEY="kunci_baru"
$ApiKey = $env:GOOGLE_PLACES_KEY
if (-not $ApiKey) { $ApiKey = "API KEY" }

$OutDir = "scrape_results"
$MaxPlaces = 60

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Daftar kota (Nama = query, File = nama file output)
$kota = @(
  # ----- Sumatera -----
  @{ Nama = "Banda Aceh";      File = "banda_aceh" }
  @{ Nama = "Medan";           File = "medan" }
  @{ Nama = "Padang";          File = "padang" }
  @{ Nama = "Bukittinggi";     File = "bukittinggi" }
  @{ Nama = "Pekanbaru";       File = "pekanbaru" }
  @{ Nama = "Batam";           File = "batam" }
  @{ Nama = "Tanjung Pinang";  File = "tanjung_pinang" }
  @{ Nama = "Jambi";           File = "jambi" }
  @{ Nama = "Palembang";       File = "palembang" }
  @{ Nama = "Bengkulu";        File = "bengkulu" }
  @{ Nama = "Bandar Lampung";  File = "bandar_lampung" }
  @{ Nama = "Pangkal Pinang";  File = "pangkal_pinang" }
  # ----- Jawa -----
  @{ Nama = "Jakarta";         File = "jakarta" }
  @{ Nama = "Bogor";           File = "bogor" }
  @{ Nama = "Depok";           File = "depok" }
  @{ Nama = "Tangerang";       File = "tangerang" }
  @{ Nama = "Bekasi";          File = "bekasi" }
  @{ Nama = "Bandung";         File = "bandung" }
  @{ Nama = "Cirebon";         File = "cirebon" }
  @{ Nama = "Sukabumi";        File = "sukabumi" }
  @{ Nama = "Semarang";        File = "semarang" }
  @{ Nama = "Surakarta Solo";  File = "solo" }
  @{ Nama = "Magelang";        File = "magelang" }
  @{ Nama = "Yogyakarta";      File = "yogyakarta" }
  @{ Nama = "Surabaya";        File = "surabaya" }
  @{ Nama = "Malang";          File = "malang" }
  @{ Nama = "Batu Malang";     File = "batu" }
  @{ Nama = "Banyuwangi";      File = "banyuwangi" }
  @{ Nama = "Serang";          File = "serang" }
  # ----- Bali & Nusa Tenggara -----
  @{ Nama = "Denpasar";        File = "denpasar" }
  @{ Nama = "Ubud";            File = "ubud" }
  @{ Nama = "Kuta Bali";       File = "kuta" }
  @{ Nama = "Seminyak";        File = "seminyak" }
  @{ Nama = "Nusa Dua Bali";   File = "nusa_dua" }
  @{ Nama = "Mataram Lombok";  File = "mataram" }
  @{ Nama = "Senggigi Lombok"; File = "senggigi" }
  @{ Nama = "Labuan Bajo";     File = "labuan_bajo" }
  @{ Nama = "Kupang";          File = "kupang" }
  # ----- Kalimantan -----
  @{ Nama = "Pontianak";       File = "pontianak" }
  @{ Nama = "Palangkaraya";    File = "palangkaraya" }
  @{ Nama = "Banjarmasin";     File = "banjarmasin" }
  @{ Nama = "Balikpapan";      File = "balikpapan" }
  @{ Nama = "Samarinda";       File = "samarinda" }
  @{ Nama = "Tarakan";         File = "tarakan" }
  # ----- Sulawesi -----
  @{ Nama = "Manado";          File = "manado" }
  @{ Nama = "Palu";            File = "palu" }
  @{ Nama = "Makassar";        File = "makassar" }
  @{ Nama = "Kendari";         File = "kendari" }
  @{ Nama = "Gorontalo";       File = "gorontalo" }
  @{ Nama = "Mamuju";          File = "mamuju" }
  # ----- Maluku & Papua -----
  @{ Nama = "Ambon";           File = "ambon" }
  @{ Nama = "Ternate";         File = "ternate" }
  @{ Nama = "Jayapura";        File = "jayapura" }
  @{ Nama = "Sorong";          File = "sorong" }
  @{ Nama = "Manokwari";       File = "manokwari" }
)

if ($ApiKey -eq "PASTE_KUNCI_BARU_DISINI") {
  Write-Host "ERROR: API key belum diisi. Edit `$ApiKey di baris atas, atau set:" -ForegroundColor Red
  Write-Host '  $env:GOOGLE_PLACES_KEY="kunci_baru_anda"' -ForegroundColor Yellow
  exit 1
}

$totalReview = 0
$i = 0
foreach ($k in $kota) {
  $i++
  $out = Join-Path $OutDir ("reviews_{0}.jsonl" -f $k.File)

  if (Test-Path $out) {
    Write-Host "[$i/$($kota.Count)] [SKIP] $($k.Nama) -> sudah ada"
    continue
  }

  Write-Host "[$i/$($kota.Count)] [RUN] $($k.Nama) -> $out"
  python scrape_reviews.py --mode google --api_key $ApiKey --query "hotel di $($k.Nama)" --kota "$($k.Nama)" --expand --max_places $MaxPlaces --out $out

  if (Test-Path $out) {
    $n = (Get-Content $out | Measure-Object -Line).Lines
    $totalReview += $n
  }
  Start-Sleep -Seconds 1   # jeda sopan antar-kota
}

Write-Host ""
Write-Host "Selesai. Perkiraan total review terkumpul (sesi ini): $totalReview" -ForegroundColor Green
Write-Host "Langkah berikut: gabung semua file jsonl di '$OutDir' -> dedup -> anotasi."
