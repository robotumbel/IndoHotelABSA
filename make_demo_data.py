# -*- coding: utf-8 -*-
"""
make_demo_data.py — Buat dataset DEMO kecil (review hotel Indonesia beranotasi)
untuk menguji pipeline end-to-end. BUKAN dataset riset; hanya bukti kode jalan.
Menulis train.jsonl / val.jsonl / test.jsonl.
"""
import json, random

# (teks, [(aspek, polaritas), ...]) — dibuat tangan, realistis
DATA = [
    ("Lokasinya strategis dekat Malioboro, tapi kamar mandinya bau dan AC tidak dingin.",
     [("Lokasi","POSITIF"),("Kebersihan","NEGATIF"),("Kamar & Fasilitas","NEGATIF")]),
    ("Harga terjangkau, sarapannya enak dan variatif. Recommended!",
     [("Harga","POSITIF"),("Makanan & Minuman","POSITIF")]),
    ("WiFi lemot banget, parkir juga sempit. Untung stafnya ramah.",
     [("Fasilitas Pendukung","NEGATIF"),("Pelayanan","POSITIF")]),
    ("Kamar luas dan bersih, kasur empuk, betah banget.",
     [("Kamar & Fasilitas","POSITIF"),("Kebersihan","POSITIF")]),
    ("Pelayanan lambat saat check-in, antri lama sekali.",
     [("Pelayanan","NEGATIF")]),
    ("Kolam renangnya bagus, anak-anak senang. WiFi kencang.",
     [("Fasilitas Pendukung","POSITIF")]),
    ("Sarapan hambar dan pilihan menunya sedikit.",
     [("Makanan & Minuman","NEGATIF")]),
    ("Hotel dekat stasiun, sangat memudahkan perjalanan.",
     [("Lokasi","POSITIF")]),
    ("Harga mahal untuk fasilitas yang biasa saja.",
     [("Harga","NEGATIF")]),
    ("Kamar kotor, ada rambut di seprai. Kecewa.",
     [("Kebersihan","NEGATIF")]),
    ("Staf sangat ramah dan membantu, pelayanan cepat.",
     [("Pelayanan","POSITIF")]),
    ("Lokasi agak jauh dari pusat kota dan susah cari taksi.",
     [("Lokasi","NEGATIF")]),
    ("Kamarnya nyaman, AC dingin, TV berfungsi baik.",
     [("Kamar & Fasilitas","POSITIF")]),
    ("Sarapan enak, kopinya juga mantap.",
     [("Makanan & Minuman","POSITIF")]),
    ("Parkir luas dan gratis, lift cepat.",
     [("Fasilitas Pendukung","POSITIF")]),
    ("Harga sesuai dengan kualitas, worth it.",
     [("Harga","POSITIF")]),
    ("Kamar mandi bersih tapi air panas tidak keluar.",
     [("Kebersihan","POSITIF"),("Kamar & Fasilitas","NEGATIF")]),
    ("Resepsionis cuek dan tidak informatif.",
     [("Pelayanan","NEGATIF")]),
    ("Dekat pantai, pemandangan indah dari kamar.",
     [("Lokasi","POSITIF"),("Kamar & Fasilitas","POSITIF")]),
    ("WiFi mati total sepanjang menginap.",
     [("Fasilitas Pendukung","NEGATIF")]),
    ("Menginap dua malam, kamar standar di lantai tiga.",
     [("Kamar & Fasilitas","NETRAL")]),
    ("Check-in jam dua siang sesuai aturan.",
     [("Pelayanan","NETRAL")]),
    ("Makanannya biasa saja, tidak istimewa tapi cukup.",
     [("Makanan & Minuman","NETRAL")]),
    ("Kamar sempit tapi bersih dan rapi.",
     [("Kamar & Fasilitas","NEGATIF"),("Kebersihan","POSITIF")]),
    ("Harga promo murah sekali, puas banget.",
     [("Harga","POSITIF")]),
    ("Staf membantu bawa koper, sangat sopan.",
     [("Pelayanan","POSITIF")]),
    ("Lokasi di tengah kota, dekat mall dan kuliner.",
     [("Lokasi","POSITIF")]),
    ("Kolam renang kotor dan gym tutup.",
     [("Fasilitas Pendukung","NEGATIF")]),
    ("Sarapan prasmanan lengkap dan lezat.",
     [("Makanan & Minuman","POSITIF")]),
    ("Kamar bau rokok padahal pesan non-smoking.",
     [("Kebersihan","NEGATIF"),("Kamar & Fasilitas","NEGATIF")]),
    ("Pelayanan ramah, check-out cepat tanpa ribet.",
     [("Pelayanan","POSITIF")]),
    ("Harga naik saat musim liburan, agak mahal.",
     [("Harga","NEGATIF")]),
    ("View kamar menghadap tembok, kurang menarik.",
     [("Kamar & Fasilitas","NEGATIF")]),
    ("WiFi cepat, cocok untuk kerja remote.",
     [("Fasilitas Pendukung","POSITIF")]),
    ("Lokasi susah dijangkau, jalanan macet parah.",
     [("Lokasi","NEGATIF")]),
    ("Kamar bersih wangi, handuk lembut.",
     [("Kebersihan","POSITIF"),("Kamar & Fasilitas","POSITIF")]),
    ("Restoran hotel tutup lebih awal, kecewa.",
     [("Makanan & Minuman","NEGATIF")]),
    ("Staf resepsionis senyum ramah menyambut tamu.",
     [("Pelayanan","POSITIF")]),
    ("Harga standar, tidak murah tidak mahal.",
     [("Harga","NETRAL")]),
    ("Fasilitas lengkap: kolam, gym, spa tersedia.",
     [("Fasilitas Pendukung","POSITIF")]),
    ("Kamar pengap dan AC berisik sepanjang malam.",
     [("Kamar & Fasilitas","NEGATIF")]),
    ("Lokasi dekat bandara, praktis untuk transit.",
     [("Lokasi","POSITIF")]),
    ("Sarapan dingin dan datang terlambat.",
     [("Makanan & Minuman","NEGATIF"),("Pelayanan","NEGATIF")]),
    ("Kebersihan terjaga, kamar disinfeksi rapi.",
     [("Kebersihan","POSITIF")]),
    ("Harga kemahalan tapi pelayanan memuaskan.",
     [("Harga","NEGATIF"),("Pelayanan","POSITIF")]),
    ("Parkir sempit susah dapat tempat.",
     [("Fasilitas Pendukung","NEGATIF")]),
    ("Kamar nyaman, tenang, cocok untuk istirahat.",
     [("Kamar & Fasilitas","POSITIF")]),
    ("Lokasi strategis tapi kamar kurang bersih.",
     [("Lokasi","POSITIF"),("Kebersihan","NEGATIF")]),
]

def to_obj(i, text, pairs):
    return {"review_id": i, "text": text,
            "labels": [{"aspect": a, "polarity": p} for a, p in pairs]}

random.seed(42)
rows = [to_obj(i+1, t, p) for i, (t, p) in enumerate(DATA)]
random.shuffle(rows)

n = len(rows)
n_tr = int(n*0.7); n_va = int(n*0.15)
train, val, test = rows[:n_tr], rows[n_tr:n_tr+n_va], rows[n_tr+n_va:]

for name, part in [("train", train), ("val", val), ("test", test)]:
    with open(f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}.jsonl : {len(part)} review")
print(f"Total: {n} review demo")
