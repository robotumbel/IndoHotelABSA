# -*- coding: utf-8 -*-
"""
build_blind_apps.py — build the BLINDED re-annotation apps.

Reviewer 1 observed that the gold subset was verified from a visible LLM draft,
so the reported Fleiss' kappa may not measure independent annotation. This script
builds the check: a 100-review random sub-sample of the gold set with **all labels
cleared**, presented to the same three annotators with no model suggestion of any
kind, so that a blinded kappa can be computed and compared with the verification
kappa on exactly the same reviews.

Differences from build_annotator_apps.py (the original verification tool):
  * every aspect starts at "tidak disebut" (N/A) — nothing is pre-filled;
  * the interface never mentions AI, a draft, or a suggested label;
  * a separate localStorage key, so earlier verification progress cannot leak in;
  * exports to anotator_{N}_blind.jsonl, so files cannot be confused.

Run from the project root:
    python submission_DataInBrief/revision_R1/build_blind_apps.py

Outputs into revision_R1/blind_check/:
    Anotator-{1,2,3}-Blind.zip     one self-contained HTML app each
    blind_subset_ids.txt           the 100 review_ids drawn
    blind_input.jsonl              the cleared records handed to annotators
"""
import json
import os
import random
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "blind_check")
GOLD = "gold_final.jsonl"
N = 100
SEED = 42
ANNOTATORS = ["1", "2", "3"]

HTML = r"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anotasi ABSA Hotel - Anotator __AID__</title>
<style>
:root{--navy:#143a5c;--pos:#1b5e20;--neg:#b02a1f;--neu:#8a6d00;}
*{box-sizing:border-box;} body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#1a1a1a;}
header{background:var(--navy);color:#fff;padding:12px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:10;}
header h1{font-size:15px;margin:0;} .sp{flex:1;}
button{cursor:pointer;border:none;border-radius:6px;padding:8px 14px;font-size:14px;}
.btn{background:#2d6cb4;color:#fff;} .btn.ghost{background:#e4e9ef;color:#143a5c;} .btn:disabled{opacity:.4;cursor:default;}
#wrap{max-width:820px;margin:18px auto;padding:0 16px;}
.card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:20px;}
.meta{font-size:13px;color:#555;margin-bottom:8px;} .meta b{color:#143a5c;}
.review{font-size:16px;line-height:1.6;background:#fbf7ee;border-left:4px solid #d8b24a;padding:14px 16px;border-radius:6px;margin-bottom:16px;white-space:pre-wrap;}
table{width:100%;border-collapse:collapse;} th,td{padding:7px 6px;text-align:left;} th{font-size:12px;color:#555;border-bottom:2px solid #eee;}
.aspect{font-weight:600;} .pills{display:flex;gap:6px;flex-wrap:wrap;}
.pill{padding:5px 10px;border-radius:16px;border:2px solid #ccc;background:#fff;font-size:13px;user-select:none;}
.pill.sel-NA{border-color:#999;background:#eee;color:#555;}
.pill.sel-POSITIF{border-color:var(--pos);background:#e6f4ea;color:var(--pos);font-weight:700;}
.pill.sel-NEGATIF{border-color:var(--neg);background:#fbeae8;color:var(--neg);font-weight:700;}
.pill.sel-NETRAL{border-color:var(--neu);background:#fdf6e0;color:var(--neu);font-weight:700;}
.nav{display:flex;gap:10px;align-items:center;margin-top:16px;flex-wrap:wrap;}
.prog{font-size:13px;} .hint{font-size:12px;color:#777;margin-top:6px;}
.badge{font-size:12px;padding:3px 8px;border-radius:10px;background:#eef;}
.done{background:#e6f4ea;color:var(--pos);}
.intro{background:#fff;border-radius:10px;padding:22px;box-shadow:0 1px 4px rgba(0,0,0,.08);}
.intro h2{margin-top:0;color:var(--navy);}
.note{background:#fff8e6;border:1px solid #e0c169;border-radius:6px;padding:12px 14px;margin:14px 0;font-size:14px;}
</style></head><body>
<header>
  <h1>Anotasi ABSA Hotel &mdash; Anotator __AID__</h1>
  <span class="sp"></span>
  <span class="prog" id="prog"></span>
  <button class="btn ghost" id="exportBtn">&#128190; Export Hasil</button>
</header>
<div id="wrap">
  <div class="intro" id="intro">
    <h2>Panduan Singkat</h2>
    <p>Anda akan menganotasi <b id="n"></b> ulasan hotel. <b>Semua label masih kosong.</b>
    Tugas Anda: baca tiap ulasan, lalu tentukan sendiri label untuk ketujuh aspek.</p>
    <div class="note">
      <b>Putaran ini berbeda dari yang sebelumnya.</b> Tidak ada label awal, tidak ada
      saran dari sistem apa pun. Semua aspek dimulai dari <b>tidak disebut</b>, dan
      Anda yang menentukan seluruhnya. Kerjakan sendiri, tanpa berdiskusi dengan
      anotator lain.
    </div>
    <ul>
      <li>7 aspek: Lokasi, Kebersihan, Pelayanan, Kamar &amp; Fasilitas, Harga, Makanan &amp; Minuman, Fasilitas Pendukung.</li>
      <li>Tiap aspek: pilih <b>tidak disebut</b> / <b>positif</b> / <b>negatif</b> / <b>netral</b>.</li>
      <li>Hanya beri polaritas pada aspek yang memang <b>disebut</b> dalam ulasan; sisanya biarkan <b>tidak disebut</b>.</li>
      <li>Perhatikan negasi ("tidak bersih" = negatif). Nilai dari sudut pandang tamu.</li>
      <li>Bila disebut secara faktual tanpa penilaian, pilih <b>netral</b>.</li>
      <li>Progress tersimpan otomatis di browser. Selesai &rarr; klik <b>Export Hasil</b> &rarr; kirim file ke peneliti.</li>
    </ul>
    <p class="hint">Pintasan keyboard pada aspek terpilih: N=tidak disebut, P=positif, G=negatif, E=netral. Panah &larr;/&rarr; navigasi.</p>
    <button class="btn" id="start">Mulai Anotasi &rarr;</button>
  </div>

  <div class="card" id="card" style="display:none;">
    <div class="meta"><b>#<span id="idx"></span></b> / <span id="total"></span>
      &nbsp;|&nbsp; Hotel: <span id="hotel"></span> &nbsp;|&nbsp; Kota: <span id="city"></span>
      &nbsp;|&nbsp; Rating: <span id="rating"></span>
      <span class="badge done" id="verBadge" style="display:none;">&#10003; sudah</span></div>
    <div class="review" id="review"></div>
    <table><thead><tr><th style="width:38%">Aspek</th><th>Sentimen</th></tr></thead>
      <tbody id="aspects"></tbody></table>
    <div class="nav">
      <button class="btn ghost" id="prev">&larr; Sebelumnya</button>
      <button class="btn" id="next">Simpan &amp; Lanjut &rarr;</button>
      <span class="sp"></span>
      <input type="number" id="jump" min="1" style="width:70px;padding:6px;">
      <button class="btn ghost" id="jumpBtn">Lompat</button>
    </div>
  </div>
</div>
<script>
const AID="__AID__";
const DATA=__DATA__;
const ASPECTS=["Lokasi","Kebersihan","Pelayanan","Kamar & Fasilitas","Harga","Makanan & Minuman","Fasilitas Pendukung"];
const OPTS=["NA","POSITIF","NEGATIF","NETRAL"];
const LABELS={NA:"tidak disebut",POSITIF:"positif",NEGATIF:"negatif",NETRAL:"netral"};
const LS="absa_blind_"+AID;
let data=DATA.map(r=>({...r,labels:[],verified:false}));
let cur=0;
const $=id=>document.getElementById(id);
$("n").textContent=data.length;

const saved=localStorage.getItem(LS);
if(saved){try{const s=JSON.parse(saved);data.forEach(r=>{if(s[r.review_id]){r.labels=s[r.review_id].labels;r.verified=s[r.review_id].verified;}});}catch(_){}}

function curLabels(){const m={};(data[cur].labels||[]).forEach(l=>m[l.aspect]=l.polarity);return m;}
function setLabel(a,p){let l=(data[cur].labels||[]).filter(x=>x.aspect!==a);if(p!=="NA")l.push({aspect:a,polarity:p});data[cur].labels=l;renderAspects();}
function renderAspects(){const m=curLabels();const tb=$("aspects");tb.innerHTML="";
  ASPECTS.forEach(a=>{const tr=document.createElement("tr");
    const t1=document.createElement("td");t1.className="aspect";t1.textContent=a;tr.appendChild(t1);
    const t2=document.createElement("td");const pl=document.createElement("div");pl.className="pills";
    OPTS.forEach(o=>{const c=m[a]||"NA";const b=document.createElement("span");b.className="pill"+(c===o?(" sel-"+o):"");b.textContent=LABELS[o];b.onclick=()=>setLabel(a,o);pl.appendChild(b);});
    t2.appendChild(pl);tr.appendChild(t2);tb.appendChild(tr);});}
function render(){const r=data[cur];$("idx").textContent=cur+1;$("total").textContent=data.length;
  $("hotel").textContent=r.hotel||"-";$("city").textContent=r.city||"-";$("rating").textContent=r.rating??"-";
  $("review").textContent=r.text;$("verBadge").style.display=r.verified?"inline-block":"none";
  $("prog").textContent="Selesai: "+data.filter(x=>x.verified).length+"/"+data.length;$("prev").disabled=cur===0;renderAspects();}
function save(){data[cur].verified=true;const s=JSON.parse(localStorage.getItem(LS)||"{}");
  data.forEach(r=>{if(r.verified)s[r.review_id]={labels:r.labels||[],verified:true};});localStorage.setItem(LS,JSON.stringify(s));}
$("start").onclick=()=>{$("intro").style.display="none";$("card").style.display="block";cur=data.findIndex(r=>!r.verified);if(cur<0)cur=0;render();};
$("next").onclick=()=>{save();if(cur<data.length-1)cur++;render();};
$("prev").onclick=()=>{if(cur>0)cur--;render();};
$("jumpBtn").onclick=()=>{const v=parseInt($("jump").value);if(v>=1&&v<=data.length){cur=v-1;render();}};
$("exportBtn").onclick=()=>{const out=data.map(r=>JSON.stringify({review_id:r.review_id,corpus_review_id:r.corpus_review_id,hotel:r.hotel,city:r.city,rating:r.rating,text:r.text,labels:r.labels||[],verified:!!r.verified,annotator:AID,blind:true})).join("\n");
  const b=new Blob([out],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="anotator_"+AID+"_blind.jsonl";a.click();};
document.addEventListener("keydown",e=>{if($("card").style.display==="none")return;
  if(e.key==="ArrowRight")$("next").click();else if(e.key==="ArrowLeft")$("prev").click();});
</script></body></html>"""

README = """ANOTASI ABSA HOTEL (PUTARAN KEDUA) - Anotator {aid}
=========================================================

Terima kasih sudah membantu sekali lagi.

APA BEDANYA DENGAN PUTARAN PERTAMA?
  Pada putaran pertama, tiap aspek sudah terisi label awal dan tugas Anda
  memeriksa serta mengoreksinya.

  Putaran ini SEMUA LABEL KOSONG. Tidak ada label awal, tidak ada saran dari
  sistem apa pun. Anda menentukan sendiri seluruh label, dari nol.

  Tujuannya: mengukur seberapa konsisten para anotator ketika bekerja tanpa
  bantuan label awal. Ini permintaan reviewer jurnal. TIDAK ADA jawaban yang
  "benar" atau "salah" yang sedang diuji, dan ini bukan penilaian atas kerja
  Anda sebelumnya.

CARA PAKAI (tanpa install):
1. Double-klik file "Anotasi-Anotator-{aid}-Blind.html"
   -> terbuka di browser (Chrome/Edge/Firefox).
2. Baca panduan singkat, klik "Mulai Anotasi".
3. Untuk SETIAP ulasan:
   - Baca ulasannya.
   - Untuk tiap dari 7 aspek, klik: tidak disebut / positif / negatif / netral.
   - Aspek yang tidak disinggung dalam ulasan: biarkan "tidak disebut".
   - Klik "Simpan & Lanjut".
4. Progress tersimpan otomatis di browser (boleh tutup & lanjut nanti,
   ASAL pakai browser & laptop yang sama).
5. Setelah SEMUA selesai, klik "Export Hasil" (kanan atas)
   -> terunduh file "anotator_{aid}_blind.jsonl".
6. Kirim file itu kembali ke peneliti.

PENTING:
- Kerjakan SENDIRI. Jangan berdiskusi dengan anotator lain selama mengerjakan.
- Jangan membuka hasil anotasi Anda yang dulu.
- Nilai sentimen dari sudut pandang tamu terhadap aspek tersebut.
- Perhatikan kata negasi (mis. "tidak bersih" = negatif).
- Bila aspek disebut secara faktual tanpa penilaian, pilih "netral".

Jumlah ulasan: {n}
Perkiraan waktu: sekitar 2-3 jam. Boleh dicicil.

Aspek:
  Lokasi, Kebersihan, Pelayanan, Kamar & Fasilitas, Harga,
  Makanan & Minuman, Fasilitas Pendukung.
"""


def main():
    if not os.path.exists(GOLD):
        sys.exit(f"run this from the project root; {GOLD} not found here")

    rows = []
    with open(GOLD, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"gold set: {len(rows)} reviews")

    rows.sort(key=lambda r: r["review_id"])
    rnd = random.Random(SEED)
    picked = sorted(rnd.sample(rows, N), key=lambda r: r["review_id"])
    print(f"drawn for the blinded round: {len(picked)} (seed={SEED})")

    os.makedirs(OUTDIR, exist_ok=True)

    cleared = []
    for r in picked:
        cleared.append({
            "review_id": r["review_id"],
            "corpus_review_id": r.get("corpus_review_id"),
            "hotel": r.get("hotel", ""),
            "city": r.get("city", ""),
            "rating": r.get("rating"),
            "text": r["text"],
            "labels": [],
        })

    with open(os.path.join(OUTDIR, "blind_input.jsonl"), "w", encoding="utf-8") as f:
        for r in cleared:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUTDIR, "blind_subset_ids.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {N} review_ids drawn from gold_final.jsonl with random.Random({SEED}).sample\n")
        f.write("# for the blinded re-annotation round (Reviewer 1, Comment 4).\n")
        for r in cleared:
            f.write(f"{r['review_id']}\n")

    data_json = json.dumps(cleared, ensure_ascii=False)
    for aid in ANNOTATORS:
        html = HTML.replace("__AID__", aid).replace("__DATA__", data_json)
        hname = f"Anotasi-Anotator-{aid}-Blind.html"
        with open(hname, "w", encoding="utf-8") as f:
            f.write(html)
        zpath = os.path.join(OUTDIR, f"Anotator-{aid}-Blind.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(hname)
            z.writestr(f"PANDUAN-Anotator-{aid}.txt", README.format(aid=aid, n=N))
        os.remove(hname)
        print(f"  {zpath}  ({os.path.getsize(zpath)//1024} KB)")

    print(f"\nDone. Send one ZIP to each annotator.")
    print("When their anotator_{1,2,3}_blind.jsonl come back, put them in this")
    print("folder and run:  python compute_blind_kappa.py")


if __name__ == "__main__":
    main()
