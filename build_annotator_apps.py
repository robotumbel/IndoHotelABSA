# -*- coding: utf-8 -*-
"""
build_annotator_apps.py — Buat aplikasi verifikasi MANDIRI (self-contained HTML)
per anotator + ZIP. Data ditanam di dalam HTML (tanpa server/install/exe).
Anotator cukup double-klik HTML -> verifikasi -> Export -> kirim balik file JSONL.

Menghasilkan: dist/Anotator-1.zip, dist/Anotator-2.zip, dist/Anotator-3.zip
"""
import json, os, zipfile

GOLD = "gold_subset.jsonl"
ANNOTATORS = ["1", "2", "3"]        # 3 anotator (review sama -> Kappa 3-arah)
OUTDIR = "dist"

rows = []
with open(GOLD, encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
data_json = json.dumps(rows, ensure_ascii=False)

HTML = r"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verifikasi ABSA Hotel - Anotator __AID__</title>
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
</style></head><body>
<header>
  <h1>Verifikasi ABSA Hotel &mdash; Anotator __AID__</h1>
  <span class="sp"></span>
  <span class="prog" id="prog"></span>
  <button class="btn ghost" id="exportBtn">&#128190; Export Hasil</button>
</header>
<div id="wrap">
  <div class="intro" id="intro">
    <h2>Panduan Singkat</h2>
    <p>Anda akan memeriksa <b id="n"></b> ulasan hotel. Setiap ulasan sudah diberi
    label AWAL (oleh AI). <b>Tugas Anda: periksa dan koreksi</b> tiap aspek.</p>
    <ul>
      <li>7 aspek: Lokasi, Kebersihan, Pelayanan, Kamar &amp; Fasilitas, Harga, Makanan &amp; Minuman, Fasilitas Pendukung.</li>
      <li>Tiap aspek: pilih <b>tidak disebut</b> / <b>positif</b> / <b>negatif</b> / <b>netral</b>.</li>
      <li>Perhatikan negasi ("tidak bersih" = negatif). Nilai dari sudut pandang tamu.</li>
      <li><b>Jangan asal setuju</b> dengan label AI &mdash; periksa sungguh-sungguh.</li>
      <li>Progress tersimpan otomatis di browser. Selesai &rarr; klik <b>Export Hasil</b> &rarr; kirim file ke peneliti.</li>
    </ul>
    <p class="hint">Pintasan keyboard pada aspek terpilih: N=tidak disebut, P=positif, G=negatif, E=netral. Panah &larr;/&rarr; navigasi.</p>
    <button class="btn" id="start">Mulai Verifikasi &rarr;</button>
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
const LS="absa_anotator_"+AID;
let data=DATA.map(r=>({...r,labels:(r.labels||[]).slice(),verified:false}));
let cur=0;
const $=id=>document.getElementById(id);
$("n").textContent=data.length;

// muat progress
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
$("exportBtn").onclick=()=>{const out=data.map(r=>JSON.stringify({review_id:r.review_id,hotel:r.hotel,city:r.city,rating:r.rating,text:r.text,labels:r.labels||[],verified:!!r.verified,annotator:AID})).join("\n");
  const b=new Blob([out],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="anotator_"+AID+"_verified.jsonl";a.click();};
document.addEventListener("keydown",e=>{if($("card").style.display==="none")return;
  if(e.key==="ArrowRight")$("next").click();else if(e.key==="ArrowLeft")$("prev").click();});
</script></body></html>"""

README = """PANDUAN VERIFIKASI ABSA HOTEL - Anotator {aid}
====================================================

TERIMA KASIH sudah membantu memverifikasi data penelitian ini.

CARA PAKAI (sangat mudah, tanpa install):
1. Double-klik file "Verifikasi-Anotator-{aid}.html"
   -> akan terbuka di browser (Chrome/Edge/Firefox).
2. Baca panduan singkat di layar, klik "Mulai Verifikasi".
3. Untuk SETIAP ulasan:
   - Baca ulasannya.
   - Untuk tiap dari 7 aspek, klik: tidak disebut / positif / negatif / netral.
   - Label awal dari AI sudah terisi -> PERIKSA dan KOREKSI bila salah.
   - Klik "Simpan & Lanjut".
4. Progress tersimpan otomatis di browser (boleh tutup & lanjut nanti,
   ASAL pakai browser & laptop yang sama).
5. Setelah SEMUA selesai, klik tombol "Export Hasil" (kanan atas)
   -> akan terunduh file "anotator_{aid}_verified.jsonl".
6. Kirim file itu kembali ke peneliti.

PENTING:
- Mohon periksa SUNGGUH-SUNGGUH, jangan asal setuju dengan label AI.
- Nilai sentimen dari sudut pandang tamu terhadap aspek tsb.
- Perhatikan kata negasi (mis. "tidak bersih" = negatif).

Aspek:
  Lokasi, Kebersihan, Pelayanan, Kamar & Fasilitas, Harga,
  Makanan & Minuman, Fasilitas Pendukung.
"""

os.makedirs(OUTDIR, exist_ok=True)
for aid in ANNOTATORS:
    html = HTML.replace("__AID__", aid).replace("__DATA__", data_json)
    hname = f"Verifikasi-Anotator-{aid}.html"
    with open(hname, "w", encoding="utf-8") as f:
        f.write(html)
    zpath = os.path.join(OUTDIR, f"Anotator-{aid}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(hname)
        z.writestr(f"PANDUAN-Anotator-{aid}.txt", README.format(aid=aid))
    os.remove(hname)
    kb = os.path.getsize(zpath) // 1024
    print(f"{zpath}  ({kb} KB)")
print(f"\nSelesai. {len(ANNOTATORS)} ZIP di folder '{OUTDIR}/'. Bagikan ke tiap anotator.")
