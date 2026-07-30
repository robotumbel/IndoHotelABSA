"""
llm_fewshot_absa.py — Template ABSA few-shot / zero-shot dengan LLM.

Memprediksi pasangan (aspek, polaritas) untuk tiap review menggunakan LLM via
prompting instruksi Bahasa Indonesia. Output JSONL kompatibel evaluate_absa.py.

Mendukung dua backend (pilih yang tersedia):
  A) LLM lokal via HuggingFace (mis. Merak/SahabatAI/Llama) — offline, gratis.
  B) API (mis. OpenAI-compatible) — set env var; ganti fungsi call_api().

Butuh (mode lokal): pip install torch transformers
Pemakaian:
  python llm_fewshot_absa.py --test test.jsonl --shots 3 --backend hf \
         --model <nama_model_hf> --pred_out pred_llm.jsonl
"""
import argparse
import json
import re

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
POLARITIES = ["POSITIF", "NEGATIF", "NETRAL"]

SYSTEM = (
    "Anda adalah anotator Aspect-Based Sentiment Analysis untuk ulasan hotel. "
    "Untuk setiap ulasan, keluarkan HANYA JSON berupa daftar objek "
    '{"aspect": <aspek>, "polarity": <sentimen>}.\n'
    f"Aspek valid: {ASPECTS}.\n"
    f"Sentimen valid: {POLARITIES}.\n"
    "Jika sebuah aspek tidak disebut, jangan sertakan. "
    "Jangan tambahkan penjelasan apa pun di luar JSON."
)

FEWSHOT_EXAMPLES = [
    ("Lokasinya strategis dekat Malioboro, tapi kamar mandinya bau.",
     [{"aspect": "Lokasi", "polarity": "POSITIF"},
      {"aspect": "Kebersihan", "polarity": "NEGATIF"}]),
    ("Harga terjangkau dan sarapannya enak. Recommended!",
     [{"aspect": "Harga", "polarity": "POSITIF"},
      {"aspect": "Makanan & Minuman", "polarity": "POSITIF"}]),
    ("WiFi lemot, parkir sempit, untung stafnya ramah.",
     [{"aspect": "Fasilitas Pendukung", "polarity": "NEGATIF"},
      {"aspect": "Pelayanan", "polarity": "POSITIF"}]),
    ("Check-in jam 2 siang. Kamar standar di lantai 4.",
     [{"aspect": "Pelayanan", "polarity": "NETRAL"},
      {"aspect": "Kamar & Fasilitas", "polarity": "NETRAL"}]),
    ("Kamar luas dan bersih, kasur empuk, betah!",
     [{"aspect": "Kamar & Fasilitas", "polarity": "POSITIF"},
      {"aspect": "Kebersihan", "polarity": "POSITIF"}]),
]


def build_prompt(review, shots):
    parts = [SYSTEM, ""]
    for txt, labels in FEWSHOT_EXAMPLES[:shots]:
        parts.append(f'Ulasan: "{txt}"')
        parts.append("Jawaban: " + json.dumps(labels, ensure_ascii=False))
        parts.append("")
    parts.append(f'Ulasan: "{review}"')
    parts.append("Jawaban:")
    return "\n".join(parts)


def parse_output(text):
    """Ambil JSON pertama dari keluaran LLM; validasi label; kembalikan list."""
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for o in arr:
        a, p = o.get("aspect"), o.get("polarity")
        if a in ASPECTS and p in POLARITIES:
            out.append({"aspect": a, "polarity": p})
    return out


# ---------------------------------------------------------------
# Backend A — HuggingFace lokal
# ---------------------------------------------------------------
def make_hf_generator(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None)

    def gen(prompt):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen


# ---------------------------------------------------------------
# Backend B — API (template; sesuaikan ke provider Anda)
# ---------------------------------------------------------------
def make_api_generator(model_name):
    """
    Template pemanggilan API OpenAI-compatible. Set API key via env var.
    Ganti isi fungsi sesuai SDK provider yang Anda pakai.
    """
    import os
    from openai import OpenAI            # pip install openai
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def gen(prompt):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=160)
        return resp.choices[0].message.content
    return gen


# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--pred_out", default="pred_llm.jsonl")
    ap.add_argument("--shots", type=int, default=3, help="0=zero-shot")
    ap.add_argument("--backend", choices=["hf", "api"], default="hf")
    ap.add_argument("--model", required=True, help="nama model HF atau API")
    args = ap.parse_args()

    gen = (make_hf_generator(args.model) if args.backend == "hf"
           else make_api_generator(args.model))

    rows = []
    with open(args.test, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results = []
    for i, r in enumerate(rows, 1):
        prompt = build_prompt(r["text"], args.shots)
        raw = gen(prompt)
        labels = parse_output(raw)
        results.append({"review_id": r.get("review_id", i), "labels": labels})
        if i % 25 == 0:
            print(f"  {i}/{len(rows)} selesai")

    with open(args.pred_out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Prediksi ditulis -> {args.pred_out}")
    print("Evaluasi: python evaluate_absa.py --gold " + args.test + " --pred " + args.pred_out)


if __name__ == "__main__":
    main()
