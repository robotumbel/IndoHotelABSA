"""
evaluate_absa.py — Evaluasi Aspect-Based Sentiment Analysis (IndoHotelABSA).

Menghitung:
  - ACD  : Precision/Recall/F1 per-aspek + macro/micro-F1 (deteksi aspek)
  - ACP  : akurasi & macro-F1 polaritas (pada aspek yang benar terdeteksi)
  - Joint: F1 pasangan (aspek+polaritas) benar

Format input (JSONL, satu review per baris):
  {"review_id": 1, "labels": [{"aspect": "Kebersihan", "polarity": "POSITIF"}, ...]}

Pemakaian:
  python evaluate_absa.py --gold gold.jsonl --pred pred.jsonl
"""
import argparse
import json
from collections import defaultdict

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
POLARITIES = ["POSITIF", "NEGATIF", "NETRAL"]


def load_jsonl(path):
    data = {}
    # utf-8-sig menoleransi Byte-Order-Mark (BOM) yang sering ditambah editor Windows
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = obj["review_id"]
            # set pasangan (aspek, polaritas) dan set aspek saja
            pairs, aspects = set(), set()
            for lab in obj.get("labels", []):
                a, p = lab["aspect"], lab["polarity"]
                pairs.add((a, p))
                aspects.add(a)
            data[rid] = {"pairs": pairs, "aspects": aspects,
                         "pol": {a: p for a, p in pairs}}
    return data


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def evaluate(gold, pred):
    ids = sorted(gold.keys())

    # ---- ACD per-aspek ----
    per_aspect = {a: [0, 0, 0] for a in ASPECTS}  # tp, fp, fn
    for rid in ids:
        g = gold[rid]["aspects"]
        p = pred.get(rid, {}).get("aspects", set())
        for a in ASPECTS:
            if a in g and a in p:
                per_aspect[a][0] += 1
            elif a in p and a not in g:
                per_aspect[a][1] += 1
            elif a in g and a not in p:
                per_aspect[a][2] += 1

    print("\n=== ACD (Aspect Category Detection) ===")
    print(f"{'Aspek':<22}{'P':>7}{'R':>7}{'F1':>7}")
    f1s = []
    TP = FP = FN = 0
    for a in ASPECTS:
        tp, fp, fn = per_aspect[a]
        pr, rc, f1 = prf(tp, fp, fn)
        f1s.append(f1)
        TP += tp; FP += fp; FN += fn
        print(f"{a:<22}{pr:>7.3f}{rc:>7.3f}{f1:>7.3f}")
    macro_f1 = sum(f1s) / len(f1s)
    _, _, micro_f1 = prf(TP, FP, FN)
    print(f"{'MACRO-F1':<22}{'':>14}{macro_f1:>7.3f}")
    print(f"{'MICRO-F1':<22}{'':>14}{micro_f1:>7.3f}")

    # ---- ACP: akurasi polaritas pada aspek yang benar terdeteksi ----
    pol_correct = pol_total = 0
    pol_per = {p: [0, 0, 0] for p in POLARITIES}  # tp, fp, fn (macro-F1 polaritas)
    for rid in ids:
        gp = gold[rid]["pol"]
        pp = pred.get(rid, {}).get("pol", {})
        for a in gp:
            if a in pp:  # aspek terdeteksi di prediksi
                pol_total += 1
                if pp[a] == gp[a]:
                    pol_correct += 1
                    pol_per[gp[a]][0] += 1
                else:
                    pol_per[pp[a]][1] += 1
                    pol_per[gp[a]][2] += 1
    acp_acc = pol_correct / pol_total if pol_total else 0.0
    pol_f1s = [prf(*pol_per[p])[2] for p in POLARITIES]
    print("\n=== ACP (Aspect Category Polarity) ===")
    print(f"Akurasi polaritas (pada aspek benar): {acp_acc:.3f}")
    print(f"Macro-F1 polaritas: {sum(pol_f1s)/len(pol_f1s):.3f}")

    # ---- Joint ABSA ----
    jTP = jFP = jFN = 0
    for rid in ids:
        g = gold[rid]["pairs"]
        p = pred.get(rid, {}).get("pairs", set())
        jTP += len(g & p)
        jFP += len(p - g)
        jFN += len(g - p)
    jpr, jrc, jf1 = prf(jTP, jFP, jFN)
    print("\n=== Joint ABSA (aspek + polaritas) ===")
    print(f"Precision {jpr:.3f} | Recall {jrc:.3f} | F1 {jf1:.3f}")

    return {"acd_macro_f1": macro_f1, "acd_micro_f1": micro_f1,
            "acp_acc": acp_acc, "joint_f1": jf1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="file gold JSONL")
    ap.add_argument("--pred", required=True, help="file prediksi JSONL")
    args = ap.parse_args()
    gold = load_jsonl(args.gold)
    pred = load_jsonl(args.pred)
    evaluate(gold, pred)


if __name__ == "__main__":
    main()
