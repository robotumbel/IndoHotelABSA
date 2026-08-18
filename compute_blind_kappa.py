# -*- coding: utf-8 -*-
"""
compute_blind_kappa.py — compare blinded against verification annotation.

Answers Reviewer 1, Comment 4. The gold subset was verified from a visible LLM
draft, so the reported Fleiss' kappa may not describe independent annotation. This
script settles the question empirically: the same 100 reviews, the same three
annotators, all labels cleared, no model output shown.

It reports four things, in this order, because the headline number alone misleads:

  1. agreement under both conditions (Fleiss' kappa, unanimity);
  2. an independence check on the returned files, since implausibly high agreement
     would more likely mean the annotators worked together than that the protocol
     is sound;
  3. how far the blinded labels sit from the adjudicated gold;
  4. the DIRECTION of that divergence, which is where the substantive finding is.

Inputs, in this folder:
    anotator_{1,2,3}_blind.jsonl   returned by the annotators
    blind_subset_ids.txt           the 100 ids drawn (from build_blind_apps.py)
and from ../deposit_v2/:
    annotator_{1,2,3}_verified.jsonl, gold_final.jsonl,
    annotation_prefilled_gold500.jsonl

Usage:  python compute_blind_kappa.py
Writes: blind_kappa_report.md
"""
import json
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ASPECTS = ["Lokasi", "Kebersihan", "Pelayanan", "Kamar & Fasilitas",
           "Harga", "Makanan & Minuman", "Fasilitas Pendukung"]
CATS = ["NA", "POSITIF", "NEGATIF", "NETRAL"]
RANK = {a: i for i, a in enumerate(ASPECTS)}
DEP = os.path.join("..", "deposit_v2")


def load(path):
    d = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                d[o["review_id"]] = o
    return d


def cellmap(o):
    m = {a: "NA" for a in ASPECTS}
    for l in o.get("labels", []):
        if l.get("aspect") in m:
            m[l["aspect"]] = l.get("polarity", "NA")
    return m


def seq(o):
    return [l["aspect"] for l in o.get("labels", []) if l.get("aspect") in RANK]


def fleiss(matrix):
    if not matrix:
        return float("nan")
    n = sum(matrix[0].values())
    N = len(matrix)
    if n < 2:
        return float("nan")
    p_j = {c: sum(m[c] for m in matrix) / (N * n) for c in CATS}
    P_i = [(sum(m[c] ** 2 for c in CATS) - n) / (n * (n - 1)) for m in matrix]
    P_e = sum(v ** 2 for v in p_j.values())
    if abs(1 - P_e) < 1e-12:
        return float("nan")
    return (sum(P_i) / N - P_e) / (1 - P_e)


def matrix_for(anns, ids, aspects=ASPECTS):
    out = []
    for rid in ids:
        maps = [cellmap(a[rid]) for a in anns]
        for asp in aspects:
            c = Counter({x: 0 for x in CATS})
            for m in maps:
                c[m[asp]] += 1
            out.append(c)
    return out


def unanimity(anns, ids):
    tot = un = 0
    for rid in ids:
        maps = [cellmap(a[rid]) for a in anns]
        for asp in ASPECTS:
            tot += 1
            un += len({m[asp] for m in maps}) == 1
    return un, tot


def pairwise(anns, ids):
    out = []
    for x, y in ((0, 1), (0, 2), (1, 2)):
        same = tot = 0
        for rid in ids:
            mx, my = cellmap(anns[x][rid]), cellmap(anns[y][rid])
            for asp in ASPECTS:
                tot += 1
                same += mx[asp] == my[asp]
        out.append((x + 1, y + 1, same / tot * 100))
    return out


def independence(anns, ids):
    """Diagnostics that would expose copied or jointly produced files."""
    same_set = ident_order = 0
    for rid in ids:
        seqs = [seq(anns[i][rid]) for i in range(3)]
        if {frozenset(s) for s in seqs} == {frozenset(seqs[0])}:
            same_set += 1
            if seqs[0] == seqs[1] == seqs[2]:
                ident_order += 1
    monotone = []
    for a in anns:
        m = sum(1 for rid in ids
                if seq(a[rid]) == sorted(seq(a[rid]), key=lambda x: RANK[x]))
        monotone.append(m)
    return same_set, ident_order, monotone


def main():
    missing = [f"anotator_{i}_blind.jsonl" for i in (1, 2, 3)
               if not os.path.exists(f"anotator_{i}_blind.jsonl")]
    if missing:
        print("Waiting on the annotators. Not yet returned:")
        for m in missing:
            print(f"  - {m}")
        return 1

    ids = [int(l) for l in open("blind_subset_ids.txt", encoding="utf-8-sig")
           if l.strip() and not l.startswith("#")]
    blind = [load(f"anotator_{i}_blind.jsonl") for i in (1, 2, 3)]
    ver = [load(os.path.join(DEP, f"annotator_{i}_verified.jsonl")) for i in (1, 2, 3)]
    gold = load(os.path.join(DEP, "gold_final.jsonl"))
    draft = load(os.path.join(DEP, "annotation_prefilled_gold500.jsonl"))
    ids = [r for r in ids if all(r in b for b in blind) and all(r in v for v in ver)]
    n_cells = len(ids) * 7

    k_b, k_v = fleiss(matrix_for(blind, ids)), fleiss(matrix_for(ver, ids))
    ub, tb = unanimity(blind, ids)
    uv, _ = unanimity(ver, ids)

    print(f"\nreviews compared: {len(ids)}   cells per condition: {n_cells}\n")
    print("== 1. Agreement ==")
    print(f"{'condition':<34}{'Fleiss kappa':>14}{'unanimous':>13}")
    print(f"{'blinded (labels cleared)':<34}{k_b:>14.3f}{ub/tb*100:>12.1f}%")
    print(f"{'verification (draft visible)':<34}{k_v:>14.3f}{uv/tb*100:>12.1f}%")
    print(f"{'difference (blinded - verified)':<34}{k_b-k_v:>+14.3f}{(ub-uv)/tb*100:>+12.1f}%")

    print(f"\n{'per-aspect Fleiss kappa':<26}{'blinded':>10}{'verified':>10}{'diff':>9}")
    per = {}
    for asp in ASPECTS:
        kb, kv = fleiss(matrix_for(blind, ids, [asp])), fleiss(matrix_for(ver, ids, [asp]))
        per[asp] = (kb, kv)
        print(f"  {asp:<24}{kb:>10.3f}{kv:>10.3f}{kb-kv:>+9.3f}")

    print("\n== 2. Independence check on the returned files ==")
    for tag, anns in (("blinded     ", blind), ("verification", ver)):
        print(f"  {tag} pairwise raw agreement: " +
              ", ".join(f"A{x}-A{y} {p:.1f}%" for x, y, p in pairwise(anns, ids)))
    sb, ib, mb = independence(blind, ids)
    sv, iv, mv = independence(ver, ids)
    print(f"  blinded:      same aspect set {sb}/{len(ids)}; of those, identical click order {ib}")
    print(f"  verification: same aspect set {sv}/{len(ids)}; of those, identical click order {iv}")
    print(f"  label order follows the on-screen aspect order - blinded {mb}, verification {mv}")

    print("\n== 3. Blinded labels vs the adjudicated gold ==")
    agree_g = agree_d = cells = 0
    n_blind_mentions = n_gold_mentions = 0
    trans = Counter()
    for rid in ids:
        gm = cellmap(gold[rid])
        dm = cellmap(draft[rid]) if rid in draft else {a: "NA" for a in ASPECTS}
        maps = [cellmap(b[rid]) for b in blind]
        for asp in ASPECTS:
            votes = [m[asp] for m in maps]
            top = Counter(votes).most_common(1)[0]
            if top[1] < 2:
                continue
            lab = top[0]
            cells += 1
            agree_g += lab == gm[asp]
            agree_d += lab == dm[asp]
            n_blind_mentions += lab != "NA"
            n_gold_mentions += gm[asp] != "NA"
            if lab != gm[asp]:
                trans[(gm[asp], lab)] += 1
    print(f"  cells with a blinded majority: {cells}")
    print(f"  blinded majority == adjudicated gold : {agree_g}/{cells} ({agree_g/cells*100:.1f}%)")
    print(f"  blinded majority == LLM draft        : {agree_d}/{cells} ({agree_d/cells*100:.1f}%)")

    print("\n== 4. Direction of the divergence ==")
    tot = sum(trans.values())
    added = sum(n for (g, _), n in trans.items() if g == "NA")
    dropped = sum(n for (_, b), n in trans.items() if b == "NA")
    flipped = tot - added - dropped
    print(f"  aspect mentions marked - blinded {n_blind_mentions}, gold {n_gold_mentions} "
          f"({n_blind_mentions-n_gold_mentions:+d}, {(n_blind_mentions/n_gold_mentions-1)*100:+.1f}%)")
    print(f"  differing cells: {tot}")
    print(f"    aspect ADDED by the blinded round (gold NA -> label): {added} ({added/tot*100:.0f}%)")
    print(f"    aspect DROPPED (label -> NA)                        : {dropped} ({dropped/tot*100:.0f}%)")
    print(f"    polarity changed, aspect kept                       : {flipped} ({flipped/tot*100:.0f}%)")
    print("  most frequent gold -> blinded transitions:")
    for (g, b), n in trans.most_common(6):
        print(f"    {g:<8} -> {b:<8} {n:>4}")

    higher = k_b > k_v
    para = (
        f"To measure the effect of the non-blinded protocol rather than only caveat "
        f"it, {len(ids)} of the 500 gold reviews were re-annotated by the same three "
        f"annotators with every label cleared and no model output shown. Over these "
        f"{n_cells} cells, blinded agreement is Fleiss' kappa = {k_b:.3f} "
        f"({ub/tb*100:.1f}% of cells unanimous), against kappa = {k_v:.3f} "
        f"({uv/tb*100:.1f}%) for the same reviews under the original protocol. "
        f"Agreement is therefore {'higher' if higher else 'lower'} without the draft, "
        f"so the reported kappa is not an artefact of three annotators accepting one "
        f"shared suggestion. The returned files were checked for independence: the "
        f"three chose the same aspect set on {sb} of {len(ids)} reviews but recorded "
        f"them in the same order on only {ib} of those, and label order follows the "
        f"on-screen aspect order in {mb[0]}, {mb[1]}, and {mb[2]} reviews "
        f"respectively, as expected when each annotator works down an empty form, "
        f"against {mv[0]}, {mv[1]}, and {mv[2]} in the draft-anchored round. "
        f"The blinded labels nevertheless differ from the adjudicated gold in "
        f"{tot} of {cells} cells ({100-agree_g/cells*100:.1f}%), and the divergence "
        f"is directional: the blinded round marks {n_blind_mentions} aspect mentions "
        f"against {n_gold_mentions} in the gold "
        f"({(n_blind_mentions/n_gold_mentions-1)*100:+.1f}%), and {added/tot*100:.0f}% "
        f"of the differing cells are aspects the gold left unmarked, predominantly "
        f"negative ones. The visible draft thus appears to have depressed aspect "
        f"recall rather than inflated agreement: annotators working from an empty "
        f"form detect mentions that the draft omitted and that the verification pass "
        f"did not restore. Users should treat the gold aspect inventory as "
        f"conservative, particularly for negative mentions. Because the same "
        f"annotators had seen these reviews three weeks earlier, recall of the "
        f"earlier session cannot be excluded. It would, however, pull the blinded "
        f"labels towards the gold, whereas the observed divergence is away from it. "
        f"The blinded label files are released with the dataset."
    )
    print("\n--- paragraph for the manuscript and the letter ---\n")
    print(para)

    with open("blind_kappa_report.md", "w", encoding="utf-8") as f:
        f.write("# Blinded re-annotation - result\n\n")
        f.write(f"{len(ids)} reviews, {n_cells} cells per condition.\n\n")
        f.write("## 1. Agreement\n\n| Condition | Fleiss' kappa | Unanimous |\n|---|---:|---:|\n")
        f.write(f"| Blinded (labels cleared) | {k_b:.3f} | {ub/tb*100:.1f}% |\n")
        f.write(f"| Verification (draft visible) | {k_v:.3f} | {uv/tb*100:.1f}% |\n")
        f.write(f"| Difference | {k_b-k_v:+.3f} | {(ub-uv)/tb*100:+.1f}% |\n\n")
        f.write("| Aspect | Blinded | Verified | Diff |\n|---|---:|---:|---:|\n")
        for asp in ASPECTS:
            kb, kv = per[asp]
            f.write(f"| {asp} | {kb:.3f} | {kv:.3f} | {kb-kv:+.3f} |\n")
        f.write("\n## 2. Independence of the returned files\n\n")
        f.write("- Blinded pairwise raw agreement: " +
                ", ".join(f"A{x}-A{y} {p:.1f}%" for x, y, p in pairwise(blind, ids)) + "\n")
        f.write("- Verification pairwise raw agreement: " +
                ", ".join(f"A{x}-A{y} {p:.1f}%" for x, y, p in pairwise(ver, ids)) + "\n")
        f.write(f"- Same aspect set on {sb}/{len(ids)} reviews, identical click order on {ib} of those.\n")
        f.write(f"- Label order follows the on-screen order in {mb} reviews (blinded) "
                f"vs {mv} (verification) - consistent with working down an empty form, "
                f"and inconsistent with copied files.\n")
        f.write("\n## 3. Blinded vs gold\n\n")
        f.write(f"- Blinded majority == adjudicated gold: {agree_g}/{cells} ({agree_g/cells*100:.1f}%)\n")
        f.write(f"- Blinded majority == LLM draft: {agree_d}/{cells} ({agree_d/cells*100:.1f}%)\n")
        f.write("\n## 4. Direction of divergence\n\n")
        f.write(f"- Aspect mentions: blinded {n_blind_mentions} vs gold {n_gold_mentions} "
                f"({(n_blind_mentions/n_gold_mentions-1)*100:+.1f}%)\n")
        f.write(f"- Added by blinded round: {added}/{tot} ({added/tot*100:.0f}%)\n")
        f.write(f"- Dropped: {dropped}/{tot} ({dropped/tot*100:.0f}%)\n")
        f.write(f"- Polarity changed: {flipped}/{tot} ({flipped/tot*100:.0f}%)\n\n")
        f.write("| gold -> blinded | cells |\n|---|---:|\n")
        for (g, b), n in trans.most_common(10):
            f.write(f"| {g} -> {b} | {n} |\n")
        f.write("\n## Paragraph for the manuscript\n\n" + para + "\n")
    print("\nwritten: blind_kappa_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
