"""Step 3: threshold calibration (paper Sec. III-F.4, Fig. 5) and threshold sensitivity.

    python calibrate.py
Run after extract.py. Uses the GT reference as o_c / A_c (the setting that reproduces the paper).
"""
import os
import random

import numpy as np
import pandas as pd

from common import ROOT
from metrics import EXT_DIR, Embedder, split_attrs

N_RANDOM = 99_998
OIP_TAU, ATTR_TAU = 0.7, 0.6

# Reconstructed list: the paper's 20 curated pairs are lost except smooth-sleek and glossy-shiny.
SYNONYMS = [
    ("smooth", "sleek"), ("glossy", "shiny"), ("matte", "dull"), ("red", "crimson"),
    ("round", "circular"), ("worn", "weathered"), ("metallic", "metal"), ("wooden", "made of wood"),
    ("transparent", "clear"), ("rectangular", "oblong"), ("dark", "dim"), ("large", "big"),
    ("small", "tiny"), ("curved", "rounded"), ("textured", "rough"), ("striped", "lined"),
    ("silver", "silvery"), ("polished", "shiny"), ("faded", "washed out"), ("thin", "slender"),
]


def main():
    files = {f[:-4]: pd.read_csv(os.path.join(EXT_DIR, f), keep_default_na=False)
             for f in sorted(os.listdir(EXT_DIR)) if f.endswith(".csv")}
    gt = files.pop("GT_reference_L0").set_index("ann_id")
    perturbed = {k: v for k, v in files.items() if "_clean_" not in k}

    emb = Embedder()
    for df in [gt.reset_index(), *files.values()]:
        emb.fit(df["object"])
        emb.fit(a for s in df["attributes"] for a in split_attrs(s))
    emb.fit(w for p in SYNONYMS for w in p)

    # Random unrelated pairs: phrases drawn from two different regions (occurrence-weighted).
    rng = random.Random(0)
    obj_pool, attr_pool = [], []
    for df in files.values():
        for r in df.itertuples():
            if r.object:
                obj_pool.append((r.ann_id, r.object))
            attr_pool += [(r.ann_id, a) for a in split_attrs(r.attributes)]

    def random_pairs(pool, n):
        out = []
        while len(out) < n:
            (ia, a), (ib, b) = rng.choice(pool), rng.choice(pool)
            if ia != ib and a != b:
                out.append(emb.cos(a, b))
        return np.array(out)

    # Paper floor (0.1538) is reproduced by pairing a random object with a random attribute (0.157);
    # same-type pairs give a higher floor and are reported for reference only.
    objs = sorted({o for _, o in obj_pool})
    attrs = sorted({a for _, a in attr_pool})
    rand = np.array([emb.cos(rng.choice(objs), rng.choice(attrs)) for _ in range(N_RANDOM)])
    floor = rand.mean()
    same_type = np.concatenate([random_pairs(obj_pool, N_RANDOM // 2), random_pairs(attr_pool, N_RANDOM // 2)])

    # Matched pairs: reference object vs perturbed object, and each reference attribute vs its best match.
    matched_obj, matched_attr = [], []
    for df in perturbed.values():
        for r in df.itertuples():
            ref = gt.loc[r.ann_id]
            matched_obj.append(emb.cos(ref["object"], r.object))
            Ac, An = split_attrs(ref["attributes"]), split_attrs(r.attributes)
            if Ac and An:
                matched_attr += list(emb.matrix(Ac, An).max(axis=1))
    matched_obj, matched_attr = np.array(matched_obj), np.array(matched_attr)

    syn = np.array([emb.cos(a, b) for a, b in SYNONYMS])

    print(f"Random object-attribute pairs (N={len(rand):,}): floor = {floor:.4f}  (paper 0.1538)")
    print(f"  (same-type object/object + attr/attr pairs: floor {same_type.mean():.4f})")
    print(f"  share > 0.2: {(rand > 0.2).mean():.1%}   >= 0.6: {(rand >= 0.6).mean():.2%}   >= 0.7: {(rand >= 0.7).mean():.2%}")
    print(f"  tau / floor: OIP {OIP_TAU / floor:.1f}x (paper 4.5x), APR/HAR {ATTR_TAU / floor:.1f}x (paper 3.9x)")
    print(f"Matched object pairs (N={len(matched_obj):,}): mean {matched_obj.mean():.3f}, "
          f">= 0.7: {(matched_obj >= 0.7).sum():,} pairs (paper N=14,440)")
    print(f"Matched attribute best-match pairs (N={len(matched_attr):,}): mean {matched_attr.mean():.3f}, >= 0.6: {(matched_attr >= 0.6).mean():.1%}")
    print(f"Synonym pairs (N={len(syn)}): {(syn >= ATTR_TAU).sum()}/{len(syn)} >= 0.6")
    for (a, b), c in sorted(zip(SYNONYMS, syn), key=lambda x: x[1]):
        print(f"    {a:12s} {b:12s} {c:.3f}")

    out = os.path.join(ROOT, "outputs", "calibration")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "random.npy"), rand)
    np.save(os.path.join(out, "matched_obj.npy"), matched_obj)
    np.save(os.path.join(out, "matched_attr.npy"), matched_attr)
    pd.DataFrame(SYNONYMS, columns=["a", "b"]).assign(cos=syn).to_csv(os.path.join(out, "synonyms.csv"), index=False)


if __name__ == "__main__":
    main()
