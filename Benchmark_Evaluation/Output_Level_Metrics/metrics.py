"""Step 2: OIP / CSR / APR / HAR (paper Table 3) from the extractions written by extract.py.

    python metrics.py               # reference = GPT-4o DAM-PACO reference caption (paper setting)
    python metrics.py --ref clean   # reference = the same model's caption under the clean box
Writes outputs/metrics_per_region.csv and outputs/metrics_spatial.csv (mean per model/perturbation/level);
with --ref clean the files get the suffix _cleanref.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from common import ROOT

EXT_DIR = os.path.join(ROOT, "outputs", "extractions")
OIP_TAU, ATTR_TAU = 0.7, 0.6


def split_attrs(s):
    # Empty attribute set is stored as ''.
    return [a for a in str(s).split("; ") if a] if isinstance(s, str) else []


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.vec = {}

    def fit(self, phrases):
        new = sorted({p for p in [*phrases, ""] if p not in self.vec})
        if new:
            embs = self.model.encode(new, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
            self.vec.update(zip(new, embs))

    def cos(self, a, b):
        if not a or not b:
            return 0.0
        return float(self.vec[a] @ self.vec[b])

    def matrix(self, A, B):
        return np.stack([self.vec[a] for a in A]) @ np.stack([self.vec[b] for b in B]).T


def region_metrics(emb, oc, on, Ac, An):
    cos_o = emb.cos(oc, on)
    # Original convention: an empty attribute set is the empty string, kept as one attribute and embedded.
    # A caption without attributes therefore gets HAR = 1 (its '' matches nothing); this reproduces the
    # paper's baseline HAR (~0.72-0.80).
    S = emb.matrix(Ac or [""], An or [""])
    apr = float((S.max(axis=1) >= ATTR_TAU).mean())
    har = float((S.max(axis=0) < ATTR_TAU).mean())
    return dict(OIP=float(cos_o >= OIP_TAU), CSR=1.0 - cos_o, APR=apr, HAR=har)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", choices=["gt", "clean"], default="gt")
    args = ap.parse_args()

    files = {f[:-4]: pd.read_csv(os.path.join(EXT_DIR, f), keep_default_na=False)
             for f in sorted(os.listdir(EXT_DIR)) if f.endswith(".csv")}
    emb = Embedder()
    for df in files.values():
        emb.fit(df["object"])
        emb.fit(a for s in df["attributes"] for a in split_attrs(s))

    rows = []
    for name, df in files.items():
        model, pert, level = name.rsplit("_", 2)
        if model == "GT" or (pert == "clean" and args.ref == "clean"):
            continue
        ref_name = "GT_reference_L0" if args.ref == "gt" else f"{model}_clean_L0"
        clean = files[ref_name].set_index("ann_id")
        for r in df.itertuples():
            c = clean.loc[r.ann_id]
            m = region_metrics(emb, c["object"], r.object, split_attrs(c["attributes"]), split_attrs(r.attributes))
            rows.append(dict(model=model, perturbation=pert, level=int(level[1:]), ann_id=r.ann_id, **m))

    per_region = pd.DataFrame(rows)
    suffix = "" if args.ref == "gt" else "_cleanref"
    per_region.to_csv(os.path.join(ROOT, "outputs", f"metrics_per_region{suffix}.csv"), index=False)
    summary = (per_region.groupby(["model", "perturbation", "level"])[["OIP", "CSR", "APR", "HAR"]]
               .mean().round(3).reset_index())
    summary.to_csv(os.path.join(ROOT, "outputs", f"metrics_spatial{suffix}.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
