"""Shared helpers: locate the clean / perturbed caption CSVs for every model."""
import glob
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(ROOT, "data", "predictions")
MODELS = ["DAM", "GPT4Roi", "Shikra", "VipLlava"]
FILE_RE = re.compile(r"^(DAM|GPT4Roi|Shikra|VipLlava)_(clean|shift|upscale|downscale)_L(\d)\.csv$")


def discover():
    """Return rows {model, perturbation, level, path}; clean files have perturbation='clean', level=0."""
    rows = []
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "*.csv"))):
        m = FILE_RE.match(os.path.basename(path))
        if m:
            rows.append(dict(model=m.group(1), perturbation=m.group(2), level=int(m.group(3)), path=path))
    return rows


def load_captions(path):
    df = pd.read_csv(path, usecols=["ann_id", "category", "caption", "pred_detailed"])
    for col in ["caption", "pred_detailed"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df
