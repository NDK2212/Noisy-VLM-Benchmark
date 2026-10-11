"""Step 1: GPT-4o-mini extracts the primary object and attribute set from every caption.

Each unique caption is sent once; raw responses are appended to cache/extractions.jsonl so a rerun
only calls the API for captions not yet cached.

    python extract.py --limit 50   # pilot on the first 50 regions of every file
    python extract.py              # everything (free if cache/extractions.jsonl is present)
"""
import argparse
import asyncio
import hashlib
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from common import ROOT, discover, load_captions

GPT_MODEL = "gpt-4o-mini"
PROMPT_VERSION = "v1"
CACHE = os.path.join(ROOT, "cache", "extractions.jsonl")
OUT_DIR = os.path.join(ROOT, "outputs", "extractions")

SYSTEM_PROMPT = """You analyze a caption that describes one region of an image.
Extract:
1. "object": the primary object or object part the caption describes, as a short noun phrase
   (1-4 words, singular, no colors/materials/adjectives, e.g. "kettle lid", "trash can", "banner").
2. "attributes": the visual attributes stated for that object: color, material, shape, texture,
   finish, pattern, state/condition, size, and visible text or markings. Each attribute is a short
   phrase of 1-3 words (e.g. "red", "polished metal", "rounded", "glossy", "worn"). Do not include
   object names, positions, or the background. Do not repeat attributes. Use [] if none are stated.
Answer with JSON only: {"object": "...", "attributes": ["...", "..."]}"""


def key(caption):
    return hashlib.sha1(f"{GPT_MODEL}|{PROMPT_VERSION}|{caption}".encode()).hexdigest()


def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["key"]] = rec
    return cache


def parse(raw):
    data = json.loads(raw)
    obj = str(data.get("object") or "").strip().lower()
    attrs = [str(a).strip().lower() for a in data.get("attributes") or [] if str(a).strip()]
    return obj, list(dict.fromkeys(attrs))


async def extract_one(client, sem, caption, fout):
    if not caption:
        rec = dict(key=key(caption), caption=caption, raw="", object="", attributes=[])
    else:
        async with sem:
            for attempt in range(6):
                try:
                    resp = await client.chat.completions.create(
                        model=GPT_MODEL,
                        temperature=0,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": f"Caption: {caption}"}],
                    )
                    raw = resp.choices[0].message.content
                    obj, attrs = parse(raw)
                    break
                except Exception as e:  # rate limits, transient errors, bad JSON
                    if attempt == 5:
                        print(f"FAILED after retries: {e!r} | {caption[:80]}")
                        return
                    await asyncio.sleep(2 ** attempt)
        rec = dict(key=key(caption), caption=caption, raw=raw, object=obj, attributes=attrs)
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fout.flush()


async def run_api(captions, concurrency):
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "a") as fout:
        await tqdm_asyncio.gather(*(extract_one(client, sem, c, fout) for c in captions))


def write_outputs(cache, limit):
    """One CSV per input file: ann_id, caption, extracted object, attributes ('' when none)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for row in discover():
        df = load_captions(row["path"])
        if row["perturbation"] == "clean" and row["model"] == "DAM":
            gt = df.head(limit) if limit else df
            recs = [cache.get(key(c)) for c in gt["caption"]]
            gt = gt[["ann_id", "category", "caption"]].rename(columns={"caption": "pred_detailed"})
            gt["object"] = [r["object"] if r else None for r in recs]
            gt["attributes"] = ["; ".join(r["attributes"]) if r else None for r in recs]
            gt.to_csv(os.path.join(OUT_DIR, "GT_reference_L0.csv"), index=False)
        df = df.drop(columns="caption")
        if limit:
            df = df.head(limit)
        recs = [cache.get(key(c)) for c in df["pred_detailed"]]
        df["object"] = [r["object"] if r else None for r in recs]
        df["attributes"] = ["; ".join(r["attributes"]) if r else None for r in recs]
        name = f"{row['model']}_{row['perturbation']}_L{row['level']}.csv"
        out = os.path.join(OUT_DIR, name)
        df.to_csv(out, index=False)
        missing = df["object"].isna().sum()
        if missing:
            print(f"{name}: {missing} captions still missing an extraction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N regions of each file (pilot)")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    captions = set()
    for row in discover():
        df = load_captions(row["path"])
        df = df.head(args.limit) if args.limit else df
        captions.update(df["pred_detailed"])
        captions.update(df["caption"])  # GPT-4o reference captions

    cache = load_cache()
    todo = sorted(c for c in captions if key(c) not in cache)
    print(f"unique captions: {len(captions)}, cached: {len(captions) - len(todo)}, to extract: {len(todo)}")
    if todo:
        asyncio.run(run_api(todo, args.concurrency))
        cache = load_cache()
    write_outputs(cache, args.limit)


if __name__ == "__main__":
    main()
