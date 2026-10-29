"""3.5-week re-plan, Must-do #1 Tier 0: extract real per-clip multi-hot
7-category labels from the `transcript` field's inline noise-event tags
(e.g. `<bird sound>...</bird sound>`, `[breathing]`), for Bronze
(`no_timestamps`, 17,884 clips, currently only a near-worthless ~100%-truthy
boolean `NoiseCategory`) and Silver (`unverified_timestamps`, 61,642
clips). The tag->category lookup is built from Gold's own real
`NoiseSubCategoryTimeStamp` (tag, category) pairs -- confirmed via direct
inspection: 172 distinct tags, ZERO map to more than one category, so this
is a clean, deterministic lookup, not a guess.

Tags are matched by normalizing away bracket/angle-bracket delimiters,
case, and underscore-vs-space variants (e.g. `[lip_smacking]` and
`[lip smacking]` both appear in the data and must normalize to the same
key). Generic wrapper tags with no Gold-derived mapping (`<noise>`,
`<PAUSE>`, `<static noise>`) are naturally excluded since they never
appear as a `tag` value in any real Gold event.

Output: `data/transcript_tag_labels.json`, `{clip_id: [category, ...]}` for
every Bronze/Silver row with at least one recognized tag, keyed by
`f"row_{i}"` where `i` is the absolute row index into the full (unfiltered)
dataset -- the same convention BronzeInferenceDataset now uses for its own
clip_id (see src/dataset.py's fix, same 3.5-week re-plan). NOT keyed by
`imageFileName`: confirmed NOT unique (90,637 rows -> only 44,139 distinct
values, up to 53 rows sharing one source image).

Usage: python -m temp.build_transcript_tag_labels
"""
import collections
import json
import os
import re

from datasets import Audio, load_dataset

OUT_PATH = "./data/transcript_tag_labels.json"

OPEN_TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z_ ]*)>")
BRACKET_TAG_RE = re.compile(r"\[([a-zA-Z][a-zA-Z_ ]*)\]")


def normalize(tag_text):
    return tag_text.strip().strip("<>[]/").strip().lower().replace("_", " ")


def extract_raw_tags(transcript):
    if not transcript:
        return []
    tags = OPEN_TAG_RE.findall(transcript) + BRACKET_TAG_RE.findall(transcript)
    return [normalize(t) for t in tags]


def main():
    full_ds = load_dataset(
        "ARTPARK-IISc/Vaani-Noise-Event-Dataset",
        split="train", token=os.environ.get("HF_TOKEN"), cache_dir="./data/hf_cache",
    ).cast_column("audio", Audio(decode=False))

    gold = full_ds.filter(lambda r: r["annotationQuality"] == "verified_timestamps")
    lookup = {}
    ambiguous = collections.defaultdict(set)
    for row in gold:
        for event in row["NoiseSubCategoryTimeStamp"] or []:
            key = normalize(event["tag"])
            ambiguous[key].add(event["category"])
            lookup[key] = event["category"]
    hard_ambiguous = {k: v for k, v in ambiguous.items() if len(v) > 1}
    assert not hard_ambiguous, f"tag->category lookup is not deterministic: {hard_ambiguous}"
    print(f"tag->category lookup built from Gold: {len(lookup)} distinct tags, "
          f"all deterministic (0 ambiguous)", flush=True)

    results = {}
    tier_counts = collections.Counter()
    tier_hit_counts = collections.Counter()
    category_counts_by_tier = collections.defaultdict(collections.Counter)
    for i, row in enumerate(full_ds):
        tier = row["annotationQuality"]
        if tier not in ("unverified_timestamps", "no_timestamps"):
            continue
        tier_counts[tier] += 1
        raw_tags = extract_raw_tags(row.get("transcript") or "")
        categories = sorted({lookup[t] for t in raw_tags if t in lookup})
        if not categories:
            continue
        tier_hit_counts[tier] += 1
        for c in categories:
            category_counts_by_tier[tier][c] += 1
        results[f"row_{i}"] = categories

    print(f"\nCoverage: Silver {tier_hit_counts['unverified_timestamps']}/"
          f"{tier_counts['unverified_timestamps']} "
          f"({100*tier_hit_counts['unverified_timestamps']/max(tier_counts['unverified_timestamps'],1):.1f}%), "
          f"Bronze {tier_hit_counts['no_timestamps']}/{tier_counts['no_timestamps']} "
          f"({100*tier_hit_counts['no_timestamps']/max(tier_counts['no_timestamps'],1):.1f}%)", flush=True)

    for tier in ("unverified_timestamps", "no_timestamps"):
        print(f"\n{tier} category distribution (clips with >=1 tag, multi-label so sums > hit count):")
        for cat, n in category_counts_by_tier[tier].most_common():
            print(f"  {cat:22s} {n}")

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    print(f"\nSaved {len(results)} clip_id -> categories entries to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
