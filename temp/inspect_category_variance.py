"""Session 9 plan, Part 1 Step 0 (mandatory gate): does NoiseSubCategoryTimeStamp's
`category` field actually carry independent, non-degenerate per-frame signal
across the 7 categories, on both Gold and Silver -- or would a 7-way multi-hot
auxiliary head collapse the same way the earlier NoiseCategory-as-boolean
attempt did (100% truthy across the self-distillation pool, trivial
always-positive solution, see temp/inspect_noise_category.py)?

Checks, per tier (Gold=verified_timestamps, Silver=unverified_timestamps):
  1. Per-category event counts (sanity check against the already-known Gold
     numbers: human_non_speech 1924, vehicle_traffic 1281, animal 1142,
     baby_child 496, phone_signal_alarm 188, singing_music 68,
     appliance_machine 58 -- from direct validationMetadata.json inspection).
  2. Per-category FRAME coverage (sum of end-start), not just clip/event
     counts -- frame coverage is what actually matters for a per-frame BCE
     target, and clip counts alone can be misleading (breathing is ~45% of
     frames despite being one tag among many).
  3. Co-occurrence: do the 7 channels ever move independently of each other
     (not all firing together whenever any noise is present)?

Usage: python -m temp.inspect_category_variance
"""
from collections import Counter, defaultdict

from datasets import Audio
from tqdm import tqdm

from src.dataset import load_full_dataset

CATEGORY_LIST = ["human_non_speech", "vehicle_traffic", "animal", "baby_child",
                 "phone_signal_alarm", "singing_music", "appliance_machine"]


def analyze_tier(full_dataset, indices, tier_name):
    print(f"\n{'='*60}\n{tier_name}: {len(indices)} rows\n{'='*60}", flush=True)
    subset = full_dataset.select(indices)
    events_col = subset["NoiseSubCategoryTimeStamp"]

    event_counts = Counter()
    frame_coverage = defaultdict(float)  # category -> total seconds covered
    rows_with_events = 0
    rows_with_multiple_categories = 0
    co_occurrence = Counter()  # frozenset of categories present in a row -> count

    for events in tqdm(events_col, desc=tier_name):
        if not events:
            continue
        rows_with_events += 1
        cats_in_row = set()
        for event in events:
            cat = event.get("category")
            if cat is None:
                continue
            event_counts[cat] += 1
            try:
                dur = float(event["end"]) - float(event["start"])
                if dur > 0:
                    frame_coverage[cat] += dur
            except (KeyError, ValueError, TypeError):
                pass
            cats_in_row.add(cat)
        if len(cats_in_row) > 1:
            rows_with_multiple_categories += 1
        co_occurrence[frozenset(cats_in_row)] += 1

    print(f"\nRows with >=1 event: {rows_with_events}/{len(indices)}", flush=True)
    print(f"Rows with >1 distinct category present: {rows_with_multiple_categories} "
          f"({100*rows_with_multiple_categories/max(1,rows_with_events):.1f}% of rows-with-events)",
          flush=True)

    print("\nPer-category event counts:", flush=True)
    for cat in CATEGORY_LIST:
        print(f"  {cat:22s}: {event_counts.get(cat, 0):6d} events, "
              f"{frame_coverage.get(cat, 0.0):9.1f}s total coverage", flush=True)
    unknown_cats = set(event_counts) - set(CATEGORY_LIST)
    if unknown_cats:
        print(f"  UNEXPECTED category values found: {unknown_cats}", flush=True)

    print("\nTop 10 co-occurring category sets (degenerate if dominated by one big set):", flush=True)
    for catset, count in co_occurrence.most_common(10):
        label = "{" + ", ".join(sorted(catset)) + "}" if catset else "{} (no category / all None)"
        print(f"  {label}: {count} rows", flush=True)

    # Explicit go/no-go signal
    nonzero_cats = [c for c in CATEGORY_LIST if event_counts.get(c, 0) > 0]
    print(f"\n{tier_name} GO/NO-GO: {len(nonzero_cats)}/7 categories have >0 events.", flush=True)
    if rows_with_events > 0:
        largest_single_set_frac = co_occurrence.most_common(1)[0][1] / rows_with_events
        print(f"Largest single co-occurrence pattern covers {100*largest_single_set_frac:.1f}% of "
              f"rows-with-events (near 100% would indicate degenerate/collapsed co-firing).", flush=True)


def main():
    full_dataset = load_full_dataset().cast_column("audio", Audio(decode=False))
    qualities = full_dataset["annotationQuality"]
    gold_indices = [i for i, q in enumerate(qualities) if q == "verified_timestamps"]
    silver_indices = [i for i, q in enumerate(qualities) if q == "unverified_timestamps"]
    print(f"Gold (verified_timestamps): {len(gold_indices)} rows", flush=True)
    print(f"Silver (unverified_timestamps): {len(silver_indices)} rows", flush=True)

    analyze_tier(full_dataset, gold_indices, "GOLD")
    analyze_tier(full_dataset, silver_indices, "SILVER")


if __name__ == "__main__":
    main()
