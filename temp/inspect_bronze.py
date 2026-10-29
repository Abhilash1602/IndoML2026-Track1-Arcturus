"""Inspect what fields Bronze-tier (annotationQuality == "no_timestamps") rows
actually carry, beyond raw audio, before deciding how to treat them in the
pseudo-labeling pool. Also samples a few Silver (unverified_timestamps) rows
for comparison.

Usage: python -m temp.inspect_bronze
"""
from datasets import Audio

from src.dataset import load_full_dataset


def show_samples(dataset, tier_name, n=3):
    print(f"\n===== {tier_name} (annotationQuality == tier) — {n} sample rows =====")
    count = 0
    for row in dataset:
        if row["annotationQuality"] != tier_name:
            continue
        print(f"\n--- sample {count} ---")
        for key, value in row.items():
            if key == "audio":
                continue  # skip raw bytes/array, just show it's present
            preview = str(value)
            if len(preview) > 200:
                preview = preview[:200] + "...(truncated)"
            print(f"  {key}: {preview}")
        count += 1
        if count >= n:
            break
    if count == 0:
        print(f"  (no rows found with annotationQuality == {tier_name!r})")


def main():
    full_dataset = load_full_dataset().cast_column("audio", Audio(decode=False))
    print(f"Total rows in dataset: {len(full_dataset)}")

    qualities = full_dataset["annotationQuality"]
    from collections import Counter
    counts = Counter(qualities)
    print(f"annotationQuality distribution: {dict(counts)}")

    show_samples(full_dataset, "no_timestamps", n=3)
    show_samples(full_dataset, "unverified_timestamps", n=3)


if __name__ == "__main__":
    main()
