"""Fix F investigation: does NoiseCategory actually carry negative examples across
the full Silver+Bronze pool, or is the Bronze clip-tag loss training against an
effectively-always-positive target (which would explain tag_loss collapsing to
~0.0000 by epoch 12 of the last Phase 2 run -- a trivial "always predict positive"
solution, not a signal the model is usefully learning from)?

Usage: python -m temp.inspect_noise_category
"""
from collections import Counter

from datasets import Audio

from src.dataset import load_full_dataset


def main():
    full_dataset = load_full_dataset().cast_column("audio", Audio(decode=False))
    qualities = full_dataset["annotationQuality"]
    pool_indices = [
        i for i, q in enumerate(qualities)
        if q in {"unverified_timestamps", "no_timestamps"}
    ]
    print(f"Silver+Bronze pool size: {len(pool_indices)}")

    tags = full_dataset["NoiseCategory"]
    pool_tags = [tags[i] for i in pool_indices]

    has_tag = sum(1 for t in pool_tags if t)
    no_tag = len(pool_tags) - has_tag
    print(f"has_tag (truthy NoiseCategory): {has_tag} ({100 * has_tag / len(pool_tags):.2f}%)")
    print(f"no_tag  (falsy/empty NoiseCategory): {no_tag} ({100 * no_tag / len(pool_tags):.2f}%)")

    value_counts = Counter(t for t in pool_tags if t)
    print(f"\nDistinct non-empty NoiseCategory values: {len(value_counts)}")
    for value, count in value_counts.most_common(20):
        print(f"  {value!r}: {count}")


if __name__ == "__main__":
    main()
