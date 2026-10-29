"""Measure Vaani's actual Gold event-duration distribution, to calibrate SEBBs'
absolute-time hyperparameter candidates (step_filter_lengths, merge_thresholds_abs)
instead of guessing or copying DESED's multi-second scale.

Usage: python -m temp.calibrate_sebbs_ranges
"""
import numpy as np
from datasets import Audio

from src.dataset import load_full_dataset


def main():
    full_ds = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold = full_ds.filter(lambda row: row["annotationQuality"] == "verified_timestamps")

    durations = []
    for row in gold:
        for event in row.get("NoiseSubCategoryTimeStamp") or []:
            if "start" not in event or "end" not in event:
                continue
            duration = float(event["end"]) - float(event["start"])
            if duration > 0:
                durations.append(duration)

    durations = np.array(durations)
    print(f"Total real Gold events: {len(durations)}")
    if len(durations) == 0:
        print("No events found — check the filter/field names before trusting anything downstream.")
        return

    percentiles = [10, 25, 50, 75, 90]
    values = np.percentile(durations, percentiles)

    print(f"\nEvent duration distribution (seconds):")
    print(f"  min:    {durations.min():.4f}")
    for p, v in zip(percentiles, values):
        print(f"  p{p:>2}:    {v:.4f}")
    print(f"  max:    {durations.max():.4f}")
    print(f"  mean:   {durations.mean():.4f}")

    print("\nSuggested SEBBs candidates (derived from the above, not guessed):")
    print(f"  step_filter_lengths ~ (p10, p25, p50) = "
          f"({values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f})")
    print(f"  merge_thresholds_abs ~ (p10/2, p25/2, p50/2) = "
          f"({values[0] / 2:.3f}, {values[1] / 2:.3f}, {values[2] / 2:.3f})")


if __name__ == "__main__":
    main()
