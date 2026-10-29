"""Fast data-sanity check: is real audio actually reaching the model, and are
ground-truth masks non-degenerate? Run this BEFORE any more training.

Usage: python -m temp.diagnose_pipeline
"""
import torch
from datasets import Audio

from src.config import Config
from src.dataset import VaaniDataset, load_full_dataset


def main():
    config = Config()
    print(f"Device: {config.DEVICE}")

    full_dataset = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold_dataset = full_dataset.filter(lambda row: row["annotationQuality"] == "verified_timestamps")
    split = gold_dataset.train_test_split(test_size=0.2, seed=42)
    train_hf = split["train"].cast_column("audio", Audio(decode=True))
    train_hf_no_decode = split["train"].cast_column("audio", Audio(decode=False))

    dataset = VaaniDataset(train_hf, train_hf_no_decode, config)
    n_samples = min(200, len(dataset))
    print(f"Inspecting {n_samples} / {len(dataset)} training samples...\n")

    zero_audio_count = 0
    zero_mask_count = 0
    audio_abs_means = []
    mask_positive_fractions = []
    captured_errors = 0

    for idx in range(n_samples):
        audio, mask, clip_id = dataset[idx]

        audio_abs_mean = audio.abs().mean().item()
        audio_abs_means.append(audio_abs_mean)
        if audio_abs_mean < 1e-6:
            zero_audio_count += 1
            print(f"  [DEGENERATE AUDIO] idx={idx} clip_id={clip_id} audio.abs().mean()={audio_abs_mean:.2e}")
            if captured_errors < 5:
                # Bypass VaaniDataset's silent except to see the REAL decode error.
                try:
                    row = train_hf[idx]
                    _ = row["audio"]["array"]
                    print(f"    -> no exception raised on direct access?? array dtype/shape unclear")
                except Exception as decode_error:
                    print(f"    -> REAL ERROR: {type(decode_error).__name__}: {decode_error}")
                captured_errors += 1

        positive_fraction = mask.mean().item()
        mask_positive_fractions.append(positive_fraction)
        if positive_fraction == 0.0:
            zero_mask_count += 1

    audio_abs_means_t = torch.tensor(audio_abs_means)
    mask_fractions_t = torch.tensor(mask_positive_fractions)

    print("\n===== SUMMARY =====")
    print(f"Samples with all-zero/near-zero audio: {zero_audio_count} / {n_samples} "
          f"({100 * zero_audio_count / n_samples:.1f}%)")
    print(f"Samples with all-zero (no event) mask: {zero_mask_count} / {n_samples} "
          f"({100 * zero_mask_count / n_samples:.1f}%)")
    print(f"Audio abs().mean() across samples: min={audio_abs_means_t.min():.4f}, "
          f"mean={audio_abs_means_t.mean():.4f}, max={audio_abs_means_t.max():.4f}")
    print(f"Mask positive-frame fraction: min={mask_fractions_t.min():.4f}, "
          f"mean={mask_fractions_t.mean():.4f}, max={mask_fractions_t.max():.4f}")

    print("\n===== VERDICT =====")
    if zero_audio_count / n_samples > 0.1:
        print(">> BUG CONFIRMED: a large fraction of samples have degenerate (all-zero) audio.")
        print(">> This means VaaniDataset.__getitem__ is silently falling back to torch.zeros()")
        print(">> for many samples (audio decode is failing). The model cannot learn from silent")
        print(">> input paired with real labels — this fully explains the F1/Dice collapse.")
        print(">> Fix the audio decode path (check the except-Exception fallback's actual error,")
        print(">> HF cache integrity, soundfile/datasets versions) before any further training.")
    elif audio_abs_means_t.mean().item() < 1e-3:
        print(">> SUSPICIOUS: audio signal is present but unusually quiet across the board.")
        print(">> Worth double-checking normalization / resampling upstream.")
    elif mask_fractions_t.mean().item() < 0.01:
        print(">> Ground-truth masks are almost entirely silent (near-zero positive-frame rate).")
        print(">> This alone would not fully explain a total collapse, but it makes class")
        print(">> imbalance far more severe than expected — worth checking the timestamp")
        print(">> parsing / units in VaaniDataset.__getitem__ against a few raw samples.")
    else:
        print(">> Audio and masks both look reasonable. The collapse is likely NOT a data-loading")
        print(">> bug — look instead at training dynamics (LR too high/low, gradient issues,")
        print(">> or a genuine optimization collapse) with a print of raw sigmoid probability")
        print(">> stats (min/mean/max) partway through training.")


if __name__ == "__main__":
    main()
