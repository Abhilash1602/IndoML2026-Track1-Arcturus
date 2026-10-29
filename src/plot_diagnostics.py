import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Audio
from torch.utils.data import DataLoader

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import VaaniDataset, load_full_dataset

BASELINE_PATH = "./checkpoints/dualbranch_baseline.pth"


def load_validation_loader(config):
    full_dataset = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold_dataset = full_dataset.filter(
        lambda row: row["annotationQuality"] == "verified_timestamps"
    )
    split = gold_dataset.train_test_split(test_size=0.2, seed=42)
    validation = split["test"].cast_column("audio", Audio(decode=True))
    validation_raw = split["test"].cast_column("audio", Audio(decode=False))
    dataset = VaaniDataset(validation, validation_raw, config)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )


def main():
    config = Config()
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(
        torch.load(BASELINE_PATH, map_location=config.DEVICE, weights_only=True)
    )
    model.eval()

    validation_loader = load_validation_loader(config)
    audio, masks, _ = next(iter(validation_loader))
    audio = audio.to(config.DEVICE, non_blocking=True)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(audio)).cpu().numpy()
    ground_truth = masks.numpy()

    probability = probabilities[0]
    target = ground_truth[0]
    frames = min(len(probability), len(target))
    probability, target = probability[:frames], target[:frames]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(frames), 0, target, color="red", alpha=0.3, label="Ground truth")
    ax.plot(probability, color="blue", label="Predicted probability")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig("diagnostic_plot.png")


if __name__ == "__main__":
    main()
