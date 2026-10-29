import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import Audio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import compute_event_f1, compute_segment_dice, extract_events_from_mask
from temp.config import Config
from temp.dataset import VaaniDataset, load_full_dataset
from temp.model import ModularSED


CHECKPOINT_PATH = "temp/checkpoints/best_sed_model.pth"
METRICS_PATH = "temp/metrics.json"
CURVES_PLOT_PATH = "temp/plots/training_curves.png"
DIAGNOSTIC_PLOT_PATH = "temp/plots/diagnostic_plot.png"


def load_gold_loaders(config):
    full_ds = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold_data = full_ds.filter(lambda row: row["annotationQuality"] == "verified_timestamps")
    split_ds = gold_data.train_test_split(test_size=0.2, seed=42)

    train_hf = split_ds["train"].cast_column("audio", Audio(decode=True))
    val_hf = split_ds["test"].cast_column("audio", Audio(decode=True))
    train_hf_no_decode = split_ds["train"].cast_column("audio", Audio(decode=False))
    val_hf_no_decode = split_ds["test"].cast_column("audio", Audio(decode=False))

    train_dataset = VaaniDataset(train_hf, train_hf_no_decode, config)
    val_dataset = VaaniDataset(val_hf, val_hf_no_decode, config)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")
    return train_loader, val_loader


def evaluate(model, val_loader, criterion, config):
    model.eval()
    val_loss = 0.0
    all_dice, all_f1 = [], []
    hop_sec = config.HOP_LENGTH / config.SR

    with torch.no_grad():
        for audio, mask, _ in tqdm(val_loader, desc="[Val]"):
            audio, mask = audio.to(config.DEVICE), mask.to(config.DEVICE)
            logits = model(audio)

            min_frames = min(logits.shape[1], mask.shape[1])
            logits, mask = logits[:, :min_frames], mask[:, :min_frames]

            val_loss += criterion(logits, mask).item()

            probs = torch.sigmoid(logits).cpu().numpy()
            gt_masks = mask.cpu().numpy()

            for i in range(probs.shape[0]):
                pred_binary = (probs[i] > 0.5).astype(np.int32)
                gt_binary = gt_masks[i].astype(np.int32)

                all_dice.append(compute_segment_dice(pred_binary, gt_binary))

                pred_events = extract_events_from_mask(pred_binary, hop_sec=hop_sec)
                gt_events = extract_events_from_mask(gt_binary, hop_sec=hop_sec)
                all_f1.append(compute_event_f1(pred_events, gt_events))

    return val_loss / len(val_loader), float(np.mean(all_dice)), float(np.mean(all_f1))


def plot_training_curves(history):
    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, [e["train_loss"] for e in history], label="train_loss")
    axes[0, 0].plot(epochs, [e["val_loss"] for e in history], label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [e["val_dice"] for e in history], color="teal")
    axes[0, 1].set_title("Val Segment Dice")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(epochs, [e["val_f1"] for e in history], color="darkorange")
    axes[1, 0].set_title("Val Event F1")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(epochs, [e["combined_score"] for e in history], color="purple")
    axes[1, 1].set_title("Combined Score (Dice + F1)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0, 2)

    fig.tight_layout()
    fig.savefig(CURVES_PLOT_PATH)
    plt.close(fig)


def plot_diagnostic(model, val_loader, config):
    model.eval()
    audio, mask, _ = next(iter(val_loader))
    audio = audio.to(config.DEVICE)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(audio)).cpu().numpy()
    ground_truth = mask.numpy()

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
    fig.savefig(DIAGNOSTIC_PLOT_PATH)
    plt.close(fig)


def main():
    os.makedirs("temp/checkpoints", exist_ok=True)
    os.makedirs("temp/plots", exist_ok=True)

    config = Config()
    print(f"Hardware utilized: {config.DEVICE}")

    train_loader, val_loader = load_gold_loaders(config)

    model = ModularSED(config).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    best_combined_score = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    history = []

    config_dict = {
        key: value for key, value in vars(Config).items()
        if not key.startswith("__") and not callable(value) and key != "DEVICE"
    }
    config_dict["DEVICE"] = str(config.DEVICE)
    wandb.init(
        project="vaani-track1",
        name="temp-control-modularsed",
        config=config_dict,
        job_type="control_experiment",
    )

    print("Commencing Full Gold Training with Local Validation...")
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            epoch_loss = 0.0
            for audio, mask, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} [Train]"):
                audio, mask = audio.to(config.DEVICE), mask.to(config.DEVICE)

                optimizer.zero_grad()
                logits = model(audio)

                min_frames = min(logits.shape[1], mask.shape[1])
                logits, mask = logits[:, :min_frames], mask[:, :min_frames]

                loss = criterion(logits, mask)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)
            val_loss, avg_dice, avg_f1 = evaluate(model, val_loader, criterion, config)
            combined_score = avg_dice + avg_f1

            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Val Dice:   {avg_dice:.4f} | Val F1: {avg_f1:.4f} | COMBINED: {combined_score:.4f}/2.0")

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": avg_dice,
                "val_f1": avg_f1,
                "combined_score": combined_score,
            })
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/dice": avg_dice,
                "val/f1": avg_f1,
                "val/combined_score": combined_score,
            })

            if combined_score > best_combined_score:
                print(f"  >> New best combined score! ({best_combined_score:.4f} -> {combined_score:.4f}). Saving model.")
                best_combined_score = combined_score
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), CHECKPOINT_PATH)
    finally:
        wandb.finish()

    print(f"\nTraining Complete. Best Validation Score: {best_combined_score:.4f}/2.0")
    model.load_state_dict(best_model_weights)

    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    plot_training_curves(history)
    plot_diagnostic(model, val_loader, config)
    print(f"Plots saved to {CURVES_PLOT_PATH} and {DIAGNOSTIC_PLOT_PATH}")


if __name__ == "__main__":
    main()
