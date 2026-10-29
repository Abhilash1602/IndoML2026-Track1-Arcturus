import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.dataset import load_full_dataset  # noqa: F401 - re-exported for train_baseline.py convenience


class VaaniDataset(Dataset):
    """Verbatim transcription of the notebook's VaaniDataset (no noise augmentation,
    total_frames = int(max_samples/HOP_LENGTH) + 1, matching the un-cropped mel length)."""

    def __init__(self, hf_dataset, hf_dataset_no_decode, config):
        self.dataset = hf_dataset
        self.dataset_no_decode = hf_dataset_no_decode
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            row = self.dataset[idx]
            audio = torch.tensor(row["audio"]["array"], dtype=torch.float32)
        except Exception:
            audio = torch.zeros(int(self.config.MAX_DURATION * self.config.SR), dtype=torch.float32)
            row = self.dataset_no_decode[idx]

        max_samples = int(self.config.MAX_DURATION * self.config.SR)

        if len(audio) < max_samples:
            audio = F.pad(audio, (0, max_samples - len(audio)))
        else:
            audio = audio[:max_samples]

        total_frames = int(max_samples / self.config.HOP_LENGTH) + 1
        mask = torch.zeros(total_frames, dtype=torch.float32)

        if row.get("NoiseSubCategoryTimeStamp"):
            for event in row["NoiseSubCategoryTimeStamp"]:
                if "start" in event and "end" in event:
                    start_frame = int((float(event["start"]) * self.config.SR) / self.config.HOP_LENGTH)
                    end_frame = int((float(event["end"]) * self.config.SR) / self.config.HOP_LENGTH)
                    start_frame = min(max(start_frame, 0), total_frames - 1)
                    end_frame = min(max(end_frame, 0), total_frames - 1)
                    if start_frame < end_frame:
                        mask[start_frame:end_frame] = 1.0

        clip_id = row.get("imageFileName", f"clip_{idx}")
        return audio, mask, clip_id
