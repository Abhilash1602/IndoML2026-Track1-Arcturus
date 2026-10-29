"""Oct 14, residual Step 0 input: the 10 s (padded) waveforms of Gold held-out + validation in
work/refiner_<set>.npz order (asserted) -> work/residual/<set>_audio.npy (N, 160000) int16."""
import numpy as np
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import OfficialValidationDataset
from train import make_gold_loaders

VAL_ROOT = "./data/official_validation_raw/validation"


def main():
    config = Config()
    _, _, gold_val_loader = make_gold_loaders(config)
    sets = {"gold_heldout": gold_val_loader.dataset,
            "official_natural": OfficialValidationDataset(VAL_ROOT, config, subset="natural"),
            "official_synthetic": OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")}
    for s, ds in sets.items():
        ref = [str(x) for x in np.load(f"./work/refiner_{s}.npz")["ids"]]
        audio, ids = [], []
        for a, _, cid in DataLoader(ds, batch_size=32, shuffle=False, num_workers=config.NUM_WORKERS):
            audio.append((a.clamp(-1, 1).numpy() * 32767).astype(np.int16)); ids += [str(c) for c in cid]
        assert ids == ref, s
        np.save(f"./work/residual/{s}_audio.npy", np.concatenate(audio))
        print(s, len(ids), flush=True)


if __name__ == "__main__":
    main()
