"""Oct 14, verifier v3 inputs: frame logits (10 ms, 1000 frames) of the three clean VG models
(trained on Gold-train + synthetic only, so Gold held-out and the official validation set are
out-of-sample for them) on the same clips and in the same order as work/refiner_<set>.npz.

Output: work/verifier3/<model>_<set>.npy (N, 1000) float16.
Usage: CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 python -m temp.cache_vg_logits
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.architecture import FusionSED, StandaloneDashengSED
from src.config import Config
from src.dataset import OfficialValidationDataset
from train import make_gold_loaders

OUT = "./work/verifier3"
VAL_ROOT = "./data/official_validation_raw/validation"
MODELS = {  # name -> (class, checkpoint, env, config overrides)
    "w2v": (FusionSED, "./checkpoints/fusion_w2vbertVG_stage1_last.pth", {}, dict(FUSION_BRANCH="w2vbert")),
    "mae": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceVG_mae_last_teacher.pth",
            {"CONTEXT": "mae"}, {}),
    "d06": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceVG_d06_last_teacher.pth",
            {"DASHENG_NAME": "mispeech/dasheng-0.6B"}, {}),
}


def build(name):
    cls, ckpt, env, ov = MODELS[name]
    for k in ("CONTEXT", "DASHENG_NAME", "CONTEXT_INIT", "CONTEXT_RANDOM"):
        os.environ.pop(k, None)
    os.environ.update(env)
    config = Config()
    for k, v in ov.items():
        setattr(config, k, v)
    m = cls(config)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return m.to(config.DEVICE).eval(), config


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    config = Config()
    _, _, gold_val_loader = make_gold_loaders(config)
    sets = {"gold_heldout": gold_val_loader.dataset,
            "official_natural": OfficialValidationDataset(VAL_ROOT, config, subset="natural"),
            "official_synthetic": OfficialValidationDataset(VAL_ROOT, config, subset="synthetic")}
    names = [n for n in MODELS if not all(os.path.exists(f"{OUT}/{n}_{s}.npy") for s in sets)]
    for name in names:
        model, cfg = build(name)
        for s, ds in sets.items():
            ref_ids = [str(x) for x in np.load(f"./work/refiner_{s}.npz")["ids"]]
            out, ids = [], []
            for audio, _, cid in DataLoader(ds, batch_size=16, shuffle=False, num_workers=cfg.NUM_WORKERS):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(model(audio.to(cfg.DEVICE)).float()[:, :1000].cpu().numpy())
                ids += [str(c) for c in cid]
            assert ids == ref_ids, f"{s}: clip order differs from work/refiner_{s}.npz"
            np.save(f"{OUT}/{name}_{s}.npy", np.concatenate(out).astype(np.float16))
            print(f"{name} {s}: {len(ids)} clips", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
