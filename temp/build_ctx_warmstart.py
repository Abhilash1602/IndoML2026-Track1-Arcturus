"""Warm-start files for the CONTEXT=mae member (Oct 12): the VT dasheng_owB teacher's Dasheng
backbone + (pre-trained | randomly initialised) ContextNet + a fresh classifier, saved as full
state_dicts so temp/train_dasheng_stage2_i2_owbce.py can load them strictly via WARM_START.

Usage: python -m temp.build_ctx_warmstart <pretrained.pth>
"""
import os
import sys

import torch

from src.architecture import StandaloneDashengSED
from src.config import Config

SRC = os.environ.get("CTX_SRC", "./checkpoints/standalone_dasheng_stage2_i2_owbceVT_owB_teacher.pth")
SUFFIX = os.environ.get("CTX_SUFFIX", "")


def main():
    pre = sys.argv[1]
    src = torch.load(SRC, map_location="cpu", weights_only=True)
    os.environ["CONTEXT"], os.environ["CONTEXT_INIT"] = "mae", pre
    for kind, rnd in (("mae", "0"), ("rand", "1")):
        os.environ["CONTEXT_RANDOM"] = rnd
        torch.manual_seed(0)
        m = StandaloneDashengSED(Config())
        missing, unexpected = m.load_state_dict({k: v for k, v in src.items() if k.startswith("dasheng.")}, strict=False)
        assert not unexpected and all(not k.startswith("dasheng.") for k in missing), (unexpected, missing[:5])
        out = f"./checkpoints/ctx_{kind}_warm{SUFFIX}.pth"
        torch.save(m.state_dict(), out)
        print(f"{out}: context {'pre-trained' if rnd == '0' else 'random'}; new keys {len(missing)}", flush=True)


if __name__ == "__main__":
    main()
