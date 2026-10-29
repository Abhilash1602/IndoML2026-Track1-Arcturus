"""Gate for StandaloneStrongSED (plan Step 2) before any GPU run.

Checks per backbone: output (B, 1000); frozen by default (only the head trains);
unfreeze_backbone() makes gradients reach the backbone; separate_params() gives 12
groups covering every backbone parameter exactly once; state_dict round-trips
strictly; tokens_to_frames centres 40 ms tokens at frame 4t + 1.5.

Usage: python -m temp.test_standalone_strong [ATST-F BEATs ASIT]
"""
import sys

import torch

from src.architecture import StandaloneStrongSED, tokens_to_frames
from src.config import Config

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def main():
    torch.manual_seed(0)
    x = torch.arange(250).float().view(1, 250, 1)
    f = tokens_to_frames(x, 4, 1000)[0, :, 0]
    check("tokens_to_frames: token 10 centred at frame 41.5", abs(float(f[41] + f[42]) / 2 - 10) < 1e-5,
          f"({float(f[41]):.3f}/{float(f[42]):.3f})")
    check("tokens_to_frames: clamped at both ends", float(f[0]) == 0.0 and float(f[-1]) == 249.0)

    for name in (sys.argv[1:] or ["ATST-F", "BEATs", "ASIT"]):
        config = Config()
        config.STRONG_BACKBONE = name
        model = StandaloneStrongSED(config).train()
        audio = 0.1 * torch.randn(2, 160000)
        out = model(audio)
        check(f"{name}: output (2, 1000)", tuple(out.shape) == (2, 1000), str(tuple(out.shape)))
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        check(f"{name}: frozen by default (no backbone param trainable)",
              not any(n.startswith("backbone.") for n in trainable))
        out.sum().backward()
        check(f"{name}: head gets gradients", model.temporal_proj.weight.grad is not None)

        model.unfreeze_backbone()
        model.zero_grad()
        model(audio).sum().backward()
        bb = list(model.backbone_parameters())
        n_grad = sum(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in bb)
        check(f"{name}: unfrozen backbone gets gradients", n_grad > 0.5 * len(bb), f"({n_grad}/{len(bb)})")
        groups = model.separate_params()
        ids = [id(p) for g in groups for p in g]
        check(f"{name}: 12 LLRD groups cover every backbone param once",
              len(groups) == 12 and len(ids) == len(set(ids)) == len(bb) and set(ids) == {id(p) for p in bb},
              f"(groups={len(groups)} grouped={len(ids)} backbone={len(bb)})")

        sd = model.state_dict()
        clone = StandaloneStrongSED(config)
        missing, unexpected = clone.load_state_dict(sd, strict=True)
        check(f"{name}: state_dict round-trips strictly", not missing and not unexpected)

    print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else " -- StandaloneStrongSED gate PASSED"))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
