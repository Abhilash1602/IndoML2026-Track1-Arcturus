"""Gate for plan Step 3 before any GPU run.

1. Refactor regression: DualBranchSED._cnn_frames and Standalone*.frame_features
   extraction must reproduce the pre-refactor outputs of load-bearing checkpoints
   exactly (snapshot: scratchpad/refactor_reference_outputs.pt).
2. FusionSED per branch kind: output (B, 1000); branch frozen by default, CNN+head
   trainable; load_branch_from() takes a fine-tuned Standalone Stage-2 checkpoint
   strictly and reproduces its branch features; unfreeze_branch() lets gradients reach
   the backbone; 12 LLRD groups cover the backbone exactly; state_dict round-trips.

Usage: python -m temp.test_fusion
"""
import torch

from src.architecture import DualBranchSED, FusionSED, StandaloneBEATsSED, StandaloneDashengSED
from src.config import Config

REF = "./work/refactor_reference_outputs.pt"
BIG = dict(CNN_CH=(48, 64, 128, 256), CONF_DIM=256, CONF_HEADS=8, CONF_LAYERS=6)
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def cfg(**kw):
    c = Config()
    c.DEVICE = "cpu"
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def load(cls, path, **kw):
    m = cls(cfg(**kw))
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def main():
    ref = torch.load(REF)
    x = ref["input"]
    specs = {
        "phaseh": (DualBranchSED, "./checkpoints/dualbranch_phaseh.pth", BIG),
        "atst_stage2": (DualBranchSED, "./checkpoints/dualbranch_atst_frame_stage2.pth", dict(USE_ATST_FRAME=True)),
        "dasheng_owB": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth", {}),
        "beats_owA": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceA.pth", {}),
    }
    for name, (cls, path, kw) in specs.items():
        with torch.no_grad():
            out = load(cls, path, **kw)(x)
        check(f"refactor: {name} output unchanged", torch.equal(out, ref[name]),
              f"max|diff|={float((out - ref[name]).abs().max()):.2e}")

    torch.manual_seed(0)
    audio = 0.1 * torch.randn(2, 160000)
    cases = [("dasheng", {}, StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth"),
             ("beats", dict(BEATS_FREQ_RESHAPE=True), None, None),
             ("strong", dict(STRONG_BACKBONE="BEATs"), None, None)]
    for kind, kw, src_cls, src_path in cases:
        model = FusionSED(cfg(FUSION_BRANCH=kind, **kw)).train()
        out = model(audio)
        check(f"{kind}: output (2, 1000)", tuple(out.shape) == (2, 1000), str(tuple(out.shape)))
        bb = list(model.branch_parameters())
        check(f"{kind}: branch frozen by default", not any(p.requires_grad for p in bb))
        check(f"{kind}: CNN + head trainable",
              model.stem[0].weight.requires_grad and model.temporal_proj.weight.requires_grad)
        if src_path:
            src = load(src_cls, src_path)
            model.load_branch_from(src.state_dict())
            model.eval()
            with torch.no_grad():
                same = torch.allclose(model.branch.frame_features(audio), src.frame_features(audio), atol=1e-6)
            check(f"{kind}: load_branch_from reproduces the fine-tuned branch features", same)
            model.train()
        model.unfreeze_branch()
        model.zero_grad()
        model(audio).sum().backward()
        n_grad = sum(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in bb)
        check(f"{kind}: unfrozen branch gets gradients", n_grad > 0.5 * len(bb), f"({n_grad}/{len(bb)})")
        groups = model.separate_params()
        ids = [id(p) for g in groups for p in g]
        check(f"{kind}: 12 LLRD groups cover the backbone exactly",
              len(groups) == 12 and len(ids) == len(set(ids)) and set(ids) == {id(p) for p in bb},
              f"(groups={len(groups)} grouped={len(ids)} backbone={len(bb)})")
        clone = FusionSED(cfg(FUSION_BRANCH=kind, **kw))
        missing, unexpected = clone.load_state_dict(model.state_dict(), strict=True)
        check(f"{kind}: state_dict round-trips strictly", not missing and not unexpected)

    print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else " -- fusion gate PASSED"))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
