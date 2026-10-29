"""Gate for the two-convention (DUAL_HEAD) model before any GPU run.

1. Flag off: existing checkpoints give byte-identical outputs (snapshot from the
   fusion refactor, scratchpad/refactor_reference_outputs.pt).
2. Flag on, per class (Dasheng, BEATs): a single-head checkpoint warm-starts with the
   synthetic head copied from the natural one (so at step 0 both heads equal the old
   model exactly); OUTPUT_HEAD nat/syn/both shapes; each head's loss only reaches its
   own head parameters; state_dict round-trips strictly.

Usage: python -m temp.test_dual_head
"""
import torch

from src.architecture import StandaloneBEATsSED, StandaloneDashengSED, load_dual_warm_start
from src.config import Config

REF = "./work/refactor_reference_outputs.pt"
CKPT = {"dasheng": (StandaloneDashengSED, "./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth", "dasheng_owB"),
        "beats": (StandaloneBEATsSED, "./checkpoints/standalone_beats_stage2_owbceA.pth", "beats_owA")}
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


def main():
    ref = torch.load(REF)
    x = ref["input"]
    for kind, (cls, path, ref_key) in CKPT.items():
        sd = torch.load(path, map_location="cpu", weights_only=True)
        single = cls(cfg())
        single.load_state_dict(sd)
        single.eval()
        with torch.no_grad():
            out = single(x)
        check(f"{kind}: flag off reproduces the pre-change output exactly", torch.equal(out, ref[ref_key]),
              f"max|diff|={float((out - ref[ref_key]).abs().max()):.2e}")

        dual = cls(cfg(DUAL_HEAD=True, OUTPUT_HEAD="both"))
        copied = load_dual_warm_start(dual, sd)
        dual.eval()
        with torch.no_grad():
            both = dual(x)
        check(f"{kind}: warm start copies the head", copied)
        check(f"{kind}: OUTPUT_HEAD=both -> (B, T, 2)", tuple(both.shape) == (2, 1000, 2), str(tuple(both.shape)))
        check(f"{kind}: at step 0 both heads equal the single-head model",
              torch.equal(both[..., 0], ref[ref_key]) and torch.equal(both[..., 1], ref[ref_key]))
        dual.output_head = "syn"
        with torch.no_grad():
            syn = dual(x)
        dual.output_head = "nat"
        with torch.no_grad():
            nat = dual(x)
        check(f"{kind}: OUTPUT_HEAD nat/syn give (B, T)", tuple(nat.shape) == tuple(syn.shape) == (2, 1000))

        dual.output_head = "both"
        dual.train()
        dual.zero_grad()
        dual(x)[..., 0].sum().backward()
        g_nat = dual.classifier[0].weight.grad
        g_syn = dual.classifier_syn[0].weight.grad
        check(f"{kind}: natural loss reaches only the natural head",
              g_nat is not None and float(g_nat.abs().sum()) > 0 and (g_syn is None or float(g_syn.abs().sum()) == 0))
        clone = cls(cfg(DUAL_HEAD=True, OUTPUT_HEAD="both"))
        missing, unexpected = clone.load_state_dict(dual.state_dict(), strict=True)
        check(f"{kind}: dual state_dict round-trips strictly", not missing and not unexpected)

    print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else " -- dual-head gate PASSED"))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
