"""Gate for the BEATs time-axis fix (Config.BEATS_FREQ_RESHAPE) before any GPU run.

1. Real BEATs: a burst in one 160 ms block must change tokens t*8..t*8+7 of THAT
   block most (confirms speechbrain's time-major flattening, index = t*8 + f).
2. beats_tokens_to_frames: a planted per-token pattern must come out with no
   frequency sawtooth along time, token t centred at 10 ms frame 16t+7.5.
3. StandaloneBEATsSED with the flag: output (B, 1000), gradients reach freq_proj,
   and the flag-off model still loads an old checkpoint strictly.

Usage: python -m temp.test_beats_reshape
"""
import torch

from src.architecture import StandaloneBEATsSED, beats_tokens_to_frames
from src.config import Config

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def main():
    torch.manual_seed(0)
    config = Config()
    config.DEVICE = "cpu"

    # 1. token order on the real model
    model = StandaloneBEATsSED(config).eval()
    quiet = 0.001 * torch.randn(1, 160000)
    loud = quiet.clone()
    t_block = 30  # 160 ms block 30 -> samples [30*16*160, 31*16*160)
    lo, hi = t_block * 16 * 160 + 400, (t_block + 1) * 16 * 160 - 400
    loud[0, lo:hi] += 0.5 * torch.randn(hi - lo)
    with torch.no_grad():
        wl = torch.ones(1)
        a = model.beats(quiet, wl)[0][0]
        b = model.beats(loud, wl)[0][0]
    change = (b - a).norm(dim=-1)  # (496,)
    per_block = change.view(-1, 8).mean(1)  # time-major grouping
    per_block_fmajor = change.view(8, -1).mean(0)  # frequency-major alternative
    check("real BEATs: 496 tokens", change.numel() == 496, f"({change.numel()})")
    check("real BEATs: time-major -- burst block has the largest change",
          int(per_block.argmax()) == t_block,
          f"argmax block={int(per_block.argmax())} (expected {t_block}); "
          f"f-major reading argmax={int(per_block_fmajor.argmax())}")

    # 2. helper on a planted pattern: token value = time index, frequency adds a sawtooth
    T, F_, D = 62, 8, 4
    t_idx = torch.arange(T).float().repeat_interleave(F_)
    f_idx = torch.arange(F_).float().repeat(T)
    feats = torch.stack([t_idx, f_idx, torch.zeros_like(t_idx), torch.zeros_like(t_idx)], -1).unsqueeze(0)
    proj = torch.nn.Linear(F_ * D, D, bias=False)
    with torch.no_grad():
        proj.weight.zero_()
        proj.weight[0, 0] = 1.0  # output ch0 = time value of band 0
        for f in range(F_):
            proj.weight[1, f * D + 1] = 1.0 / F_  # output ch1 = mean frequency index (constant 3.5)
        out = beats_tokens_to_frames(feats, F_, proj, 1000)[0]
    check("helper: output (1000, D)", tuple(out.shape) == (1000, D))
    check("helper: no frequency sawtooth along time (ch1 constant)",
          torch.allclose(out[:, 1], torch.full((1000,), 3.5), atol=1e-5))
    ch0 = out[:, 0]
    check("helper: monotone time channel", bool((ch0[1:] - ch0[:-1] >= -1e-6).all()))
    check("helper: token 10 centred at frame 167.5", abs(float(ch0[167] + ch0[168]) / 2 - 10.0) < 1e-4,
          f"(frames 167/168 -> {float(ch0[167]):.3f}/{float(ch0[168]):.3f})")

    # 3. model with the flag
    config.BEATS_FREQ_RESHAPE = True
    fixed = StandaloneBEATsSED(config).train()
    x = 0.1 * torch.randn(2, 160000)
    out = fixed(x)
    check("model: output (2, 1000)", tuple(out.shape) == (2, 1000), str(tuple(out.shape)))
    out.sum().backward()
    g = fixed.freq_proj.weight.grad
    check("model: gradient reaches freq_proj", g is not None and float(g.abs().sum()) > 0)
    config.BEATS_FREQ_RESHAPE = False
    old = StandaloneBEATsSED(config)
    missing, unexpected = old.load_state_dict(
        torch.load("./checkpoints/standalone_beats_stage2_owbceA.pth", map_location="cpu", weights_only=True))
    check("flag off: old checkpoint loads strictly", not missing and not unexpected)

    print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else " -- BEATs reshape gate PASSED"))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
