"""Gate for OWBCE (src/loss.py boundary_weights + SEDLoss._bce) before any GPU run.

Usage: python -m temp.test_owbce
"""
import torch

from src.config import Config
from src.loss import SEDLoss, boundary_weights

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def main():
    torch.manual_seed(0)
    T = 200
    mask = torch.zeros(1, T)
    mask[0, 50:120] = 1.0  # onset frame 50, offset -> first silent frame 120

    w = boundary_weights(mask, height=4.0, half_width_frames=10)[0]
    check("shape preserved", w.shape == (T,))
    check("peak at onset frame", abs(w[50].item() - 5.0) < 1e-5, f"w[50]={w[50]:.4f}")
    check("peak at offset frame", abs(w[120].item() - 5.0) < 1e-5, f"w[120]={w[120]:.4f}")
    check("symmetric around onset", torch.allclose(w[40:50], w[51:61].flip(0), atol=1e-6))
    check("window support is +-10 frames",
          w[39].item() == 1.0 and w[40].item() > 1.0 and w[60].item() > 1.0 and w[61].item() == 1.0,
          f"w[39]={w[39]:.4f} w[40]={w[40]:.4f} w[60]={w[60]:.4f} w[61]={w[61]:.4f}")
    check("interior of event is weight 1", w[80].item() == 1.0)
    check("far silence is weight 1", w[0].item() == 1.0 and w[199].item() == 1.0)

    edge = torch.zeros(1, T)
    edge[0, :30] = 1.0
    edge[0, 180:] = 1.0
    we = boundary_weights(edge, 4.0, 10)[0]
    check("clip-edge event: no weight at frame 0 / last frame",
          we[0].item() == 1.0 and we[199].item() == 1.0 and abs(we[30].item() - 5.0) < 1e-5
          and abs(we[180].item() - 5.0) < 1e-5)

    close = torch.zeros(1, T)
    close[0, 50:53] = 1.0  # boundaries 3 frames apart: responses overlap, must be clamped
    wc = boundary_weights(close, 4.0, 10)[0]
    check("overlapping boundaries clamped to 1+height", wc.max().item() <= 5.0 + 1e-6, f"max={wc.max():.4f}")

    # Height=0 path must be byte-identical to the original unweighted SEDLoss.
    config = Config()
    base = SEDLoss(config)
    logits = torch.randn(4, T)
    targets = (torch.rand(4, T) > 0.7).float()
    ref = (config.LOSS_BCE_W * torch.nn.BCEWithLogitsLoss()(logits, targets)
           + config.LOSS_DICE_W * (1 - (2 * (torch.sigmoid(logits) * targets).sum(-1) + 1e-5)
                                   / (torch.sigmoid(logits).sum(-1) + targets.sum(-1) + 1e-5)).mean()
           + config.LOSS_BND_W * torch.nn.functional.mse_loss(
               (torch.sigmoid(logits)[:, 1:] - torch.sigmoid(logits)[:, :-1]).abs(),
               (targets[:, 1:] - targets[:, :-1]).abs()))
    check("height=0 reproduces original SEDLoss exactly", torch.equal(base(logits, targets), ref),
          f"{base(logits, targets).item():.8f} vs {ref.item():.8f}")

    # Normalisation: with a uniform per-frame BCE, the weighted mean equals the plain mean.
    config.OWBCE_HEIGHT, config.OWBCE_HALF_WIDTH = 12.0, 20
    ow = SEDLoss(config)
    real_mask = torch.zeros(4, T)
    real_mask[:, 60:140] = 1.0
    uniform_logits = torch.where(real_mask > 0, torch.full_like(real_mask, 1.0), torch.full_like(real_mask, -1.0))
    check("weighted BCE keeps scale (uniform error -> same value)",
          torch.allclose(ow._bce(uniform_logits, real_mask), base._bce(uniform_logits, real_mask), atol=1e-6))

    # Gradient concentrates near boundaries.
    lg = torch.zeros(1, T, requires_grad=True)
    ow._bce(lg, mask).backward()
    g = lg.grad.abs()[0]
    near = g[45:56].mean().item()
    far = g[75:95].mean().item()
    check("gradient concentrated near boundaries (A: h4 w10 / B: h12 w20)", near > 5 * far,
          f"near={near:.2e} far={far:.2e} ratio={near / far:.1f}")

    # Mixup produces max-union masks of 0/1; also fine with soft values (no NaN, weights >= 1).
    soft = torch.rand(2, T)
    ws = boundary_weights(soft, 4.0, 10)
    check("soft masks: finite, >=1", torch.isfinite(ws).all().item() and ws.min().item() >= 1.0)

    print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else " -- OWBCE gate PASSED"))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
