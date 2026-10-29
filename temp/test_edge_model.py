"""Unit checks for src/edge_model.py (plan Step 2), run before any real training.

Usage: python -m temp.test_edge_model
"""
import numpy as np
import torch

from src.edge_model import HALF, HOP, WIN, EdgeModel, crop_batch, move_edges, normalise, span_rows


def test_crop_alignment():
    T = 1000
    mel = torch.arange(T, dtype=torch.float32)[None].repeat(80, 1)
    c = torch.tensor([0, 5, 500, 999])
    cr = crop_batch(mel, c, pad_value=-1.0)
    assert cr.shape == (4, 80, WIN)
    for j, cc in enumerate(c.tolist()):
        assert cr[j, 0, HALF] == cc, (cc, cr[j, 0, HALF])
        for k in (0, WIN - 1):
            want = cc - HALF + k
            assert cr[j, 0, k] == (want if 0 <= want < T else -1.0)
    print("crop alignment OK")


def test_span():
    c = torch.tensor([100])
    s = span_rows(c, torch.tensor([90]), torch.tensor([130]))[0]
    pos = 100 - HALF + np.arange(WIN)
    assert torch.equal(s, torch.tensor(((pos >= 90) & (pos < 130)).astype(np.float32)))
    print("span OK")


class Fixed(torch.nn.Module):
    """Puts the argmax at HALF + d_on (onset head) / HALF + d_off (offset head)."""
    n_extra = 0

    def __init__(self, d_on, d_off):
        super().__init__()
        self.d = (d_on, d_off)

    def forward(self, mel, span, extra=None):
        out = torch.zeros(mel.shape[0], 2, WIN)
        out[:, 0, HALF + self.d[0]] = 10.0
        out[:, 1, HALF + self.d[1]] = 10.0
        return out


def test_move_edges():
    mel = np.random.randn(80, 1000).astype(np.float32)
    ev = [(1.0, 1.3), (2.0, 4.0), (5.0, 5.5)]
    out = move_edges(Fixed(3, -5), mel, ev, "cpu")
    assert out[0] == (1.03, 1.25) and out[2] == (5.03, 5.45) and out[1] == (2.0, 4.0), out
    out = move_edges(Fixed(-20, 20), mel, ev, "cpu", min_conf=1.1)  # nothing confident enough
    assert out == ev, out
    print("move_edges OK")


def test_learns_step():
    """A clean energy step at a random position inside the crop must be found to +-1 bin."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = EdgeModel(width=64, gru=64)
    opt = torch.optim.Adam(model.parameters(), 2e-3)

    def batch(n):
        t = rng.integers(20, WIN - 20, n)
        x = np.random.randn(n, 80, WIN).astype(np.float32) * 0.3
        for i in range(n):
            x[i, :, t[i]:] += 3.0
        return normalise(torch.tensor(x)), torch.tensor(t)

    for _ in range(300):
        x, t = batch(64)
        span = (torch.arange(WIN)[None] >= t[:, None]).float()
        loss = torch.nn.functional.cross_entropy(model(x, span * 0 + 0.5)[:, 0], t)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    x, t = batch(256)
    pred = model(x, torch.full((256, WIN), 0.5))[:, 0].argmax(-1)
    acc = (pred - t).abs().le(1).float().mean().item()
    assert acc > 0.9, acc
    print(f"learns a planted step: {acc:.1%} within +-1 bin")


if __name__ == "__main__":
    test_crop_alignment()
    test_span()
    test_move_edges()
    test_learns_step()
    print("all edge-model checks passed")
