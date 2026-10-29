"""Unit checks for src/sedt.py before any real training.

Usage: VAANI_DEVICE=cuda:0 python -m temp.test_sedt
"""
import os

import numpy as np
import torch

from src.sedt import CLIP, SEDT, giou_1d, hungarian, predict_events, select_events, sedt_loss, to_onoff
from temp.official_metric import match_events
from temp.train_refiner import refs_of
from temp.train_refiner_val import S

DEVICE = os.environ.get("VAANI_DEVICE", "cuda:0")


def boxes_for(onoff):
    on, off = onoff[..., 0], onoff[..., 1]
    return torch.stack([(on + off) / 2 / CLIP, (off - on) / CLIP], -1)


def test_giou():
    a = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(giou_1d(a, a), torch.ones(1, 1))
    g = giou_1d(a, torch.tensor([[3.0, 4.0]]))
    assert g.item() < 0 and abs(g.item() - (0 - 1 / 3)) < 1e-6, g
    print("giou OK")


def test_hungarian_and_loss():
    refs = [torch.tensor([[1.0, 1.5], [4.0, 6.0], [7.2, 7.4]])]
    perm = [2, 0, 1]
    onoff = torch.full((1, 10, 2), 0.0)
    onoff[0, :, 1] = 0.5
    for q, r in enumerate(perm):
        onoff[0, q + 3] = refs[0][r]
    logits = torch.full((1, 10), -8.0)
    logits[0, 3:6] = 8.0
    qi, ri = hungarian(logits, onoff, refs)[0]
    got = dict(zip(qi.tolist(), ri.tolist()))
    assert got == {3: 2, 4: 0, 5: 1}, got
    boxes = boxes_for(onoff)
    assert torch.allclose(to_onoff(boxes), onoff, atol=1e-5)
    loss, parts = sedt_loss([(logits, boxes)], refs)
    assert loss.item() < 0.05 and parts["l1"] < 1e-5 and parts["giou"] < 1e-5, (loss, parts)
    # a matched box 0.1 s off costs more than an exact one
    onoff2 = onoff.clone(); onoff2[0, 3, 1] += 0.1
    loss2, _ = sedt_loss([(logits, boxes_for(onoff2))], refs)
    assert loss2 > loss
    print(f"hungarian + loss OK (perfect-prediction loss {loss.item():.4f}, 0.1 s-off {loss2.item():.4f})")


def test_overfit_batch():
    torch.manual_seed(0)
    tok = torch.from_numpy(np.load(f"{S}/sedt/gold_train_tok.npy", mmap_mode="r")[:16].astype(np.float32)).to(DEVICE)
    mel = torch.from_numpy(np.load(f"{S}/sedt/gold_train_mel.npy", mmap_mode="r")[:16].astype(np.float32)).to(DEVICE)
    refs_np = refs_of(np.load(f"{S}/sedt/gold_train_mask.npy", mmap_mode="r")[:16].astype(np.float32))
    refs = [torch.tensor(r, dtype=torch.float32, device=DEVICE).reshape(-1, 2) for r in refs_np]
    model = SEDT().to(DEVICE)
    outs = model(tok, mel)
    assert len(outs) == 3 and outs[-1][0].shape == (16, 10) and outs[-1][1].shape == (16, 10, 2)
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    model.train()
    first = None
    for step in range(1200):
        loss, parts = sedt_loss(model(tok, mel), refs)
        first = loss.item() if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
    # Guards against input collapse (the silent-audio cache bug gave loss ~12, F1 0). The strict
    # scorer tolerance (+-50 ms on short events) keeps a 1,200-step overfit around F1 0.77-0.84.
    assert loss.item() < first / 4, (first, loss.item())
    model.eval()
    tp = fp = fn = 0
    for ev, r in zip(predict_events(model, tok, mel), refs_np):
        a, b, c = match_events(r, select_events(ev, 0.5))
        tp += a; fp += b; fn += c
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"overfit 16 clips: final loss {loss.item():.3f} {parts} | event F1 on the batch {f1:.3f}")
    assert f1 > 0.7, f1


if __name__ == "__main__":
    test_giou()
    test_hungarian_and_loss()
    test_overfit_batch()
    print("SEDT checks passed")
