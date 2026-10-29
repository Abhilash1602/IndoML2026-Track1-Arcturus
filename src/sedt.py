"""SEDT-style event-set detector (plan Step 2): DETR for 1-D sound events.

Encoder memory = pretrained Dasheng tokens (25 Hz, 768-d, frozen/cached) fused with a 10 ms
log-mel CNN stem downsampled to the same 250 steps. A decoder with N learned queries emits, per
query, an event logit and a (centre, width) box on the 10 s clip. Trained with one-to-one
Hungarian matching against the reference events -- the same one-to-one structure as the
official Event F1. The L1 term is scaled by each reference's scorer tolerance
max(0.2*dur, 0.05) s so short events' edges get the weight the metric gives them.

At inference its events are only used as cut positions inside the base (v6) regions
(src/event_lock.lock_partition), so Segment Dice is untouched.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

CLIP = 10.0
W_CLS, W_L1, W_TOL, W_GIOU = 1.0, 5.0, 1.0, 2.0
NO_EVENT_WEIGHT = 0.1


def sinusoid(n, d):
    pos = torch.arange(n).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class SEDT(nn.Module):
    def __init__(self, tok_dim=768, n_mels=80, d=256, n_queries=10, enc_layers=3, dec_layers=3, heads=8, steps=250,
                 anchors=False):
        """anchors=True (DAB-DETR style): each query owns a learned reference (centre, width),
        initialised spread evenly over the clip; its box is sigmoid(reference logit + predicted
        delta) and its query carries the sinusoidal embedding of the reference centre -- the
        standard fix for DETR's slow early convergence."""
        super().__init__()
        self.anchors = anchors
        if anchors:
            c = torch.linspace(0.5 / n_queries, 1 - 0.5 / n_queries, n_queries)
            self.ref = nn.Parameter(torch.stack([torch.logit(c), torch.full_like(c, math.log(0.05 / 0.95))], -1))
            self.ref_embed = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.register_buffer("pos_table", sinusoid(1001, d), persistent=False)

        def block(ci, co, stride):
            return [nn.Conv1d(ci, co, 5, stride=stride, padding=2), nn.BatchNorm1d(co), nn.GELU()]

        self.mel_stem = nn.Sequential(*block(n_mels, 128, 1), *block(128, 128, 2), *block(128, 128, 2))
        self.tok_norm = nn.LayerNorm(tok_dim)
        self.tok_proj = nn.Linear(tok_dim, 128)
        self.fuse = nn.Linear(256, d)
        self.register_buffer("pos", sinusoid(steps, d), persistent=False)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), enc_layers)
        self.dec_layers = nn.ModuleList(
            nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True) for _ in range(dec_layers))
        self.dec_norm = nn.LayerNorm(d)
        self.query = nn.Embedding(n_queries, d)
        self.cls = nn.Linear(d, 1)
        self.box = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        nn.init.constant_(self.cls.bias, -2.0)

    def forward(self, tok, mel):
        """tok (B, Tt, 768); mel (B, 80, 1000) log-mel -> list over decoder layers of
        (logits (B, Q), boxes (B, Q, 2) = sigmoid (centre, width) as fractions of the clip)."""
        mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-5)
        m = self.mel_stem(mel).transpose(1, 2)  # (B, 250, 128)
        t = self.tok_proj(self.tok_norm(tok))
        n = min(m.shape[1], t.shape[1])
        x = self.fuse(torch.cat([m[:, :n], t[:, :n]], -1)) + self.pos[:n]
        memory = self.encoder(x)
        q = self.query.weight[None].expand(x.shape[0], -1, -1)
        if self.anchors:
            centre = torch.sigmoid(self.ref[:, 0])
            q = q + self.ref_embed(self.pos_table[(centre * 1000).long().clamp(0, 1000)])[None]
        h = torch.zeros_like(q)
        outs = []
        for layer in self.dec_layers:
            h = layer(h + q, memory)
            z = self.dec_norm(h)
            delta = self.box(z)
            outs.append((self.cls(z).squeeze(-1), torch.sigmoid(delta + self.ref[None] if self.anchors else delta)))
        return outs


def to_onoff(boxes):
    """(..., 2) sigmoid (centre, width) -> (..., 2) (onset, offset) seconds."""
    c, w = boxes[..., 0] * CLIP, boxes[..., 1] * CLIP
    return torch.stack([(c - w / 2).clamp(0, CLIP), (c + w / 2).clamp(0, CLIP)], -1)


def giou_1d(a, b):
    """a (Q, 2), b (R, 2) onset/offset -> (Q, R) generalised IoU."""
    on = torch.maximum(a[:, None, 0], b[None, :, 0])
    off = torch.minimum(a[:, None, 1], b[None, :, 1])
    inter = (off - on).clamp_min(0)
    la, lb = (a[:, 1] - a[:, 0]).clamp_min(0), (b[:, 1] - b[:, 0]).clamp_min(0)
    union = la[:, None] + lb[None] - inter
    hull = torch.maximum(a[:, None, 1], b[None, :, 1]) - torch.minimum(a[:, None, 0], b[None, :, 0])
    iou = inter / union.clamp_min(1e-6)
    return iou - (hull - union) / hull.clamp_min(1e-6)


def pair_costs(pred, ref):
    """pred (Q, 2), ref (R, 2) seconds -> (normalised L1, tolerance-scaled smooth L1, 1 - GIoU), each (Q, R)."""
    d = (pred[:, None] - ref[None]).abs()  # (Q, R, 2)
    tol = torch.clamp(0.2 * (ref[:, 1] - ref[:, 0]), min=0.05)[None, :, None]
    l1 = d.sum(-1) / CLIP
    tl = F.smooth_l1_loss(d / tol, torch.zeros_like(d), reduction="none", beta=1.0).clamp(max=10.0).sum(-1)
    return l1, tl, 1 - giou_1d(pred, ref)


@torch.no_grad()
def hungarian(logits, onoff, refs):
    """Per clip: indices (query_idx, ref_idx) of the optimal one-to-one assignment. Cost
    matrices are built on the device for the whole batch and copied to the CPU once."""
    costs = []
    for lg, pr, rf in zip(logits, onoff, refs):
        if len(rf) == 0:
            costs.append(None)
            continue
        l1, tl, g = pair_costs(pr, rf)
        costs.append(-W_CLS * torch.sigmoid(lg)[:, None] + W_L1 * l1 + W_TOL * tl + W_GIOU * g)
    flat = [c for c in costs if c is not None]
    sizes = [c.numel() for c in flat]
    buf = torch.cat([c.reshape(-1) for c in flat]).float().cpu().numpy() if flat else np.zeros(0)
    out, pos, k = [], 0, 0
    for c in costs:
        if c is None:
            out.append((np.zeros(0, int), np.zeros(0, int)))
            continue
        n = sizes[k]
        out.append(linear_sum_assignment(buf[pos:pos + n].reshape(c.shape)))
        pos += n
        k += 1
    return out


def sedt_loss(outs, refs):
    """outs: model output list; refs: list of (R_i, 2) float tensors (seconds, on the model's
    device). Loss summed over decoder layers (auxiliary decoding losses)."""
    total, parts = 0.0, {}
    for li, (logits, boxes) in enumerate(outs):
        onoff = to_onoff(boxes)
        idx = hungarian(logits, onoff, refs)
        target = torch.zeros_like(logits)
        weight = torch.full_like(logits, NO_EVENT_WEIGHT)
        l1s, tls, gs, n = [], [], [], 0
        for b, (qi, ri) in enumerate(idx):
            if len(qi) == 0:
                continue
            qi_t, ri_t = torch.as_tensor(qi, device=logits.device), torch.as_tensor(ri, device=logits.device)
            target[b, qi_t] = 1.0
            weight[b, qi_t] = 1.0
            l1, tl, g = pair_costs(onoff[b, qi_t], refs[b][ri_t])
            l1s.append(l1.diagonal()); tls.append(tl.diagonal()); gs.append(g.diagonal())
            n += len(qi)
        cls = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / max(n, 1)
        if n:
            l1, tl, g = torch.cat(l1s).sum() / n, torch.cat(tls).sum() / n, torch.cat(gs).sum() / n
        else:
            l1 = tl = g = logits.sum() * 0
        layer = W_CLS * cls + W_L1 * l1 + W_TOL * tl + W_GIOU * g
        total = total + layer
        if li == len(outs) - 1:
            parts = {"cls": float(cls), "l1": float(l1), "tol": float(tl), "giou": float(g)}
    return total, parts


@torch.no_grad()
def predict_events(model, tok, mel):
    """-> per clip list of (onset, offset, confidence) from the last decoder layer, sorted by onset."""
    logits, boxes = model(tok, mel)[-1]
    p, oo = torch.sigmoid(logits).cpu().numpy(), to_onoff(boxes).cpu().numpy()
    out = []
    for pi, ooi in zip(p, oo):
        ev = [(round(float(a), 3), round(float(b), 3), float(c)) for (a, b), c in zip(ooi, pi) if b - a >= 0.02]
        out.append(sorted(ev))
    return out


def select_events(events, thr):
    """Confident, non-overlapping events (greedy by confidence), as (onset, offset)."""
    keep = []
    for a, b, c in sorted(events, key=lambda e: -e[2]):
        if c < thr:
            break
        if all(b <= x or a >= y for x, y in keep):
            keep.append((a, b))
    return sorted(keep)
