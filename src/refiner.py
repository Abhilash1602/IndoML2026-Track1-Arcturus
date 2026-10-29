"""Plan Step 4: boundary refiner -- an Event Proposal Network style decoder
(arXiv:2601.04178, EPN + OOL + IoU losses) placed on top of the frozen ensemble.

Why a refiner instead of a new head inside each backbone: every backbone retrain this
cycle hurt the ensemble, while the one lever that transferred was training directly on
boundaries (OWBCE). Our models only emit a blurry per-frame presence curve and the
Viterbi decode draws the edge where it crosses a threshold; the refiner instead learns,
per frame, the distance back to the onset and forward to the offset, and it sees the
10 ms log-mel spectrogram, i.e. the acoustic evidence of where a sound actually ends.

Inputs per frame: the members' logits (M), their mean logit, and an 80-band log-mel
(per-clip standardised). Outputs per frame: presence, onset and offset logits, and
d_on / d_off in seconds (softplus).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HOP = 0.01


class BoundaryRefiner(nn.Module):
    def __init__(self, n_members, n_mels=80, width=128, gru=128, n_extra=0, arch="gru"):
        """arch="gru" (default, all v7-v11 refiners) or "tf": a 4-layer Transformer encoder
        (width d_model, 4 heads, sinusoidal positions) in place of the BiGRU."""
        super().__init__()
        self.mel_proj = nn.Sequential(nn.Conv1d(n_mels, 64, 3, padding=1), nn.GELU())
        self.n_extra = n_extra  # extra per-frame input channels (e.g. VAD speech probability, member embeddings)
        self.arch = arch
        c_in = 64 + n_members + 1 + n_extra
        layers = []
        for dil in (1, 2, 4, 8):
            layers += [nn.Conv1d(c_in, width, 5, padding=2 * dil, dilation=dil), nn.BatchNorm1d(width), nn.GELU()]
            c_in = width
        self.conv = nn.Sequential(*layers)
        if arch == "tf":
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, 4, 4 * width, 0.1, batch_first=True,
                                                                       norm_first=True), 4)
            self.head = nn.Linear(width, 5)
        else:
            self.gru = nn.GRU(width, gru, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.head = nn.Linear(2 * gru, 5)  # presence, onset, offset, d_on, d_off

    def forward(self, logits, mel, extra=None):
        """logits (B, M, T) member logits; mel (B, 80, T) log-mel; extra (B, E, T) optional.
        Returns dict of (B, T)."""
        mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-5)
        parts = [self.mel_proj(mel), logits, logits.mean(1, keepdim=True)]
        if self.n_extra:
            parts.append(extra)
        x = torch.cat(parts, dim=1)
        x = self.conv(x).transpose(1, 2)
        if self.arch == "tf":
            T, d = x.shape[1], x.shape[2]
            pos = torch.arange(T, device=x.device).float()[:, None]
            div = torch.exp(torch.arange(0, d, 2, device=x.device).float() * (-np.log(10000.0) / d))
            pe = torch.zeros(T, d, device=x.device)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            x = self.tf(x + pe)
        else:
            x, _ = self.gru(x)
        h = self.head(x)
        return {"presence": h[..., 0], "onset": h[..., 1], "offset": h[..., 2],
                "d_on": F.softplus(h[..., 3]), "d_off": F.softplus(h[..., 4])}


def boundary_targets(mask):
    """mask (B, T) {0,1} -> onset/offset impulses and per-frame d_on/d_off (seconds,
    valid only inside events). Onset impulse at an event's first frame, offset impulse
    at the first frame after it (clip edges are not boundaries)."""
    m = mask.float()
    prev = F.pad(m, (1, 0))[:, :-1]
    onset = ((m - prev) > 0.5).float()
    onset[:, 0] = 0
    offset = ((prev - m) > 0.5).float()
    T = m.shape[1]
    t = torch.arange(T, device=m.device).float()
    d_on = torch.zeros_like(m)
    d_off = torch.zeros_like(m)
    mk = m.cpu().numpy() > 0.5
    for b in range(m.shape[0]):
        row = mk[b]
        i = 0
        while i < T:
            if row[i]:
                j = i
                while j < T and row[j]:
                    j += 1
                d_on[b, i:j] = (t[i:j] - i) * HOP + HOP / 2
                d_off[b, i:j] = (j - t[i:j]) * HOP - HOP / 2
                i = j
            else:
                i += 1
    return onset, offset, d_on, d_off


def refiner_loss(out, mask, targets, ool_weight=10.0, iou_weight=1.0, gamma=2.0):
    """BCE on presence + focal onset/offset loss (OOL, gamma 2) + IoU between each
    in-event frame's proposal [t - d_on, t + d_off] and its true event. `targets` =
    boundary_targets(mask), precomputed once per dataset (the per-frame loop is slow)."""
    onset, offset, d_on, d_off = targets
    m = mask.float()
    bce = F.binary_cross_entropy_with_logits(out["presence"], m)

    def focal(logit, y):
        p = torch.sigmoid(logit)
        p_t = p * y + (1 - p) * (1 - y)
        return -((1 - p_t) ** gamma * torch.log(p_t.clamp_min(1e-8))).mean()

    ool = focal(out["onset"], onset) + focal(out["offset"], offset)
    inside = m > 0.5
    if inside.any():
        p_on, p_off = out["d_on"][inside], out["d_off"][inside]
        g_on, g_off = d_on[inside], d_off[inside]
        inter = torch.minimum(p_on, g_on) + torch.minimum(p_off, g_off)
        union = torch.maximum(p_on, g_on) + torch.maximum(p_off, g_off)
        iou = (1 - inter / union.clamp_min(1e-6)).mean()
    else:
        iou = out["d_on"].sum() * 0
    return bce + ool_weight * ool + iou_weight * iou, {"bce": bce.item(), "ool": ool.item(), "iou": iou.item()}


def decode_proposals(presence, d_on, d_off, thr=0.5, max_events=30):
    """Paper's Algorithm 1 (single class): rank frames above thr by presence, greedily
    accept non-overlapping proposals [t - d_on, t + d_off]. Arrays (T,), seconds."""
    T = len(presence)
    order = np.argsort(-presence)
    events = []
    for t in order:
        if presence[t] < thr or len(events) >= max_events:
            break
        on = max(0.0, (t + 0.5) * HOP - d_on[t])
        off = min(T * HOP, (t + 0.5) * HOP + d_off[t])
        if off - on < HOP:
            continue
        if all(off <= a or on >= b for a, b in events):
            events.append((on, off))
    return sorted(events)


def refine_segments(segments, d_on, d_off):
    """Hybrid decode: keep the (Viterbi) segmentation, move each segment's edges to the
    median of its frames' regressed onset/offset positions."""
    out = []
    T = len(d_on)
    for on, off in segments:
        a, b = int(round(on / HOP)), max(int(round(off / HOP)), int(round(on / HOP)) + 1)
        idx = np.arange(a, min(b, T))
        if len(idx) == 0:
            out.append((on, off))
            continue
        centres = (idx + 0.5) * HOP
        new_on = float(np.median(centres - d_on[idx]))
        new_off = float(np.median(centres + d_off[idx]))
        if new_off - new_on >= HOP:
            out.append((max(0.0, new_on), min(T * HOP, new_off)))
        else:
            out.append((on, off))
    return out
