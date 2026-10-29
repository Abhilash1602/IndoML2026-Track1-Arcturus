"""Short-event edge model (plan Step 2).

Given a candidate event (onset, offset) from the ensemble decode, look at a 1.2 s crop of the
10 ms log-mel centred on ONE candidate edge and predict where the true edge is, as a softmax
over the crop's 120 frames. Separate onset/offset heads. The candidate span is a second input
row (1 inside the candidate event), so the model knows which side the event is on.

Only edges of short predicted events (< MAX_EVENT s) are moved, so the frame mass that drives
Segment Dice is barely touched while Event F1's strict tolerance (max(0.2*dur, 0.05) s) is the
target.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HOP = 0.01
WIN = 120
HALF = WIN // 2
MAX_EVENT = 0.8


class EdgeModel(nn.Module):
    def __init__(self, n_mels=80, n_extra=0, width=128, gru=128):
        super().__init__()
        self.n_extra = n_extra
        c_in = n_mels + 1 + n_extra
        layers = []
        for i, dil in enumerate((1, 2, 4, 1)):
            layers += [nn.Conv1d(c_in if i == 0 else width, width, 5, padding=2 * dil, dilation=dil),
                       nn.BatchNorm1d(width), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(width, gru, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * gru, 2)

    def forward(self, mel, span, extra=None):
        """mel (B, 80, WIN) normalised per crop; span (B, WIN); extra (B, n_extra, WIN) -> (B, 2, WIN)."""
        x = [mel, span[:, None]] + ([extra] if self.n_extra else [])
        h = self.conv(torch.cat(x, 1)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h).transpose(1, 2)


def normalise(crop):
    """Per-crop standardisation of (B, 80, WIN) log-mel."""
    m = crop.mean(dim=(1, 2), keepdim=True)
    s = crop.std(dim=(1, 2), keepdim=True)
    return (crop - m) / (s + 1e-5)


PAD_DB = -100.0  # AmplitudeToDB floor = log-mel of silence / zero padding


def crop_batch(mel, centres, pad_value=PAD_DB):
    """mel (80, T) tensor; centres (B,) int frame indices -> (B, 80, WIN), padded outside the clip."""
    T = mel.shape[-1]
    padded = F.pad(mel, (HALF, HALF), value=float(pad_value))
    idx = centres.long().clamp(0, T - 1)[:, None] + torch.arange(WIN, device=mel.device)[None]  # shifted by HALF
    return padded[:, idx].permute(1, 0, 2)


def span_rows(centres, onsets, offsets):
    """(B, WIN) indicator of the candidate event [onset, offset) in each crop's frame coordinates."""
    pos = centres[:, None] - HALF + torch.arange(WIN, device=centres.device)[None]
    return ((pos >= onsets[:, None]) & (pos < offsets[:, None])).float()


@torch.no_grad()
def move_edges(model, mel, events, device, max_event=MAX_EVENT, min_conf=0.0, extra=None):
    """mel: (80, T) numpy/tensor log-mel at 10 ms; events: list of (onset, offset) seconds.
    Returns events with the edges of short ones moved to the model's argmax (kept if the
    result would be shorter than 0.02 s or the argmax probability is below min_conf)."""
    short = [i for i, (a, b) in enumerate(events) if b - a < max_event]
    if not short:
        return list(events)
    mel = torch.as_tensor(np.asarray(mel, np.float32), device=device)
    on = torch.tensor([int(round(events[i][0] / HOP)) for i in short], device=device)
    off = torch.tensor([int(round(events[i][1] / HOP)) for i in short], device=device)
    out = list(events)
    new = {}
    for side, c in ((0, on), (1, off)):
        crops = normalise(crop_batch(mel, c))
        ex = crop_batch(torch.as_tensor(extra, device=device), c, 0.0) if extra is not None else None
        p = torch.softmax(model(crops, span_rows(c, on, off), ex)[:, side], -1)
        conf, arg = p.max(-1)
        moved = (c + arg - HALF).float() * HOP
        keep = conf < min_conf
        new[side] = torch.where(keep, c.float() * HOP, moved).cpu().numpy()
    for j, i in enumerate(short):
        a, b = float(new[0][j]), float(new[1][j])
        if b - a >= 0.02:
            out[i] = (round(max(a, 0.0), 3), round(b, 3))
    return out
