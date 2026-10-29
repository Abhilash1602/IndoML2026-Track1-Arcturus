"""In-domain masked-reconstruction context network (MAT-SED style, arXiv:2408.08673), Oct 12.

All our heads (Conformer/GRU context networks over BEATs/ATST/Dasheng features) learned temporal
structure only from labelled edges. This transformer context network is first pre-trained without
labels on ~92k Vaani clips (Gold + Silver + Bronze + validation train70; hold30 and test excluded):
spans of the frozen backbone's 25 Hz frame features are replaced by a learned mask token and the
network reconstructs them (L2 on standardised features at masked positions). It is then fine-tuned
as the context network of a StandaloneDashengSED member (CONTEXT=mae).
"""
import math

import torch
import torch.nn as nn


def sinusoid_table(n, d):
    pos = torch.arange(n).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    t = torch.zeros(n, d)
    t[:, 0::2], t[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return t


class ContextNet(nn.Module):
    def __init__(self, d_in=768, d=256, layers=4, heads=4, max_len=1024):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.register_buffer("pos", sinusoid_table(max_len, d), persistent=False)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.norm = nn.LayerNorm(d)
        self.recon = nn.Linear(d, d_in)
        self.d = d

    def forward(self, feats, mask=None):
        """feats (B, T, d_in); mask (B, T) bool, True = masked. Returns (hidden (B, T, d), recon (B, T, d_in))."""
        x = self.inp(feats)
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype).expand_as(x), x)
        x = self.norm(self.enc(x + self.pos[: x.size(1)].to(x.dtype)))
        return x, self.recon(x)


def span_mask(B, T, ratio=0.3, min_span=4, max_span=10, device="cpu", generator=None):
    """Boolean (B, T) mask with random spans covering about `ratio` of each sequence."""
    m = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        while m[b].float().mean() < ratio:
            L = int(torch.randint(min_span, max_span + 1, (1,), generator=generator))
            s = int(torch.randint(0, T - L + 1, (1,), generator=generator))
            m[b, s:s + L] = True
    return m.to(device)
