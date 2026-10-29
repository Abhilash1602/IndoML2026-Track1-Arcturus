"""Event-query cross-attention boundary head (Lane A, Oct 11).

Each candidate event from our pipeline becomes a query that (i) attention-pools the frames
inside it (what the sound is) and (ii) cross-attends to 10 ms frame features around its edges,
then predicts its onset and offset as distributions over +/-0.3 s (61 bins of 10 ms), plus
P(match within the official tolerance) and P(pure false positive).

Frame memory (switches, ablated in temp/train_boundary_xattn.py):
  spectral stream  : 80-band 10 ms log-mel -> CNN with frequency-only pooling (real 10 ms evidence)
  member stream    : the 5 v4 members' conformer embeddings (40 ms, upsampled) + their 10 ms logits
  member="xattn"   : per frame, the spectral stream attends over the 5 members ("cross-variable"
                     attention: which member to trust where); "mean" averages them
  fusion="xattn"   : spectral queries cross-attend to the member stream over time (time-frequency
                     cross fusion); "concat" = linear on the concatenation
  use_global       : a 40 ms whole-clip context stream that queries also attend to (long-term
                     dynamics), next to the masked local memory (short-term)
Inputs are batched clips of equal length T (frames); proposals are (P, 3): clip index, onset and
offset in frames (float).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HALF = 30  # edge window +/- 30 frames = +/- 0.3 s -> 61 bins
CTX = 100  # cross-attention window +/- 1 s around the proposal


def sinusoid_feats(x, d):
    """x (..., k) -> (..., k * d) sinusoidal features of each scalar (frames scale)."""
    freqs = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / d))
    a = x.unsqueeze(-1).float() / 4.0 * freqs
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class SpecCNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d((4, 1)))
        self.proj = nn.Linear(64 * 5, d)

    def forward(self, mel):  # (B, 80, T)
        x = self.net(mel.unsqueeze(1))  # (B, 64, 5, T)
        return self.proj(x.flatten(1, 2).transpose(1, 2))  # (B, T, d)


class BoundaryXAttn(nn.Module):
    def __init__(self, d=192, heads=4, n_members=5, emb_dim=144, dec_layers=3,
                 member="xattn", fusion="xattn", use_global=True):
        super().__init__()
        self.member, self.fusion, self.use_global = member, fusion, use_global
        self.spec = SpecCNN(d)
        self.mem_in = nn.Linear(emb_dim + 1, d)
        self.mem_id = nn.Parameter(torch.randn(n_members, d) * 0.02)
        if member == "xattn":
            self.member_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        if fusion == "xattn":
            self.fuse_attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.fuse_norm = nn.LayerNorm(d)
        else:
            self.fuse_lin = nn.Linear(2 * d, d)
        self.local_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        if use_global:
            self.global_enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        self.pool_score = nn.Linear(d, 1)
        self.geo = nn.Linear(3 * 32, d)
        self.dec = nn.ModuleList()
        for _ in range(dec_layers):
            self.dec.append(nn.ModuleDict(dict(
                self_attn=nn.MultiheadAttention(d, heads, batch_first=True), n1=nn.LayerNorm(d),
                cross=nn.MultiheadAttention(d, heads, batch_first=True), n2=nn.LayerNorm(d),
                **({"gcross": nn.MultiheadAttention(d, heads, batch_first=True), "ng": nn.LayerNorm(d)} if use_global else {}),
                ff=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)), n3=nn.LayerNorm(d))))
        self.edge_mlp = nn.ModuleDict({k: nn.Sequential(nn.Linear(2 * d + 32, d), nn.GELU(), nn.Linear(d, 1))
                                       for k in ("on", "off")})
        self.bin_pos = nn.Parameter(torch.randn(2 * HALF + 1, 32) * 0.02)
        self.match_head = nn.Linear(d, 1)
        self.fp_head = nn.Linear(d, 1)

    def memory(self, mel, logits, emb):
        """mel (B, 80, T), logits (B, M, T), emb (B, M, T25, E) -> (B, T, d), global (B, T/4, d) or None."""
        B, M, T = logits.shape
        spec = self.spec(mel)  # (B, T, d)
        e = emb.float().repeat_interleave(4, dim=2)[:, :, :T]  # (B, M, T, E)
        if e.size(2) < T:
            e = F.pad(e, (0, 0, 0, T - e.size(2)))
        tok = self.mem_in(torch.cat([e, logits.float().unsqueeze(-1) / 5.0], -1)) + self.mem_id[None, :, None]
        tok = tok.permute(0, 2, 1, 3)  # (B, T, M, d)
        if self.member == "xattn":
            q = spec.reshape(B * T, 1, -1)
            mem, _ = self.member_attn(q, tok.reshape(B * T, M, -1), tok.reshape(B * T, M, -1))
            mem = mem.reshape(B, T, -1)
        else:
            mem = tok.mean(2)
        pos = sinusoid_feats(torch.arange(T, device=mel.device)[:, None].float(), spec.size(-1)).unsqueeze(0)
        if self.fusion == "xattn":
            # windowed (+/- 0.5 s) time-frequency cross fusion: spectral queries attend to the member stream
            idx = torch.arange(T, device=mel.device)
            far = (idx[:, None] - idx[None, :]).abs() > 50
            f, _ = self.fuse_attn(spec + pos, mem + pos, mem, attn_mask=far)
            x = self.fuse_norm(spec + f)
        else:
            x = self.fuse_lin(torch.cat([spec, mem], -1))
        x = self.local_enc(x + pos)
        g = self.global_enc(F.avg_pool1d(x.transpose(1, 2), 4).transpose(1, 2)) if self.use_global else None
        return x, g

    def forward(self, mel, logits, emb, props):
        """props (P, 3) float: [clip index, onset frame, offset frame]. Returns dict of
        on/off logits (P, 61), match (P,), fp (P,)."""
        x, g = self.memory(mel, logits, emb)
        B, T, d = x.shape
        ci = props[:, 0].long()
        on, off = props[:, 1], props[:, 2]
        t = torch.arange(T, device=x.device).float()
        inside = (t[None] >= on[:, None].floor()) & (t[None] <= off[:, None].ceil())  # (P, T)
        mem = x[ci]  # (P, T, d)
        s = self.pool_score(mem).squeeze(-1).masked_fill(~inside, -1e4)
        q = (s.softmax(-1).unsqueeze(-1) * mem).sum(1)  # attention pooling (selective weights)
        q = q + self.geo(sinusoid_feats(torch.stack([on, off, off - on], -1), 32))
        window = (t[None] < on[:, None] - CTX) | (t[None] > off[:, None] + CTX)  # cross-attn mask (P, T)
        same = ci[:, None] != ci[None, :]  # proposals only see proposals of their own clip
        q = q.unsqueeze(0)  # (1, P, d) -> self-attention over all proposals with a block mask
        for layer in self.dec:
            h, _ = layer["self_attn"](q, q, q, attn_mask=same)
            q = layer["n1"](q + h)
            qq = q[0].unsqueeze(1)  # (P, 1, d)
            h, _ = layer["cross"](qq, mem, mem, key_padding_mask=window)
            qq = layer["n2"](qq + h)
            if self.use_global:
                gg = g[ci]
                h, _ = layer["gcross"](qq, gg, gg)
                qq = layer["ng"](qq + h)
            qq = layer["n3"](qq + layer["ff"](qq))
            q = qq[:, 0].unsqueeze(0)
        q = q[0]  # (P, d)
        out = {"match": self.match_head(q).squeeze(-1), "fp": self.fp_head(q).squeeze(-1)}
        offs = torch.arange(-HALF, HALF + 1, device=x.device).float()
        for k, c in (("on", on), ("off", off)):
            pos = (c.round()[:, None] + offs[None]).long()  # (P, 61)
            valid = (pos >= 0) & (pos < T)
            w = mem.gather(1, pos.clamp(0, T - 1).unsqueeze(-1).expand(-1, -1, d)) * valid.unsqueeze(-1)
            h = torch.cat([w, q[:, None].expand(-1, 2 * HALF + 1, -1), self.bin_pos[None].expand(len(q), -1, -1)], -1)
            out[k] = self.edge_mlp[k](h).squeeze(-1).masked_fill(~valid, -1e4)
        return out
