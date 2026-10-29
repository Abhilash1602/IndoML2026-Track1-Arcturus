import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class CategoryAuxLoss(nn.Module):
    """Session 9 plan, Part 1: auxiliary 7-category BCE loss, trained alongside
    (not instead of) the primary any-noise SEDLoss. Deliberately a standalone
    class rather than a change to SEDLoss.forward's signature, so every
    existing `criterion(logits, targets)` call site in the codebase is
    unaffected. `pos_weight`, if given, should be a (7,) tensor of per-category
    inverse-frequency weights computed from temp/inspect_category_variance.py's
    counts -- passed in by the caller rather than hardcoded here, since the
    right weighting differs between the Gold-only pool and a future Gold+Silver
    mix (Step 4 of the plan)."""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, cat_logits, cat_targets):
        return self.bce(cat_logits, cat_targets)


class OnsetOffsetLoss(nn.Module):
    """Session 9 plan, Part 2 Stage 2A: OOL from arXiv:2601.04178 ("Sound Event
    Detection with Boundary-Aware Optimization and Inference"). Confirmed via
    the paper's own stated equation (not assumed, per the plan's explicit
    flag): L = -(1/CT) * sum_{c,t} [ (1-p)^alpha * log(p) if y=1 else
    p^alpha * log(1-p) ], with the paper reporting alpha=2 as the **focusing
    exponent** applied symmetrically to both classes -- there is no separate
    class-balance weight, unlike this codebase's existing FocalLoss (whose
    alpha_t = alpha*y + (1-alpha)*(1-y) can only be held at a constant value
    of 0.5, not 1, for any single alpha choice, which would silently halve
    the loss magnitude relative to the paper's own formula). Kept as its own
    class rather than reusing FocalLoss for that reason -- same p_t
    convention and gamma-as-focusing-exponent shape, just without the
    per-class alpha_t multiplier.

    Targets are single-frame binary impulses at the exact onset/offset frame
    -- confirmed from the paper's text ("one at frames where an onset or
    offset occurs, and zero otherwise"), no dilation/smoothing -- superseding
    the plan's earlier tentative dilated-window idea now that the paper's
    literal convention is known."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        log_p_t = torch.log(p_t.clamp_min(1e-8))
        return -((1 - p_t) ** self.gamma * log_p_t).mean()


def boundary_weights(mask, height, half_width_frames):
    """OWBCE per-frame weights (arXiv:2403.13254): 1 + height * r, where r is the
    |first difference| of the target mask (an impulse at every onset frame and
    at the first frame after every offset) convolved with the positive lobe of
    a sine window of 2*half_width_frames+1 taps (peak 1 at the boundary), then
    clamped to 1 so two nearby boundaries don't double the weight. Clip edges
    are not boundaries (an event running into frame 0 or the last frame gets no
    extra weight there). mask: (B, T) float in [0, 1]; returns (B, T)."""
    edges = torch.zeros_like(mask)
    edges[:, 1:] = (mask[:, 1:] - mask[:, :-1]).abs()
    taps = torch.arange(2 * half_width_frames + 1, device=mask.device, dtype=mask.dtype)
    kernel = torch.sin(torch.pi * (taps + 1) / (2 * half_width_frames + 2)).view(1, 1, -1)
    response = F.conv1d(edges.unsqueeze(1), kernel, padding=half_width_frames).squeeze(1)
    return 1.0 + height * response.clamp(max=1.0)


def tolerance_boundary_weights(mask, height, scale=1.5, min_tol=5.0):
    """Tolerance-shaped OWBCE weights (Oct 11). The scorer matches an event when BOTH edges lie
    within tol = max(0.2 * duration, 0.05 s) (= max(0.2 * dur_frames, 5) frames), and every event
    counts once in Event F1. So each edge of each target event gets a sine lobe of half-width
    scale * tol (the band where the 0.5-crossing must land) and height height * min_tol / tol:
    short events (tight tolerance) get tall narrow lobes, long events low wide ones, i.e. about
    the same total weight per event. Clip edges are not boundaries. mask: (B, T) in [0, 1]."""
    hard = (mask > 0.5).float()
    B, T = hard.shape
    d = torch.diff(F.pad(hard, (1, 1)), dim=1)  # (B, T+1): +1 at onset index, -1 at offset index
    on_b, on_t = torch.nonzero(d > 0, as_tuple=True)
    off_b, off_t = torch.nonzero(d < 0, as_tuple=True)  # onsets/offsets pair up in order per clip
    w = torch.ones_like(hard)
    if on_t.numel() == 0:
        return w
    dur = (off_t - on_t).float()
    tol = torch.clamp(0.2 * dur, min=min_tol)
    amp = height * min_tol / tol
    half = scale * tol
    t = torch.arange(T, device=mask.device, dtype=hard.dtype)
    resp = torch.zeros_like(hard)
    for edge, keep in ((on_t, on_t > 0), (off_t, off_t < T)):  # skip clip-edge "boundaries"
        x = (t[None, :] - edge[:, None].float()).abs() / half[:, None]  # (E, T)
        lobe = torch.cos(0.5 * torch.pi * x.clamp(max=1.0)) * (x < 1)
        resp.index_put_((on_b.unsqueeze(1).expand_as(lobe)[keep], t.long().expand_as(lobe)[keep]),
                        (amp[:, None] * lobe)[keep], accumulate=True)
    return w + resp.clamp(max=height)


class SEDLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bce_weight, self.dice_weight, self.bnd_weight = config.LOSS_BCE_W, config.LOSS_DICE_W, config.LOSS_BND_W
        # Positive-class up-weighting for the BCE term -- noise-event frames are a
        # minority (Gold is ~45% breathing but still mostly-silence per-frame), and
        # the org's own baseline uses pos_weight=4.0 for exactly this (bce_pos in
        # their losses.py). We had no equivalent; default 1.0 keeps old behavior.
        pos_weight = getattr(config, "POS_WEIGHT", 1.0)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.owbce_height = getattr(config, "OWBCE_HEIGHT", 0.0)
        self.owbce_half_width = getattr(config, "OWBCE_HALF_WIDTH", 10)
        # OWBCE_MODE=tol: tolerance-shaped weights (tolerance_boundary_weights); default = fixed window
        import os
        self.owbce_mode = os.environ.get("OWBCE_MODE", "fixed")
        self.owbce_tol_scale = float(os.environ.get("OWBCE_TOL_SCALE", "1.5"))
        self.chunk_size = 50

    def _bce(self, logits, targets):
        if self.owbce_height <= 0:
            return self.bce(logits, targets)
        per_frame = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.bce.pos_weight, reduction="none")
        # Normalised by the mean weight so the BCE term keeps its scale and the
        # existing LOSS_*_W balance and learning rates stay valid.
        if self.owbce_mode == "tol":
            weights = tolerance_boundary_weights(targets.detach(), self.owbce_height, self.owbce_tol_scale)
        else:
            weights = boundary_weights(targets.detach(), self.owbce_height, self.owbce_half_width)
        return (weights * per_frame).sum() / weights.sum()

    def _windowed_dice(self, probs, targets):
        frames = probs.shape[1]
        pad = (-frames) % self.chunk_size
        if pad:
            probs = F.pad(probs, (0, pad))
            targets = F.pad(targets, (0, pad))
        chunks = probs.shape[1] // self.chunk_size
        probs = probs.view(probs.shape[0], chunks, self.chunk_size)
        targets = targets.view(targets.shape[0], chunks, self.chunk_size)
        intersection = torch.sum(probs * targets, dim=2)
        union = torch.sum(probs, dim=2) + torch.sum(targets, dim=2)
        return (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()

    def forward(self, logits, targets):
        bce = self._bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets, dim=-1)
        union = torch.sum(probs, dim=-1) + torch.sum(targets, dim=-1)
        dice = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        boundary = F.mse_loss(torch.abs(probs[:, 1:] - probs[:, :-1]), torch.abs(targets[:, 1:] - targets[:, :-1]))
        return self.bce_weight * bce + self.dice_weight * dice + self.bnd_weight * boundary