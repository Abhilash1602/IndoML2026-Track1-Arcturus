"""Dice lock: re-partition the frame regions of a base submission (v6) into touching events.

Segment Dice is computed on the union of predicted frames, so cutting a region [a, b] into
[a, m] + [m, b] leaves it exactly unchanged; only Event F1 (one-to-one onset/offset matching)
sees the cut. Here an event-set model (SEDT) is used only as a source of cut positions inside
the base regions -- it can never add or remove frames.
"""
import numpy as np

HOP = 0.01


def lock_partition(regions, events, conf=None, c_min=0.5, min_piece=0.03, merge_tol=0.03):
    """regions: base events [(a, b)]; events: event-model events [(on, off)] with optional
    confidences. Each region is cut at the onsets/offsets of confident events that fall inside
    it at least min_piece from both ends (cuts closer than merge_tol are merged). Returns
    touching pieces covering exactly the base regions."""
    conf = [1.0] * len(events) if conf is None else conf
    edges = sorted(x for (on, off), c in zip(events, conf) if c >= c_min for x in (on, off))
    out = []
    for a, b in regions:
        cuts = []
        for x in edges:
            if a + min_piece <= x <= b - min_piece and (not cuts or x - cuts[-1] >= merge_tol):
                if not cuts or x - cuts[-1] >= min_piece:
                    cuts.append(round(float(x), 2))  # on the 10 ms grid: exact under any rasterisation
        pts = [a] + cuts + [b]
        out += [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    return out


def frame_mask(events, n_frames=None, hop=HOP):
    """Same rasterisation as the scorer (temp/official_metric.events_to_frames)."""
    if n_frames is None:
        n_frames = int(max((off for _, off in events), default=0) / hop) + 2
    m = np.zeros(n_frames, bool)
    for on, off in events:
        m[max(0, int(round(on / hop))):min(n_frames, int(round(off / hop)))] = True
    return m


def frames_equal(a, b, hop=HOP):
    n = int(max([off for _, off in list(a) + list(b)], default=0) / hop) + 2
    return bool(np.array_equal(frame_mask(a, n, hop), frame_mask(b, n, hop)))
