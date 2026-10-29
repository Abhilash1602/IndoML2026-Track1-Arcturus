"""Gated edge correction (Sep 29 root-cause analysis, STRATEGY.md).

Root cause of the Natural gap: synthetic references are exact mix times (full acoustic
extent) while Natural references are human judgements, which are tighter, and training
mixes both. Natural predictions therefore come out too wide. Each predicted event
shows its domain in its edge sharpness (median edge contrast: Natural ~0.2, synthetic
~0.8; a classifier fit on our own data separates the official domains with AUC 0.98).

Correction: per predicted-duration bin, the median (true - predicted) onset/offset
offset, fitted on the Gold held-out split (never validation), applied only to
predicted events with edge contrast < GATE and duration < MAX_DUR.
"""
import json

import numpy as np

HOP = 0.01
BINS = [0, .2, .3, .4, .5, .7, 1.0, 1.5, 2.5, 5, 99]
GATE = 0.5      # a-priori midpoint of the Natural (~0.2) and synthetic (~0.8) edge-contrast medians
MAX_DUR = 1.0   # only short predictions (longer ones: correction cost synthetic Dice)
MIN_PAIRS = 20  # bins with fewer fitted pairs get no correction


def fit_table(refs, preds):
    """refs/preds: per-clip lists of (onset, offset). Uses 1:1 overlapping pairs only."""
    corr = {b: [] for b in range(len(BINS) - 1)}
    for g, p in zip(refs, preds):
        for po, pf in p:
            ov = [(o, f) for o, f in g if o < pf and f > po]
            if len(ov) != 1 or len([1 for a, b in p if a < ov[0][1] and b > ov[0][0]]) != 1:
                continue
            corr[int(np.searchsorted(BINS, pf - po, side="right") - 1)].append((ov[0][0] - po, ov[0][1] - pf))
    return {b: ([float(np.median(np.array(v)[:, 0])), float(np.median(np.array(v)[:, 1]))]
                if len(v) >= MIN_PAIRS else [0.0, 0.0]) for b, v in corr.items()}


def save_table(table, path):
    with open(path, "w") as fh:
        json.dump({"bins": BINS, "gate": GATE, "max_dur": MAX_DUR, "table": {str(k): v for k, v in table.items()}}, fh, indent=1)


def load_table(path):
    with open(path) as fh:
        d = json.load(fh)
    assert d["bins"] == BINS and d["gate"] == GATE and d["max_dur"] == MAX_DUR
    return {int(k): v for k, v in d["table"].items()}


def edge_contrast(prob, onset, offset):
    """Mean probability jump across the event's edges (3 frames in vs 3 frames out).
    Edges on the clip border are unmeasurable -> 1.0 (treated as sharp, left alone)."""
    T = len(prob)
    a, b = int(round(onset / HOP)), int(round(offset / HOP))
    c = []
    if a >= 3 and b > a:
        c.append(prob[a:min(a + 3, b)].mean() - prob[a - 3:a].mean())
    if b <= T - 3 and b > a:
        c.append(prob[max(b - 3, a):b].mean() - prob[b:b + 3].mean())
    return float(np.mean(c)) if c else 1.0


def correct_events(events, prob, table, clip_sec=None):
    """events: list of (onset, offset) seconds from the decode; prob: the (T,) ensemble
    probability they were decoded from. Returns corrected events (same order)."""
    out = []
    end = clip_sec if clip_sec is not None else len(prob) * HOP
    for on, off in events:
        if off - on >= MAX_DUR or edge_contrast(prob, on, off) >= GATE:
            out.append((on, off))
            continue
        a, c = table[int(np.searchsorted(BINS, off - on, side="right") - 1)]
        no, nf = max(0.0, on + a), min(end, off + c)
        out.append((no, nf) if nf - no >= 0.02 else (on, off))
    return out
