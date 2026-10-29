"""Corrected event-F1/Dice matching official metric semantics, verbatim-copied
scorer logic from nagshubhadip/noise-event-detection-and-removal's evaluate.py
(labeled "copied from the Track 1 Evaluation page" -- the real Codabench scorer).

Two fixes vs. src/utils.py's compute_event_f1 (used everywhere in this project's
local validation so far):
  1. Micro-averaged F1: pool TP/FP/FN across ALL clips into one global F1, instead
     of averaging a separate per-clip F1 (which scores 0.0 for a correct true-negative
     clip and drags down the macro average -- Natural likely has more of these than
     our synthetic set, so this plausibly explains part of the local/real Natural-F1
     gap seen all session).
  2. Global closest-first matching: collect every valid (ref, pred) pair across a
     clip, sort ALL of them by distance, and assign matches greedily in that global
     order -- not per-reference in list order (which is what compute_event_f1 does).
"""
import numpy as np


def match_events(ref_events, pred_events, tolerance_frac=0.20):
    """ref_events/pred_events: list of (onset, offset) tuples for ONE clip."""
    matched_ref, matched_pred, candidates = set(), set(), []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)
    tp = len(matched_ref)
    return tp, len(pred_events) - tp, len(ref_events) - tp


def micro_event_f1(ref_list, pred_list):
    """ref_list/pred_list: parallel lists, one list-of-(onset,offset) per clip.
    Returns (f1, precision, recall) pooled across the whole set."""
    TP = FP = FN = 0
    for ref_events, pred_events in zip(ref_list, pred_list):
        tp, fp, fn = match_events(ref_events, pred_events)
        TP += tp; FP += fp; FN += fn
    p = TP / (TP + FP) if (TP + FP) else 0.0
    r = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return f1, p, r


def events_to_frames(events, n_frames, hop_sec):
    mask = np.zeros(n_frames, dtype=np.int32)
    for on, off in events:
        s, e = int(round(on / hop_sec)), int(round(off / hop_sec))
        mask[max(0, s):min(n_frames, e)] = 1
    return mask


def macro_segment_dice(ref_list, pred_list, hop_sec):
    """Per-clip Dice on the frame grid, macro-averaged -- matches
    src/utils.py's compute_segment_dice already; kept here for a single
    self-contained scoring entry point."""
    scores = []
    for ref_events, pred_events in zip(ref_list, pred_list):
        all_ev = ref_events + pred_events
        if not all_ev:
            scores.append(1.0)
            continue
        n_frames = int(max(off for _, off in all_ev) / hop_sec) + 2
        rm = events_to_frames(ref_events, n_frames, hop_sec)
        pm = events_to_frames(pred_events, n_frames, hop_sec)
        inter, total = int((rm & pm).sum()), int(rm.sum() + pm.sum())
        scores.append(1.0 if total == 0 else 2.0 * inter / total)
    return float(np.mean(scores)) if scores else 0.0


def events_from_mask(mask, hop_sec, min_dur=0.0):
    events, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            events.append((start * hop_sec, i * hop_sec))
            start = None
    if start is not None:
        events.append((start * hop_sec, len(mask) * hop_sec))
    return [(a, b) for a, b in events if b - a >= min_dur]
