"""SEBBs (Sound Event Bounding Boxes) as an alternative to Viterbi decoding.

Kept isolated from src/utils.py so that files which don't need SEBBs (train.py,
generate_submission.py's default path, etc.) never require the `sebbs`/
`sed_scores_eval` packages to be installed just to import this project's core
utilities.

Requires (on the machine actually running this, not this sandbox):
    pip install sed_scores_eval
    pip install git+https://github.com/merlresearch/sebbs.git

API verified directly against the installed packages (not just the README):
- sed_scores_eval.io.create_score_dataframe(scores_2d, timestamps, event_classes)
- sebbs.csebbs.tune(scores, ground_truth, audio_durations, *, step_filter_lengths,
  merge_thresholds_abs, merge_thresholds_rel, selection_fn, **selection_kwargs)
  -> (CSEBBsPredictor, best_values) when folds=None (the default).
- sebbs.csebbs.select_best_cbf(..., onset_collar=0.2, offset_collar=0.2,
  offset_collar_rate=0.2, classwise=True) -> passed as selection_fn to tune().
- CSEBBsPredictor.predict(scores, audio_ids=None, return_sed_scores=False) -> dict
  of audio_id -> list of (onset, offset, class, confidence) tuples.

Note on collar semantics vs. this project's own Event F1 metric
(tolerance = max(0.2 * duration, 0.05), applied symmetrically to onset AND
offset — see src/utils.py's compute_event_f1): sed_scores_eval's collar-based
F-score applies that duration-scaled-or-floor rule to OFFSET only
(offset_collar_for_gt_event = max(offset_collar, offset_collar_rate * duration));
onset_collar is a plain fixed value with no duration scaling. There is no
"onset_collar_rate". So onset_collar=0.05 is the closest available
approximation to our own onset tolerance, not an exact match — it will be
stricter than our real metric for events longer than 0.25s. This is why
sebbs_predicted_events() below is deliberately re-scored with THIS project's
own compute_segment_dice/compute_event_f1, rather than trusting select_best_cbf's
internal score as equivalent to our real metric.
"""
import numpy as np
import sed_scores_eval
from sebbs import csebbs

from src.utils import extract_events_from_mask


def build_sed_scores_eval_inputs(probabilities, ground_truth_masks, hop_sec, clip_duration_sec):
    """Bridge this project's cached (probabilities, ground_truth) numpy arrays
    (as produced by optimize_postprocessing.py's cache_validation_predictions)
    into the dict formats sed_scores_eval / sebbs expect. Returns
    (scores, ground_truth, audio_durations, clip_ids)."""
    clip_ids = [f"val_{i}" for i in range(len(probabilities))]
    scores, ground_truth, audio_durations = {}, {}, {}
    for probability, mask, clip_id in zip(probabilities, ground_truth_masks, clip_ids):
        timestamps = np.arange(len(probability) + 1, dtype=np.float64) * hop_sec
        scores[clip_id] = sed_scores_eval.io.create_score_dataframe(
            probability.reshape(-1, 1).astype(np.float64),
            timestamps=timestamps,
            event_classes=["noise"],
        )
        events = extract_events_from_mask(mask.astype(np.int32), hop_sec)
        ground_truth[clip_id] = [(event["onset"], event["offset"], "noise") for event in events]
        audio_durations[clip_id] = clip_duration_sec
    return scores, ground_truth, audio_durations, clip_ids


def tune_sebbs(
    probabilities,
    ground_truth_masks,
    hop_sec,
    clip_duration_sec,
    step_filter_lengths,
    merge_thresholds_abs,
    merge_thresholds_rel,
    onset_collar=0.05,
    offset_collar=0.05,
    offset_collar_rate=0.2,
    either_abs_or_rel_threshold=False,
):
    """Grid-search SEBBs' hyper-parameters against collar-based F1 on cached
    validation data. Returns (predictor, best_values, scores, ground_truth,
    audio_durations, clip_ids) — the latter four are needed again to re-score
    the predictor's own output with this project's Dice/F1 functions.

    Note: merge_thresholds_abs/rel are NOT time/duration quantities — per
    sebbs.csebbs._merge_segments' actual source, they threshold the *score*
    dip between two candidate segments (event_score/gap_score < rel,
    event_score - gap_score < abs), governing whether a probability dip
    looks like a real gap vs. noise. Only step_filter_lengths is a duration
    in seconds. either_abs_or_rel_threshold=False lets abs and rel candidates
    combine jointly instead of being evaluated as two separate families
    (tune()'s own default is True, i.e. one family or the other, not both)."""
    scores, ground_truth, audio_durations, clip_ids = build_sed_scores_eval_inputs(
        probabilities, ground_truth_masks, hop_sec, clip_duration_sec
    )
    predictor, best_values = csebbs.tune(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        step_filter_lengths=step_filter_lengths,
        merge_thresholds_abs=merge_thresholds_abs,
        merge_thresholds_rel=merge_thresholds_rel,
        selection_fn=csebbs.select_best_cbf,
        either_abs_or_rel_threshold=either_abs_or_rel_threshold,
        onset_collar=onset_collar,
        offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
    )
    return predictor, best_values, scores, ground_truth, audio_durations, clip_ids


def sebbs_predicted_events(predictor, scores, clip_id, detection_threshold):
    """Run the tuned predictor and return this clip's DETECTED events (SEBB confidence >
    detection_threshold) in this project's {"onset": ..., "offset": ...} format.

    Bug fixed Oct 8: the earlier version returned every SEBB candidate and discarded the
    confidence, i.e. no detection threshold at all -- every background segment became an
    event, which is why all three earlier SEBB attempts scored the same ~0.38 Natural (and
    synthetic collapsed). SEBBs separate extent (change points) from confidence; the
    detection threshold is what turns candidates into detections."""
    predictions = predictor.predict(scores, audio_ids=[clip_id])
    clip_predictions = predictions.get(clip_id, [])
    return [{"onset": float(onset), "offset": float(offset)}
            for onset, offset, _, confidence in clip_predictions if confidence > detection_threshold]
