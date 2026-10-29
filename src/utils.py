import numpy as np


def compute_segment_dice(pred_mask, gt_mask):
    intersection, total = np.sum(pred_mask * gt_mask), np.sum(pred_mask) + np.sum(gt_mask)
    return 1.0 if total == 0 else 2.0 * intersection / total


def extract_events_from_mask(mask, hop_sec=0.01, min_duration=0.05):
    events, start = [], None
    for index, value in enumerate(mask):
        if value == 1 and start is None:
            start = index * hop_sec
        elif value == 0 and start is not None:
            end = index * hop_sec
            if end - start >= min_duration:
                events.append({"onset": start, "offset": end})
            start = None
    if start is not None and len(mask) * hop_sec - start >= min_duration:
        events.append({"onset": start, "offset": len(mask) * hop_sec})
    return events


def compute_event_f1(pred_events, gt_events):
    matched, true_positive = set(), 0
    for reference in gt_events:
        tolerance = max(0.2 * (reference["offset"] - reference["onset"]), 0.05)
        candidates = [(abs(pred["onset"] - reference["onset"]) + abs(pred["offset"] - reference["offset"]), i)
                      for i, pred in enumerate(pred_events) if i not in matched
                      and abs(pred["onset"] - reference["onset"]) <= tolerance
                      and abs(pred["offset"] - reference["offset"]) <= tolerance]
        if candidates:
            _, index = min(candidates); matched.add(index); true_positive += 1
    precision = true_positive / len(pred_events) if pred_events else 0.0
    recall = true_positive / len(gt_events) if gt_events else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extract_events_hysteresis(
    probs,
    hop_sec,
    high_thresh=0.55,
    low_thresh=0.30,
    min_duration_sec=0.1,
    max_gap_sec=0.2,
):
    events, start = [], None
    for index, probability in enumerate(probs):
        if start is None and probability >= high_thresh:
            start = index * hop_sec
        elif start is not None and probability < low_thresh:
            end = index * hop_sec
            events.append({"onset": start, "offset": end})
            start = None
    if start is not None:
        events.append({"onset": start, "offset": len(probs) * hop_sec})

    merged = []
    for event in events:
        if merged and event["onset"] - merged[-1]["offset"] <= max_gap_sec:
            merged[-1]["offset"] = event["offset"]
        else:
            merged.append(event)

    return [
        event for event in merged
        if event["offset"] - event["onset"] >= min_duration_sec
    ]


def viterbi_decode(probabilities, p_stay_silence=0.90, p_stay_event=0.95):
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    num_frames = len(probs)
    log_emission = np.stack([np.log(1 - probs), np.log(probs)], axis=1)
    log_trans = np.array([
        [np.log(p_stay_silence), np.log(1 - p_stay_silence)],
        [np.log(1 - p_stay_event), np.log(p_stay_event)],
    ])
    log_delta = np.empty((num_frames, 2))
    backpointer = np.zeros((num_frames, 2), dtype=np.int64)
    log_delta[0] = np.log(0.5) + log_emission[0]
    for t in range(1, num_frames):
        scores = log_delta[t - 1][:, None] + log_trans
        backpointer[t] = np.argmax(scores, axis=0)
        log_delta[t] = scores[backpointer[t], np.arange(2)] + log_emission[t]
    states = np.zeros(num_frames, dtype=np.int64)
    states[-1] = np.argmax(log_delta[-1])
    for t in range(num_frames - 2, -1, -1):
        states[t] = backpointer[t + 1, states[t + 1]]
    return states


def extract_events_viterbi(states, hop_sec):
    events, start = [], None
    for index, state in enumerate(states):
        if state == 1 and start is None:
            start = index
        elif state == 0 and start is not None:
            events.append({"onset": round(start * hop_sec, 3), "offset": round(index * hop_sec, 3)})
            start = None
    if start is not None:
        events.append({"onset": round(start * hop_sec, 3), "offset": round(len(states) * hop_sec, 3)})
    return events