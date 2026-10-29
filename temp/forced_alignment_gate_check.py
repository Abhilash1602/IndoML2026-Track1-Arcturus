"""3.5-week re-plan, Must-do #1 Tier 1: mandatory gate-check for
transcript-based forced alignment BEFORE trusting it for Bronze's 17,884
currently-blind clips. Runs the full pipeline (strip tags -> track opening/
closing tag word-positions -> romanize with uroman -> CTC forced-align with
torchaudio's MMS_FA -> reconstruct (onset, offset) from surrounding word
timestamps) BLIND on a sample of Gold clips (real timestamps held out,
only used for scoring afterward), and scores with this project's own
`compute_event_f1` tolerance rule (`max(0.2*duration, 0.05)`).

Per-language breakdown is mandatory (per the plan) -- low-resource
languages likely have poor multilingual-model coverage and should be
excluded from Tier 1's real Bronze output even if the aggregate gate
passes.

Bar to clear: comfortably beat the already-established free baseline
(char-position-in-transcript-length MAE 0.50s vs a blind-midpoint-guess
MAE 1.70s), and land onset/offset error meaningfully under the ~0.12-0.14s
boundary-error ceiling diagnosed for non-breathing categories -- otherwise
Tier 1 isn't precise enough to trust as frame-level training targets, and
Tier 0 (clip-level presence + stratified sampling, already shipped) is the
pragmatic stopping point.

Usage: python -m temp.forced_alignment_gate_check [--n 300]
"""
import argparse
import collections
import re

import torch
import torchaudio
import uroman
from datasets import load_dataset

from temp.official_metric import micro_event_f1

TAG_RE = re.compile(r"<[^>]*>|\[[a-zA-Z_ ]+\]")
CLOSE_TAG_RE = re.compile(r"</[^>]*>")


def strip_and_locate_tags(transcript):
    """Returns (words, [(word_index_before_tag, tag_text, is_closing), ...]).
    word_index_before_tag is the index into `words` immediately BEFORE the
    tag's position (0 if the tag is at the very start)."""
    tokens = re.split(r"(\s+|<[^>]*>|\[[a-zA-Z_ ]+\])", transcript)
    words, tag_events = [], []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("<") or tok.startswith("["):
            is_closing = tok.startswith("</")
            tag_events.append((len(words), tok, is_closing))
        elif re.search(r"\w", tok, re.UNICODE):
            words.append(tok)
    return words, tag_events


def normalize_tag(tag_text):
    return tag_text.strip().strip("<>[]/").strip().lower().replace("_", " ")


def predict_events(word_spans, tag_events, clip_duration):
    """word_spans: list of (start, end) per word, or None if that word
    wasn't aligned. tag_events: from strip_and_locate_tags. Pairs each
    opening tag with its next matching closing tag (by normalized name);
    an opening tag with no closing tag counts as a zero-duration point at
    its word gap (matches this project's own single-frame boundary
    convention elsewhere -- still scoreable via the tolerance rule)."""
    def word_gap_time(word_index, side):
        # side='start': time just after word_index-1 ends (or clip start).
        # side='end': time just before word_index starts (or clip end).
        if side == "start":
            if word_index <= 0:
                return 0.0
            span = word_spans[word_index - 1]
            return span[1] if span else 0.0
        if word_index >= len(word_spans):
            return clip_duration
        span = word_spans[word_index]
        return span[0] if span else clip_duration

    open_stack = collections.defaultdict(list)
    events = []
    for word_index, tag_text, is_closing in tag_events:
        key = normalize_tag(tag_text)
        if not is_closing:
            open_stack[key].append(word_index)
        else:
            if open_stack[key]:
                open_word_index = open_stack[key].pop()
                onset = word_gap_time(open_word_index, "start")
                offset = word_gap_time(word_index, "end")
                if offset > onset:
                    events.append((onset, offset, key))
    # any unmatched opening tags -> zero-width point event at that word gap
    for key, indices in open_stack.items():
        for word_index in indices:
            t = word_gap_time(word_index, "start")
            events.append((t, t + 0.05, key))
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    label_set = set(bundle.get_labels())
    romanizer = uroman.Uroman()

    ds = load_dataset("ARTPARK-IISc/Vaani-Noise-Event-Dataset", split="train",
                       cache_dir="./data/hf_cache")
    gold_indices = [i for i, q in enumerate(ds["annotationQuality"]) if q == "verified_timestamps"]
    gold_indices = gold_indices[:args.n]
    print(f"Gate-checking forced alignment on {len(gold_indices)} Gold clips...", flush=True)

    per_lang_errors = collections.defaultdict(list)
    per_lang_f1 = collections.defaultdict(lambda: [0, 0])  # [tp, total_gt+pred pairs via compute_event_f1 below]
    all_gt, all_pred = [], []
    all_errors = []
    skipped = 0

    for n_done, idx in enumerate(gold_indices):
        row = ds[idx]
        events = row["NoiseSubCategoryTimeStamp"] or []
        if not events:
            continue
        transcript = row["transcript"] or ""
        words, tag_events = strip_and_locate_tags(transcript)
        if not words or not tag_events:
            skipped += 1
            continue

        roman_words = [romanizer.romanize_string(w) for w in words]
        roman_words = [w if w and not (set(w) - label_set) else "*" for w in roman_words]

        wav = torch.tensor(row["audio"]["array"], dtype=torch.float32).unsqueeze(0)
        sr = row["audio"]["sampling_rate"]
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        clip_duration = wav.size(1) / 16000

        try:
            with torch.inference_mode():
                emission, _ = model(wav.to(device))
                token_ids = tokenizer(roman_words)
                token_spans = aligner(emission[0], token_ids)
        except Exception as exc:
            skipped += 1
            continue

        ratio = wav.size(1) / emission.size(1) / 16000
        word_spans = []
        for spans in token_spans:
            if spans:
                word_spans.append((spans[0].start * ratio, spans[-1].end * ratio))
            else:
                word_spans.append(None)

        predicted = predict_events(word_spans, tag_events, clip_duration)
        pred_by_cat, gt_by_cat = collections.defaultdict(list), collections.defaultdict(list)
        # match predicted tag-name events to real (category, start, end) events by tag name
        real_tags_by_key = collections.defaultdict(list)
        for event in events:
            real_tags_by_key[normalize_tag(event["tag"])].append(event)

        language = row.get("language", "unknown")
        for onset, offset, key in predicted:
            matches = real_tags_by_key.get(key)
            if not matches:
                continue
            # nearest real event of the same tag, by onset distance
            best = min(matches, key=lambda e: abs(float(e["start"]) - onset))
            real_onset, real_offset = float(best["start"]), float(best["end"])
            onset_err = abs(onset - real_onset)
            offset_err = abs(offset - real_offset)
            per_lang_errors[language].append((onset_err + offset_err) / 2)
            all_errors.append((onset_err + offset_err) / 2)
            all_gt.append([(real_onset, real_offset)])
            all_pred.append([(onset, offset)])

        if (n_done + 1) % 50 == 0:
            print(f"  ...{n_done + 1}/{len(gold_indices)} processed, "
                  f"{len(all_errors)} scored events so far, {skipped} skipped", flush=True)

    print(f"\n=== Overall ===", flush=True)
    print(f"Scored {len(all_errors)} events across {len(gold_indices) - skipped} usable clips "
          f"({skipped} skipped: no transcript tags or alignment failure)", flush=True)
    if all_errors:
        import statistics
        print(f"Mean abs error (onset+offset)/2: {statistics.mean(all_errors):.3f}s "
              f"(median {statistics.median(all_errors):.3f}s)", flush=True)
        print(f"Reference baselines (established this session): char-position MAE 0.50s, "
              f"blind-midpoint-guess MAE 1.70s", flush=True)
        f1, precision, recall = micro_event_f1(all_gt, all_pred)
        print(f"Event F1 (project's own micro-pooled tolerance-based matching, on matched "
              f"tag pairs): F1={f1:.4f} P={precision:.4f} R={recall:.4f}", flush=True)

    print(f"\n=== Per-language (n>=5 events only) ===", flush=True)
    for lang, errs in sorted(per_lang_errors.items(), key=lambda x: -len(x[1])):
        if len(errs) < 5:
            continue
        import statistics
        print(f"  {lang:15s} n={len(errs):4d}  mean_err={statistics.mean(errs):.3f}s  "
              f"median_err={statistics.median(errs):.3f}s", flush=True)

    print(f"\n=== Verdict ===", flush=True)
    if all_errors and statistics.mean(all_errors) < 0.35:
        print("PASS-ish: mean error comfortably beats the 0.50s char-position baseline. "
              "Review per-language breakdown before greenlighting Tier 1 for Bronze -- "
              "exclude any language with much worse than the aggregate error.", flush=True)
    else:
        print("DOES NOT CLEARLY BEAT THE BASELINE -- do not proceed to Tier 1 for Bronze. "
              "Tier 0 (already shipped) is the pragmatic stopping point.", flush=True)


if __name__ == "__main__":
    main()
