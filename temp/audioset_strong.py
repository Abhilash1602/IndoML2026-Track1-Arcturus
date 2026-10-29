"""AudioSet-Strong as labelled natural noise events, harmonised to Vaani's annotation style
(plan Phase N1a).

Source: Google's temporally-strong AudioSet labels (CC-BY-4.0; annotators drew onset/offset on a
spectrogram) for the evaluation set, audio from the `agkphysics/AudioSet` HF mirror (eval
parquet). Language is irrelevant: these are the noise sounds Vaani annotators mark.

Vaani "noise events" are short discrete sounds (validation Natural: breathing/inhaling 0.32 s
median, horns 599/655 vehicle events, bird/insect/barking, child talking/yelling, beeps/rings),
not continuous background. So:
  * MAP only the matching short sound types to our 7 categories (table below); Speech is never
    a target;
  * KEEP a clip only if it contains Speech (Vaani is speech-dominant) and >= 1 mapped event;
  * DROP clips containing long background classes Vaani does not label (wind, water, roadway
    noise, crowds, ...), so no unlabelled noise is taught as negative;
  * KEEP mapped events within the category's Vaani duration range (<= p95 on validation
    Natural), dropping clips with longer mapped spans;
  * targets = merged any-noise mask (the scoring convention) on the 10 ms grid, 10 s.
Teacher harmonisation of edges is applied afterwards (temp/gen_pseudo_v11-style relabel,
`harmonise()` below): keep teacher events overlapping AudioSet events (Vaani-style edges), drop
teacher events without AudioSet support, add AudioSet short events the teacher missed.

Usage: python -m temp.audioset_strong survey | build
"""
import csv
import glob
import io
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

D = "./work/audioset_strong"

MAP = {  # AudioSet display name -> Vaani category
    **{k: "human_non_speech" for k in ("Breathing", "Gasp", "Sniff", "Pant", "Sigh", "Cough", "Throat clearing",
                                        "Snoring", "Hiccup", "Sneeze", "Snort", "Chewing, mastication")},
    **{k: "vehicle_traffic" for k in ("Vehicle horn, car horn, honking, toot", "Air horn, truck horn", "Train horn",
                                       "Car passing by", "Motorcycle", "Reversing beeps", "Bicycle bell")},
    **{k: "animal" for k in ("Bird vocalization, bird call, bird song", "Chirp, tweet", "Crow", "Bark", "Dog",
                              "Insect", "Moo", "Goat", "Crowing, cock-a-doodle-doo", "Chicken, rooster", "Cat",
                              "Bird", "Caterwaul", "Howl", "Whimper (dog)", "Livestock, farm animals, working animals")},
    **{k: "baby_child" for k in ("Child speech, kid speaking", "Children shouting", "Baby cry, infant cry",
                                  "Baby laughter", "Children playing")},
    **{k: "phone_signal_alarm" for k in ("Ringtone", "Telephone bell ringing", "Beep, bleep", "Alarm", "Doorbell",
                                          "Telephone", "Busy signal", "Chirp tone", "Alarm clock", "Ringing tone, ringback tone",
                                          "Telephone dialing, DTMF")},
    **{k: "singing_music" for k in ("Music", "Singing", "Male singing", "Female singing", "Child singing", "Whistling")},
    **{k: "appliance_machine" for k in ("Mechanical fan", "Blender, food processor", "Vacuum cleaner",
                                         "Sewing machine", "Microwave oven")},
}
BACKGROUND = ("Wind", "Wind noise (microphone)", "Water", "Rain", "Traffic noise, roadway noise", "Crowd",
              "Applause", "Cheering", "Waterfall", "Stream", "Ocean", "Engine", "Heavy engine (low frequency)",
              "Medium engine (mid frequency)", "Light engine (high frequency)", "Aircraft", "Jet engine",
              "Hubbub, speech noise, speech babble", "Inside, small room", "Inside, large room or hall",
              "Outside, urban or manmade", "Outside, rural or natural",
              "Mechanisms")  # generic machinery spans whole clips in AudioSet (median 9.98 s); Vaani rarely labels it
SPEECH = ("Speech", "Male speech, man speaking", "Female speech, woman speaking", "Conversation", "Narration, monologue")


def vaani_duration_caps():
    """Per-category p95 event duration on validation Natural (Vaani annotators)."""
    import ast
    rows = json.load(open("./data/official_validation_raw/validation/validationMetadata.json", encoding="utf-8"))
    d = defaultdict(list)
    for r in rows:
        if r.get("syntheticData"):
            continue
        v = r.get("NoiseSubCategoryTimeStamp") or "[]"
        v = ast.literal_eval(v) if isinstance(v, str) else v
        for e in v:
            d[e["category"]].append(float(e["end"]) - float(e["start"]))
    return {c: float(np.percentile(x, 95)) for c, x in d.items()}


def load_strong():
    names = dict(csv.reader(open(f"{D}/mid_to_display_name.tsv"), delimiter="\t"))
    seg = defaultdict(list)
    with open(f"{D}/audioset_eval_strong.tsv") as fh:
        next(fh)
        for line in fh:
            s, a, b, mid = line.rstrip("\n").split("\t")
            seg[s].append((float(a), float(b), names.get(mid, mid)))
    return seg


def select(seg, caps):
    keep, why = {}, Counter()
    for s, evs in seg.items():
        labels = {n for _, _, n in evs}
        if not labels & set(SPEECH):
            why["no speech"] += 1
            continue
        if labels & set(BACKGROUND):
            why["background class"] += 1
            continue
        mapped = [(a, b, MAP[n]) for a, b, n in evs if n in MAP]
        if not mapped:
            why["no mapped event"] += 1
            continue
        if any(b - a > caps.get(c, 10.0) for a, b, c in mapped):
            why["mapped span longer than Vaani p95"] += 1
            continue
        keep[s] = mapped
        why["kept"] += 1
    return keep, why


def survey():
    caps = vaani_duration_caps()
    seg = load_strong()
    keep, why = select(seg, caps)
    cats = Counter(c for v in keep.values() for _, _, c in v)
    durs = defaultdict(list)
    for v in keep.values():
        for a, b, c in v:
            durs[c].append(b - a)
    print(f"strong eval segments {len(seg)}; selection: {dict(why)}")
    print("Vaani p95 caps:", {k: round(v, 2) for k, v in caps.items()})
    for c, n in cats.most_common():
        print(f"  {c:20s} {n:5d} events, AudioSet median {np.median(durs[c]):.2f}s")


def build():
    """Decode the kept clips from the HF eval parquet -> 16 kHz mono 10 s int16 audio + 10 ms masks."""
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    import torchaudio
    caps = vaani_duration_caps()
    keep, _ = select(load_strong(), caps)
    by_ytid = {s.rsplit("_", 1)[0]: s for s in keep}
    audio, masks, ids, cats = [], [], [], []
    for f in sorted(glob.glob(f"{D}/hf/data/eval/*.parquet")):
        t = pq.read_table(f, columns=["video_id", "audio"]).to_pylist()
        for r in t:
            s = by_ytid.get(r["video_id"])
            if s is None:
                continue
            try:
                w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            except Exception:
                continue
            w = torch.as_tensor(w if w.ndim == 1 else w.mean(1))
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            w = torch.nn.functional.pad(w, (0, max(0, 160000 - w.numel())))[:160000]
            m = np.zeros(1000, np.uint8)
            for a, b, c in keep[s]:
                m[int(round(a * 100)):int(round(min(b, 10.0) * 100))] = 1
            audio.append((w.clamp(-1, 1).numpy() * 32767).astype(np.int16))
            masks.append(m)
            ids.append(s)
            cats.append(sorted({c for _, _, c in keep[s]}))
        print(f"{os.path.basename(f)}: {len(ids)} clips so far", flush=True)
    np.savez(f"{D}/audioset_strong_selected.npz", audio=np.stack(audio), mask=np.stack(masks), ids=np.array(ids))
    json.dump(dict(zip(ids, cats)), open(f"{D}/categories.json", "w"))
    print(f"saved {len(ids)} clips; positive frame rate {np.stack(masks).mean():.3f}", flush=True)


def harmonise():
    """Vaani-style edges: run the teacher (full v11g system with the round-1 students, w 0.70) on
    the selected clips and reconcile with AudioSet's events:
      keep    teacher events that overlap an AudioSet event (confirmed sound, Vaani-style edges);
      drop    teacher events with no AudioSet support (false positives);
      add     AudioSet events the teacher missed (all kept classes are short-sound types).
    -> audioset_strong_harmonised.npz (audio int16, mask uint8, ids)."""
    import torch
    from src.edge_correction import load_table
    from temp.pipeline_v11 import extract_items, final_events, postproc_models, verifier_view
    z = np.load(f"{D}/audioset_strong_selected.npz")
    caps = vaani_duration_caps()
    keep, _ = select(load_strong(), caps)
    post, table = postproc_models(), load_table("./checkpoints/edge_correction_v5.json")
    ov = lambda x, y: min(x[1], y[1]) > max(x[0], y[0])
    masks, stats = [], Counter()

    def items():
        for a, s in zip(z["audio"], z["ids"]):
            yield str(s), torch.from_numpy(a.astype(np.float32) / 32767)

    def sink(d):
        teacher = final_events(d, verifier_view([d]), 0, 0.70, post, table, gated_blend=True)
        ref = [(a, min(b, 10.0)) for a, b, _ in keep[d["id"]]]
        kept = []
        for e in teacher:
            hit = [r for r in ref if ov(e, r)]
            if not hit:
                continue
            span = max(r[1] for r in hit) - min(r[0] for r in hit)
            # keep the teacher's (Vaani-style) edges only when it is about the confirmed sound's size;
            # a much longer teacher event that merely touches a short AudioSet sound is not kept
            if e[1] - e[0] <= 2.0 * span + 0.3:
                kept.append(e)
            else:
                stats["too long"] += 1
        added = [r for r in ref if not any(ov(r, e) for e in kept)]
        stats["teacher"] += len(teacher); stats["kept"] += len(kept); stats["added"] += len(added)
        stats["dropped"] += len(teacher) - len(kept)
        m = np.zeros(1000, np.uint8)
        for a, b in kept + added:
            m[int(round(a * 100)):int(round(b * 100))] = 1
        masks.append(m)

    extract_items(items(), member_prefix=os.environ.get("MEMBER_PREFIX", "VS"), total=len(z["ids"]), sink=sink)
    np.savez(f"{D}/audioset_strong_harmonised.npz", audio=z["audio"], mask=np.stack(masks), ids=z["ids"])
    print(f"harmonised {len(masks)} clips; {dict(stats)}; positive frame rate {np.stack(masks).mean():.3f} "
          f"(AudioSet raw {z['mask'].mean():.3f})", flush=True)


if __name__ == "__main__":
    {"survey": survey, "build": build, "harmonise": harmonise}[sys.argv[1]]()
