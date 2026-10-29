"""v11: the full recipe from raw test audio, with an optional refiner blend of the base probability.

Every rule that produced the real best (v10_long, 1.5845) is recomputed from the audio here instead
of being copied from earlier zips:
  base  P = sigmoid(w * ensemble_logit + (1 - w) * logit(mean presence of the 5 voter refiners)),
        w = 1 -> exactly v6 (short clips: mean of the 10 members' logits; long clips: v6's
        window-averaged probability). Decoded with v6's Viterbi + gated edge correction.
  gate  Natural-looking clips (mean edge contrast of the base events < 0.5).
  1.    merge adjacent base events when >= 3 of the 5 refiners have one event spanning both;
  2.    drop (unmerged) base events the v7 refiner (voter 0) does not support (P1 rule) or the
        GBM false-positive filter flags at > 0.5 (P3 rule);
  3.    verifier v1: drop > 0.35, merge > 0.6.
Short clips (<= 10 s) are padded to one 10 s window as release/v6_inference.ShortClips does; long
clips use v6's 10 s / 2 s-overlap windows (temp/long_clips.py). Validation out-of-fold: blend 0.5
+ recipe +0.0154 pooled vs +0.0126 for the recipe on the plain ensemble.

Usage: python -m temp.pipeline_v11 --w 1.0 0.75 0.5   (w = 1.0 must reproduce submission_v10_long)
"""
import argparse
import glob
import json
import os
import pickle
import zipfile

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.decomposition import PCA
from tqdm import tqdm

import release.v6_inference as v6
from release.v7_inference import REFINER_ORDER, natural_gate, refiner_events
from src.dataset import pad_or_trim
from src.edge_correction import load_table
from src.refiner import BoundaryRefiner
from src.sedt import SEDT, predict_events
from temp.cache_verifier_features import V4
from temp.long_clips import FIVE, merge_by_votes, windows

WIDE = [f"./checkpoints/refiner_val_all_wide_s{s}.pth" for s in range(5)]  # v12: 256-wide refiners (local +0.0019 recipe)
REFINER_SETS = {"base": FIVE, "wide": WIDE}
import temp.fp_filter as fpf
import temp.verifier as vf

CACHE = f"{vf.S}/verifier/pipeline_v11_features.pkl"


def path_items(paths, config):
    for path in paths:
        try:
            wav = v6.load_wav(path, config)
        except Exception:
            wav = torch.zeros(config.target_samples)
        yield os.path.splitext(os.path.basename(path))[0], wav


def extract(paths, refiner_paths=FIVE, member_prefix="VA", extra_prefix=None, replace_v4=None):
    config = v6.make_config()
    return extract_items(path_items(paths, config), refiner_paths, member_prefix, total=len(paths), extra_prefix=extra_prefix,
                         replace_v4=replace_v4)


def load_members(member_prefix="VA"):
    """v6's 10 members; member_prefix swaps the 5 validation-tuned files owbceVA_* for owbce<prefix>_*
    (e.g. the self-trained students), keeping MEMBERS order (and so the bit-exact logit mean)."""
    saved = v6.MEMBERS
    v6.MEMBERS = [(n, c, f.replace("owbceVA_", f"owbce{member_prefix}_"), o) for n, c, f, o in saved]
    try:
        return v6.load_models("./checkpoints")
    finally:
        v6.MEMBERS = saved


def load_extra_members(prefix):
    """The 5 validation-tuned positions of v6.MEMBERS with files owbce<prefix>_* (e.g. round-1 students),
    used as EXTRA ensemble members appended after the 10 (v4 indices are unchanged)."""
    saved = v6.MEMBERS
    v6.MEMBERS = [(n, c, f.replace("owbceVA_", f"owbce{prefix}_"), o) for n, c, f, o in saved if "owbceVA_" in f]
    try:
        return v6.load_models("./checkpoints")
    finally:
        v6.MEMBERS = saved


@torch.no_grad()
def extract_items(items, refiner_paths=FIVE, member_prefix="VA", total=None, sink=None, extra_prefix=None, replace_v4=None):
    """items: iterable of (clip_id, mono 16 kHz waveform tensor). Clips <= 10 s are padded to one
    10 s window (as v6 ShortClips); longer clips use v6's 10 s / 2 s-overlap windows.
    sink: optional callable(record); records are then streamed to it instead of returned."""
    config = v6.make_config()
    dev = config.DEVICE
    names = [n for n, _, _, _ in v6.MEMBERS]
    models = load_members(member_prefix) + (load_extra_members(extra_prefix) if extra_prefix else [])
    # replace_v4 (Oct 15): the ensemble mean uses the 5 validation-tuned positions + 5 students of the v4
    # members (files owbce<replace_v4>_*, in the same position order); the ORIGINAL v4 members are still run
    # and still provide L4 / embeddings / refiner inputs, so the event-stage features are unchanged.
    rep = load_extra_members(replace_v4) if replace_v4 else []
    va_idx = [i for i, (_, _, f, _) in enumerate(v6.MEMBERS) if "owbceVA_" in f]
    feats = {}
    for name, m in zip(names, models):
        if name in V4:
            m.classifier.register_forward_hook(lambda mod, inp, out, key=name: feats.__setitem__(key, inp[0].detach()))
    v4_idx = [names.index(n) for n in V4]
    ridx = [names.index(n) for n in REFINER_ORDER]
    refiners = []
    for p in refiner_paths:
        rck = torch.load(p, map_location="cpu", weights_only=False)
        r = BoundaryRefiner(n_members=5, **rck.get("kwargs", {})).to(dev)
        r.load_state_dict(rck["state"])
        refiners.append(r.eval())
    ck = torch.load(vf.SEDT_CKPT, map_location="cpu", weights_only=False)
    sedt = SEDT(**ck.get("kwargs", {})).to(dev)
    sedt.load_state_dict(ck["state"])
    sedt.eval()
    from transformers import AutoModel
    dasheng = AutoModel.from_pretrained("mispeech/dasheng-base", trust_remote_code=True).to(dev).eval()
    d_mel = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512, hop_length=160,
                                             f_min=0, f_max=8000, n_mels=64, center=True),
        torchaudio.transforms.AmplitudeToDB(top_db=120)).to(dev)
    mel80 = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80),
        torchaudio.transforms.AmplitudeToDB()).to(dev)
    hop, win = config.HOP_LENGTH, int(v6.WINDOW_SEC * config.SR)
    out = []
    for clip_id, wav in tqdm(items, desc="clips", total=total):
        dur = wav.numel() / config.SR
        short = dur <= config.MAX_DURATION
        if short:
            wav = pad_or_trim(wav, config.target_samples)  # = v6 ShortClips
            starts = [0]
        else:
            starts = windows(wav.numel(), config.SR)
        n = wav.numel()
        T = n // hop
        T25 = int(np.ceil(T / 4))
        lmean, p6, c = np.zeros(T), np.zeros(T), np.zeros(T)
        L4, emb, c25 = np.zeros((5, T)), np.zeros((5, T25, 144)), np.zeros(T25)
        ref = np.zeros((len(refiner_paths), 3, T))
        sedt_ev = []
        for s in starts:
            x = wav[s:s + win].unsqueeze(0).to(dev)
            lg = torch.stack([m(x) for m in models])[:, 0].float()  # (10, frames)
            lgm = (torch.cat([lg[va_idx], torch.stack([m(x) for m in rep])[:, 0].float()]) if rep else lg).mean(0)
            a = s // hop
            b = min(a + lg.shape[1], T)
            lmean[a:b] += lgm.cpu().numpy()[:b - a]
            p6[a:b] += torch.sigmoid(lgm).cpu().numpy()[:b - a]
            L4[:, a:b] += lg[v4_idx].cpu().numpy()[:, :b - a]
            c[a:b] += 1
            a25 = int(round(a / 4))
            for j, key in enumerate(V4):
                e = F.avg_pool1d(feats[key].transpose(1, 2)[..., :1000], 4)[0].T.cpu().numpy()
                b25 = min(a25 + len(e), T25)
                emb[j, a25:b25] += e[:b25 - a25]
            c25[a25:min(a25 + 250, T25)] += 1
            mw = mel80(x)[..., :1000]
            for r, model in enumerate(refiners):
                o = model(lg[ridx][None, :, :1000], mw)
                bb = min(a + 1000, T)
                ref[r, 0, a:bb] += torch.sigmoid(o["presence"])[0].cpu().numpy()[:bb - a]
                ref[r, 1, a:bb] += o["d_on"][0].cpu().numpy()[:bb - a]
                ref[r, 2, a:bb] += o["d_off"][0].cpu().numpy()[:bb - a]
            tok = dasheng(d_mel(x)).hidden_states.float()
            for on, off, cf in predict_events(sedt, tok, mw)[0]:
                sedt_ev.append((round(on + s / config.SR, 3), round(off + s / config.SR, 3), cf))
        c = np.maximum(c, 1)
        mel = mel80(wav.unsqueeze(0).to(dev))[0, :, :T].cpu().numpy()
        rec = (dict(id=clip_id, dur=dur, short=short, T=T,
                        lmean=(lmean / c).astype(np.float32), p6=(p6 / c).astype(np.float32), L4=(L4 / c).astype(np.float32),
                        emb=(emb / np.maximum(c25, 1)[None, :, None]).astype(np.float16), ref=(ref / c).astype(np.float32),
                        sedt=sedt_ev, E=mel.mean(0).astype(np.float32)))
        if sink is not None:
            sink(rec)
        else:
            out.append(rec)
    return out


POSTPROC = "./checkpoints/postproc_v11.pkl"


def postproc_models():
    """The GBM false-positive filter and verifier v1 (+ its PCAs). Fitted once, exactly as for
    v9a / v10 (same data, same seeds), on Gold held-out + official validation features, and saved
    to checkpoints/postproc_v11.pkl so inference does not need the training caches."""
    if os.path.exists(POSTPROC):
        return pickle.load(open(POSTPROC, "rb"))
    fp_sets = {n: fpf.labelled_set(n) for n in ("gold_heldout", "official_natural", "official_synthetic")}
    fp_model = fpf.make_model().fit(np.concatenate([s[0] for s in fp_sets.values()]), np.concatenate([s[1] for s in fp_sets.values()]))
    sets = [vf.load_set(n) for n in ("gold_heldout", "official_natural", "official_synthetic")]
    ev_X, ev_E, ev_y, _, pr_X, pr_E, pr_y, _ = vf.build_training(sets)
    pca_e, pca_p = PCA(vf.N_PCA, random_state=0).fit(ev_E), PCA(vf.N_PCA, random_state=0).fit(pr_E)
    me = vf.gbm().fit(np.c_[ev_X, pca_e.transform(ev_E)], ev_y)
    mp = vf.gbm().fit(np.c_[pr_X, pca_p.transform(pr_E)], pr_y)
    out = (fp_model, pca_e, pca_p, me, mp)
    pickle.dump(out, open(POSTPROC, "wb"))
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


BAND = None  # (lo, hi): blend only frames whose ensemble probability lies in (lo, hi); None = every frame
VDROP, FPTHR = 0.35, 0.5  # verifier drop / GBM FP-filter thresholds (--vdrop / --fpthr)
VMERGE = 0.6  # verifier merge threshold (--vmerge)
P1_VOTES = None  # None = P1 drops events refiner voter 0 does not overlap; k = fewer than k of the 5 voters overlap (--p1-votes)
LONG_DICE = None  # (min_dur, t, alpha): Dice-oriented edges for long events (--long-dice)
SEBB_DROP = None  # (step, abs, rel, tau): drop events no confident cSEBB box supports (--sebb-drop)
V3 = None  # (verifier3 dict, test new-model logits) when --verifier3: replaces verifier v1's drop scores
V5 = None  # (mode, verifier5 model, test fold logits) when --verifier5: v5 replaces ("v5") or averages ("avg") v1 drop scores
VR = None  # verifier trained on v4-base + recipe-analog events (temp/verifier_recipe.py fit C) when --verifier-recipe


def sebb_drop(evs, p, step, a, r, tau, safe):
    """cSEBB confidence as a drop rule (Oct 12, temp/eval_csebb2.py hybrid_drop: cross-fitted on
    validation Natural + Synthetic +0.0014, P 0.905): an event is dropped when every cSEBB box
    overlapping it has confidence < tau, unless the confidence guard protects it."""
    import sed_scores_eval
    from sebbs.csebbs import CSEBBsPredictor
    if not evs:
        return evs
    ts = np.arange(len(p) + 1) * 0.01
    df = sed_scores_eval.io.create_score_dataframe(p.reshape(-1, 1).astype(np.float64), timestamps=ts, event_classes=["noise"])
    boxes = CSEBBsPredictor(step_filter_length=step, merge_threshold_abs=a, merge_threshold_rel=r,
                            sound_classes=["noise"]).predict({"c": df})["c"]
    keep = []
    for e in evs:
        conf = max([cf for x, y, _, cf in boxes if min(y, e[1]) > max(x, e[0])], default=0.0)
        if conf >= tau or safe(e):
            keep.append(e)
    return keep


def long_dice_edges(evs, p, min_dur, t, alpha):
    """Events >= min_dur s: extend each edge while p > t (or shrink while p < t), by at most
    alpha * 0.2 * duration -- inside the event-F1 tolerance of a long event, so mostly a Dice move.
    Local validation (v4 base, Oct 10): (1.5, 0.3, 1.0) gave pooled +0.0026."""
    out = []
    for a, b in evs:
        d = b - a
        if d < min_dur:
            out.append((a, b))
            continue
        lim = alpha * 0.2 * d
        i, j = int(round(a * 100)), int(round(b * 100))
        lo, hi = max(0, int(round((a - lim) * 100))), min(len(p), int(round((b + lim) * 100)))
        ii = i
        while ii - 1 >= lo and p[ii - 1] > t:
            ii -= 1
        if ii == i:
            while ii < j - 1 and p[ii] < t and (ii - i) < lim * 100:
                ii += 1
        jj = j
        while jj < hi and p[jj] > t:
            jj += 1
        if jj == j:
            while jj > ii + 1 and p[jj - 1] < t and (j - jj) < lim * 100:
                jj -= 1
        out.append((ii / 100, jj / 100))
    return out


def base_prob(d, w, gated_only=False):
    """gated_only (v11g): blend only clips whose pure-v6 events look Natural; synthetic-looking
    clips keep v6 exactly (real v11_w50: Natural F1 +0.0087 but synthetic F1 -0.0024)."""
    if gated_only and w != 1.0 and not natural_gate(base_prob(d, 1.0), 0.01):
        w = 1.0
    pe = 1 / (1 + np.exp(-d["lmean"])) if d["short"] else d["p6"]
    if w == 1.0:
        return pe
    ens = d["lmean"] if d["short"] else logit(d["p6"])
    pb = 1 / (1 + np.exp(-(w * ens + (1 - w) * logit(d["ref"][:, 0].mean(0)))))
    if BAND is not None:
        pb = np.where((pe > BAND[0]) & (pe < BAND[1]), pb, pe)
    return pb


def final_events(d, D, i, w, post, table, gated_blend=True, no_gate=False, st=None, guard=None):
    """The full v11 recipe for one clip (d = extract_items record; D/i = its verifier view).
    Returns the final [(onset, offset)] list."""
    fp_model, pca_e, pca_p, me, mp = post
    st = st if st is not None else dict(gated=0, merges=0, drops=0, vdrops=0, vmerges=0)
    P = base_prob(d, w, gated_blend)
    clip_sec = min(d["dur"], 10.0) if d["short"] else d["dur"]
    ev = [(e["onset"], e["offset"]) for e in v6.to_events(P, table, 0.01, clip_sec if d["dur"] > 0 else None)]
    if not ev or (not no_gate and not natural_gate(P, 0.01)):
        return ev
    st["gated"] += 1
    cut = (lambda x: x[:1000]) if d["short"] else (lambda x: x)
    voters = [[(e["onset"], e["offset"]) for e in refiner_events(cut(d["ref"][r, 0]), cut(d["ref"][r, 1]), cut(d["ref"][r, 2]),
                                                                 cut(P), table, clip_sec)]
              for r in range(d["ref"].shape[0])]
    fpX = np.array([fpf.features(D["L"][i], D["P"][i], D["E"][i], ev, k) for k in range(len(ev))], np.float32)
    conf = lambda e: float(P[int(e[0] * 100):max(int(e[1] * 100), int(e[0] * 100) + 1)].mean())
    safe = (lambda e: guard is not None and conf(e) >= guard)  # confident events are never dropped
    p3 = set(e for e, s in zip(ev, fp_model.predict_proba(fpX)[:, 1]) if s > FPTHR and not safe(e))
    if P1_VOTES is None:
        p1 = set(e for e in ev if not any(fpf.overlap(e, f) for f in voters[0]) and not safe(e))
    else:
        p1 = set(e for e in ev if sum(any(fpf.overlap(e, f) for f in v) for v in voters) < P1_VOTES and not safe(e))
    merged, src = merge_by_votes(ev, voters, 3)
    st["merges"] += len(ev) - len(merged)
    combo = [e for e, s in zip(merged, src) if not (s is not None and s in (p1 | p3))]
    st["drops"] += len(merged) - len(combo)
    if combo:
        er = vf.event_rows(D, i, combo)
        if V3 is not None:
            from temp.verifier3 import new_features
            vm, tl = V3
            Xn = np.array([new_features(D["L"][i], tl[d["id"]], a, b) for a, b in combo], np.float32)
            se = vm["model"].predict_proba(np.c_[np.array([f for f, _ in er], np.float32),
                                                 vm["pca"].transform(np.array([e for _, e in er])), Xn])[:, 1]
        else:
            se = me.predict_proba(np.c_[np.array([f for f, _ in er], np.float32), pca_e.transform(np.array([e for _, e in er]))])[:, 1]
        if VR is not None:
            se = VR["model"].predict_proba(np.c_[np.array([f for f, _ in er], np.float32),
                                                 VR["pca"].transform(np.array([e for _, e in er]))])[:, 1]
        if V5 is not None:
            mode, m5, tl5 = V5
            L5 = tl5[d["id"]].astype(np.float32)
            P5 = 1 / (1 + np.exp(-L5.mean(0)))
            E5 = D["E"][i][:L5.shape[1]]
            s5 = m5.predict_proba(np.array([fpf.features(L5, P5, E5, combo, k) for k in range(len(combo))], np.float32))[:, 1]
            se = s5 if mode == "v5" else (se + s5) / 2
        pr = vf.pair_rows(D, i, combo)
        mset = set()
        if pr:
            sp = mp.predict_proba(np.c_[np.array([f for f, _, _ in pr], np.float32), pca_p.transform(np.array([e for _, e, _ in pr]))])[:, 1]
            mset = {(0, k) for (_, _, k), s in zip(pr, sp) if s > VMERGE}
        dset = {(0, k) for k, s in enumerate(se) if s > VDROP and not safe(combo[k])}
        st["vdrops"] += len(dset)
        st["vmerges"] += len(mset)
        combo = vf.apply_decisions([combo], dset, mset)[0]
    if LONG_DICE is not None:
        combo = long_dice_edges(combo, P, *LONG_DICE)
    if SEBB_DROP is not None:
        before = len(combo)
        combo = sebb_drop(combo, P, *SEBB_DROP, safe)
        st["sdrops"] = st.get("sdrops", 0) + before - len(combo)
    return combo


def verifier_view(data):
    return dict(name="test", L=[d["L4"] for d in data], P=[1 / (1 + np.exp(-d["L4"].mean(0))) for d in data],
                E=[d["E"] for d in data], emb=[d["emb"].astype(np.float32) for d in data], sedt=[d["sedt"] for d in data])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, nargs="+", default=[1.0, 0.75, 0.5])
    ap.add_argument("--test-dir", default="./data/test_audio")
    ap.add_argument("--gated-blend", action="store_true", help="blend only Natural-looking clips (v11g)")
    ap.add_argument("--no-gate", action="store_true",
                    help="one pipeline for all clips: blend and event decisions on every clip, no Natural/Synthetic routing (v13ng)")
    ap.add_argument("--out-dir", default=".", help="where submission_v11*.zip is written")
    ap.add_argument("--cache", default=None, help="feature cache (reused when it exists)")
    ap.add_argument("--refiners", default="base", choices=sorted(REFINER_SETS), help="voter refiner set (v11: base, v12: wide)")
    ap.add_argument("--member-prefix", default="VA", help="validation-tuned member files owbce<prefix>_* (VA = v6; VS = self-trained)")
    ap.add_argument("--tag", default=None, help="output name prefix (default: v11 / v12 / v13ng)")
    ap.add_argument("--extra-prefix", default=None, help="append the 5 owbce<prefix>_* members as extra ensemble members")
    ap.add_argument("--replace-v4", default=None, help="ensemble mean uses owbce<prefix>_* students in place of the v4 members (originals still give features)")
    ap.add_argument("--guard", type=float, default=None, help="never drop events whose mean ensemble probability >= guard")
    ap.add_argument("--band", type=float, nargs=2, default=None, help="blend only frames with ensemble probability in (lo, hi)")
    ap.add_argument("--vdrop", type=float, default=0.35, help="verifier drop threshold")
    ap.add_argument("--vmerge", type=float, default=0.6, help="verifier merge threshold")
    ap.add_argument("--verifier3", action="store_true", help="verifier v3 drop scores (checkpoints/verifier3.pkl + work/verifier3/test_logits.pkl)")
    ap.add_argument("--verifier5", choices=["v5", "avg"], default=None, help="verifier v5 (cross-fitted) drop scores: replace or average v1")
    ap.add_argument("--verifier-recipe", action="store_true", help="verifier trained on v4-base + recipe-analog events (checkpoints/verifier_recipe.pkl)")
    ap.add_argument("--p1-votes", type=int, default=None, help="P1 drop when fewer than K of the 5 refiner voters overlap")
    ap.add_argument("--sebb-drop", type=float, nargs=4, default=None, metavar=("STEP", "ABS", "REL", "TAU"),
                    help="drop events that no cSEBB box with confidence >= TAU supports")
    ap.add_argument("--fpthr", type=float, default=0.5, help="GBM false-positive filter threshold")
    ap.add_argument("--long-dice", type=float, nargs=3, default=None, metavar=("MIN_DUR", "T", "ALPHA"),
                    help="Dice-oriented edges for long events")
    args = ap.parse_args()
    assert not (args.no_gate and args.gated_blend), "--no-gate blends every clip"
    global BAND, VDROP, FPTHR, LONG_DICE, SEBB_DROP, VMERGE, P1_VOTES, V3, V5, VR
    BAND = tuple(args.band) if args.band else None
    VDROP, FPTHR, VMERGE, P1_VOTES = args.vdrop, args.fpthr, args.vmerge, args.p1_votes
    LONG_DICE = tuple(args.long_dice) if args.long_dice else None
    SEBB_DROP = tuple(args.sebb_drop) if args.sebb_drop else None
    if args.verifier_recipe:
        VR = pickle.load(open("./checkpoints/verifier_recipe.pkl", "rb"))
    if args.verifier5:
        V5 = (args.verifier5, pickle.load(open("./checkpoints/verifier5.pkl", "rb"))["model"],
              pickle.load(open("./work/folds/test_logits.pkl", "rb")))
    if args.verifier3:
        V3 = (pickle.load(open("./checkpoints/verifier3.pkl", "rb")), pickle.load(open("./work/verifier3/test_logits.pkl", "rb")))
    if args.cache is None:
        args.cache = CACHE if (args.refiners == "base" and args.member_prefix == "VA") else \
            CACHE.replace(".pkl", f"_{args.refiners}_{args.member_prefix}{'_x' + args.extra_prefix if args.extra_prefix else ''}"
                                  f"{'_r' + args.replace_v4 if args.replace_v4 else ''}.pkl")
    paths = sorted(glob.glob(os.path.join(args.test_dir, "**", "*.wav"), recursive=True))
    if os.path.exists(args.cache):
        data = pickle.load(open(args.cache, "rb"))
    else:
        data = extract(paths, REFINER_SETS[args.refiners], args.member_prefix, args.extra_prefix, args.replace_v4)
        pickle.dump(data, open(args.cache, "wb"))
    assert sorted(d["id"] for d in data) == sorted(os.path.splitext(os.path.basename(p))[0] for p in paths), \
        "feature cache does not match --test-dir"
    table = load_table("./checkpoints/edge_correction_v5.json")

    post = postproc_models()
    D = verifier_view(data)

    for w in args.w:
        final, st = {}, dict(gated=0, merges=0, drops=0, vdrops=0, vmerges=0)
        for i, d in enumerate(data):
            final[d["id"]] = final_events(d, D, i, w, post, table, args.gated_blend, args.no_gate, st, args.guard)
        order = ([json.loads(l)["clip_id"] for l in zipfile.ZipFile("submission_v6.zip").read("predictions.jsonl").decode().splitlines()]
                 if os.path.exists("submission_v6.zip") else sorted(final))
        assert set(final) == set(order) and all(0 <= a < b for e in final.values() for a, b in e)
        name = os.path.join(args.out_dir, f"submission_{args.tag or ('v13ng' if args.no_gate else ('v11' if args.refiners == 'base' else 'v12') + ('g' if args.gated_blend else ''))}_w{int(round(w * 100))}.zip")
        with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("predictions.jsonl", "".join(json.dumps({"clip_id": c, "events": [
                {"onset": round(a, 3), "offset": round(b, 3)} for a, b in final[c]]}) + "\n" for c in order))
        msg = ""
        if os.path.exists("submission_v10_long.zip"):
            ref = {json.loads(l)["clip_id"]: [(e["onset"], e["offset"]) for e in json.loads(l)["events"]]
                   for l in zipfile.ZipFile("submission_v10_long.zip").read("predictions.jsonl").decode().splitlines()}
            msg = f"; clips identical to v10_long {np.mean([final[c] == ref.get(c) for c in order]):.1%}"
        print(f"w={w}: gated {st['gated']}, merges {st['merges']}, P1/P3 drops {st['drops']}, verifier drops {st['vdrops']}, "
              f"verifier merges {st['vmerges']}, cSEBB drops {st.get('sdrops', 0)}; events {sum(map(len, final.values()))}{msg} -> {name}", flush=True)


if __name__ == "__main__":
    main()
