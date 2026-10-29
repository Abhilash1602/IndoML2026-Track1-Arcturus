"""Lever 1: the v9a_d35_m60 recipe (real 1.5819) on the 717 test clips longer than 10 s.

Every edit so far (drops, refiner-vote merges, verifier drops) touched only clips <= 10 s; the
long clips carry 16.8% of v6's events and were left as v6. Here each long clip is processed in
v6's own windows (10 s, 2 s overlap, last window aligned to the end):
  * the 10 v6 members -> v6 probability (window probabilities averaged, as long_clip_prob does)
    and the 5 v4 members' logits and 144-d embeddings (window-averaged to full length);
  * the 5 v8_combo refiners -> presence / d_on / d_off, window-averaged, decoded on the full clip
    exactly like release/v7_inference.refiner_events;
  * the anchor SEDT -> events per window, shifted to clip time;
  * 80-band log-mel of the whole clip.
Then, on Natural-gated clips (v6-probability gate), the same steps as v9a_d35_m60:
  1. merge adjacent v6 events when >= 3 of the 5 refiners have one event spanning both;
  2. drop (unmerged) events the v7 refiner does not support (P1 rule) or the GBM false-positive
     filter flags at > 0.5 (P3 rule);
  3. verifier v1 (retrained exactly as for v9a) drops > 0.35 and merges > 0.6.
Short clips are copied byte-for-byte from submission_v9a_d35_m60.zip.

Usage: python -m temp.long_clips -> submission_v10_long.zip (+ submission_v10_long_combo.zip,
the same without the verifier step on long clips)
"""
import glob
import json
import os
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
from src.edge_correction import load_table
from src.refiner import BoundaryRefiner
from src.sedt import SEDT, predict_events
from temp.cache_verifier_features import V4
import temp.fp_filter as fpf
import temp.verifier as vf

FIVE = ["./checkpoints/refiner_val_all_x0.pth", "./checkpoints/refiner_val_train70_x0.pth"] + \
       [f"./checkpoints/refiner_val_all_x0_s{s}.pth" for s in (1, 2, 3)]  # the v8_combo voters; [0] = v7's refiner
S = vf.S
CACHE = f"{S}/verifier/long_clips.npz"


def windows(n, sr):
    win, stride = int(v6.WINDOW_SEC * sr), int((v6.WINDOW_SEC - v6.OVERLAP_SEC) * sr)
    starts = list(range(0, n - win + 1, stride))
    if not starts or starts[-1] + win < n:
        starts.append(n - win)
    return starts


@torch.no_grad()
def extract(paths):
    config = v6.make_config()
    dev = config.DEVICE
    names = [n for n, _, _, _ in v6.MEMBERS]
    models = v6.load_models("./checkpoints")
    feats = {}
    for name, m in zip(names, models):
        if name in V4:
            m.classifier.register_forward_hook(lambda mod, inp, out, key=name: feats.__setitem__(key, inp[0].detach()))
    v4_idx = [names.index(n) for n in V4]
    ridx = [names.index(n) for n in REFINER_ORDER]
    refiners = []
    for p in FIVE:
        r = BoundaryRefiner(n_members=5).to(dev)
        r.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)["state"])
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
    hop = config.HOP_LENGTH
    out = []
    for path in tqdm(paths, desc="long clips"):
        wav = v6.load_wav(path, config)
        n = wav.numel()
        T = n // hop
        T25 = int(np.ceil(T / 4))
        p6, c = np.zeros(T), np.zeros(T)
        L4, emb, c25 = np.zeros((5, T)), np.zeros((5, T25, 144)), np.zeros(T25)
        ref = np.zeros((len(FIVE), 3, T))
        sedt_ev = []
        for s in windows(n, config.SR):
            x = wav[s:s + int(v6.WINDOW_SEC * config.SR)].unsqueeze(0).to(dev)
            lg = torch.stack([m(x) for m in models])[:, 0].float()  # (10, 1000)
            a = s // hop
            b = min(a + lg.shape[1], T)
            p6[a:b] += torch.sigmoid(lg.mean(0)).cpu().numpy()[:b - a]
            L4[:, a:b] += lg[v4_idx].cpu().numpy()[:, :b - a]
            c[a:b] += 1
            a25 = int(round(a / 4))
            for j, key in enumerate(V4):
                e = F.avg_pool1d(feats[key].transpose(1, 2)[..., :1000], 4)[0].T.cpu().numpy()  # (250, 144)
                b25 = min(a25 + len(e), T25)
                emb[j, a25:b25] += e[:b25 - a25]
            c25[a25:min(a25 + 250, T25)] += 1
            mw = mel80(x)[..., :1000]
            for r, model in enumerate(refiners):
                o = model(lg[ridx][None], mw)
                ref[r, 0, a:b] += torch.sigmoid(o["presence"])[0].cpu().numpy()[:b - a]
                ref[r, 1, a:b] += o["d_on"][0].cpu().numpy()[:b - a]
                ref[r, 2, a:b] += o["d_off"][0].cpu().numpy()[:b - a]
            tok = dasheng(d_mel(x)).hidden_states.float()
            for on, off, cf in predict_events(sedt, tok, mw)[0]:
                sedt_ev.append((round(on + s / config.SR, 3), round(off + s / config.SR, 3), cf))
        c = np.maximum(c, 1)
        mel = mel80(wav.unsqueeze(0).to(dev))[0, :, :T].cpu().numpy()
        out.append(dict(id=os.path.splitext(os.path.basename(path))[0], sec=n / config.SR, p6=p6 / c, L4=L4 / c,
                        emb=emb / np.maximum(c25, 1)[None, :, None], ref=ref / c, sedt=sedt_ev, E=mel.mean(0)))
    return out


def merge_by_votes(ev, voters, need):
    pv = [sum(any(fpf.overlap(ev[k], f) and fpf.overlap(ev[k + 1], f) for f in v) for v in voters) for k in range(len(ev) - 1)]
    new, src = [], []
    for k, e in enumerate(ev):
        if new and pv[k - 1] >= need:
            new[-1] = (new[-1][0], max(new[-1][1], e[1]))
            src[-1] = None
        else:
            new.append(e)
            src.append(e)
    return new, src


def main():
    def load_zip(p):
        rows = [json.loads(l) for l in zipfile.ZipFile(p).read("predictions.jsonl").decode().splitlines()]
        return {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in rows}, [r["clip_id"] for r in rows]

    v6z, order = load_zip("submission_v6.zip")
    best, _ = load_zip("submission_v9a_d35_m60.zip")
    paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob("./data/test_audio/**/*.wav", recursive=True)}
    long_ids = sorted(k for k in order if sf.info(paths[k]).frames / sf.info(paths[k]).samplerate > 10.0)
    print(f"long clips: {len(long_ids)}", flush=True)
    data = extract([paths[k] for k in long_ids])
    table = load_table("./checkpoints/edge_correction_v5.json")

    # sanity: the recomputed v6 decode matches v6's own long-clip events
    same = np.mean([v6.to_events(d["p6"], table, 0.01, d["sec"]) == [{"onset": a, "offset": b} for a, b in v6z[d["id"]]] for d in data])
    print(f"recomputed v6 long-clip events identical to submission_v6: {same:.1%}", flush=True)

    # models, trained exactly as for v9a (same data, same seeds)
    fp_sets = {n: fpf.labelled_set(n) for n in ("gold_heldout", "official_natural", "official_synthetic")}
    fp_model = fpf.make_model().fit(np.concatenate([s[0] for s in fp_sets.values()]), np.concatenate([s[1] for s in fp_sets.values()]))
    sets = [vf.load_set(n) for n in ("gold_heldout", "official_natural", "official_synthetic")]
    ev_X, ev_E, ev_y, _, pr_X, pr_E, pr_y, _ = vf.build_training(sets)
    pca_e, pca_p = PCA(vf.N_PCA, random_state=0).fit(ev_E), PCA(vf.N_PCA, random_state=0).fit(pr_E)
    me = vf.gbm().fit(np.c_[ev_X, pca_e.transform(ev_E)], ev_y)
    mp = vf.gbm().fit(np.c_[pr_X, pca_p.transform(pr_E)], pr_y)

    D = dict(name="long", L=[d["L4"].astype(np.float32) for d in data], P=[1 / (1 + np.exp(-d["L4"].mean(0))) for d in data],
             E=[d["E"] for d in data], emb=[d["emb"].transpose(0, 1, 2).astype(np.float32) for d in data],
             sedt=[d["sedt"] for d in data])
    out_combo, out_full = {}, {}
    stats = dict(gated=0, merges=0, drops=0, vdrops=0, vmerges=0)
    for i, d in enumerate(data):
        cid, ev = d["id"], v6z[d["id"]]
        if not ev or not natural_gate(d["p6"], 0.01):
            out_combo[cid] = out_full[cid] = list(ev)
            continue
        stats["gated"] += 1
        voters = [[(e["onset"], e["offset"]) for e in refiner_events(d["ref"][r, 0], d["ref"][r, 1], d["ref"][r, 2], d["p6"], table, d["sec"])]
                  for r in range(len(FIVE))]
        # P3 flags on the unmerged v6 events (features from the v4 members, as fp_filter does)
        fpX = np.array([fpf.features(D["L"][i], D["P"][i], D["E"][i], ev, k) for k in range(len(ev))], np.float32)
        p3 = set(e for e, s in zip(ev, fp_model.predict_proba(fpX)[:, 1]) if s > 0.5)
        p1 = set(e for e in ev if not any(fpf.overlap(e, f) for f in voters[0]))
        merged, src = merge_by_votes(ev, voters, 3)
        stats["merges"] += len(ev) - len(merged)
        combo = [e for e, s in zip(merged, src) if not (s is not None and s in (p1 | p3))]
        stats["drops"] += len(merged) - len(combo)
        out_combo[cid] = combo
        # verifier v1 on the combo events (drop > 0.35, merge > 0.6)
        if not combo:
            out_full[cid] = combo
            continue
        er = vf.event_rows(D, i, combo)
        se = me.predict_proba(np.c_[np.array([f for f, _ in er], np.float32), pca_e.transform(np.array([e for _, e in er]))])[:, 1]
        pr = vf.pair_rows(D, i, combo)
        mset = set()
        if pr:
            sp = mp.predict_proba(np.c_[np.array([f for f, _, _ in pr], np.float32), pca_p.transform(np.array([e for _, e, _ in pr]))])[:, 1]
            mset = {(0, k) for (_, _, k), s in zip(pr, sp) if s > 0.6}
        dset = {(0, k) for k, s in enumerate(se) if s > 0.35}
        full = vf.apply_decisions([combo], dset, mset)[0]
        stats["vdrops"] += sum(1 for k in range(len(combo)) if (0, k) in dset)
        stats["vmerges"] += len(mset)
        out_full[cid] = full
    print(f"long clips gated {stats['gated']}/{len(data)}; refiner-vote merges {stats['merges']}, P1/P3 drops {stats['drops']}, "
          f"verifier drop flags {stats['vdrops']}, verifier merges {stats['vmerges']}", flush=True)
    for name, longs in (("submission_v10_long_combo.zip", out_combo), ("submission_v10_long.zip", out_full)):
        final = {c: (longs[c] if c in longs else best[c]) for c in order}
        assert len(final) == 5517 and all(0 <= a < b for e in final.values() for a, b in e)
        assert all(final[c] == best[c] for c in order if c not in longs)  # short clips = v9a_d35_m60
        with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("predictions.jsonl", "".join(json.dumps({"clip_id": c, "events": [
                {"onset": round(a, 3), "offset": round(b, 3)} for a, b in final[c]]}) + "\n" for c in order))
        n_long_v6 = sum(len(v6z[c]) for c in longs)
        print(f"{name}: long-clip events {n_long_v6} -> {sum(len(longs[c]) for c in longs)} "
              f"(clips changed {sum(longs[c] != v6z[c] for c in longs)}); total {sum(map(len, final.values()))}", flush=True)


if __name__ == "__main__":
    main()
