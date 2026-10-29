"""Event-verifier inputs (plan Phase 1/2).

For each clip (10 s, padded/trimmed like everywhere else):
  * emb  -- the input to `classifier` (conformer output, 144-d) of the 5 v4 members
            (atst_stage2, dasheng_owB, dasheng_owB_t, beats_s2, beats_owA), average-pooled
            to 25 Hz -> (5, 250, 144) f16. The v4 members never trained on Gold held-out or
            the official validation set, and v6 contains them, so the features mean the same
            thing on test.
  * logits -- (test only) all 10 v6 members' frame logits (10, 1000) f16, in v6 MEMBERS order,
            so the v6 probability and decode are reproduced without re-running the models.
  * mel / tok -- (test only) 80-band 10 ms log-mel and pretrained Dasheng tokens, as in
            scratchpad/sedt/ for the other sets.
Sets: test (short clips <= 10 s of ./data/test_audio, sorted), gold_heldout, official_natural,
official_synthetic (same order as scratchpad/sedt/*_ids.json, asserted).

Usage: VAANI_DEVICE=cuda:0 python -m temp.cache_verifier_features [set ...]
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import release.v6_inference as v6
from src.dataset import OfficialValidationDataset
from temp.cache_sedt_features import VAL_ROOT, gold_sets
from temp.train_refiner_val import S

OUT = os.path.join(S, "verifier")
V4 = ["atst_stage2", "dasheng_owB", "dasheng_owB_t", "beats_s2", "beats_owA"]
FRAMES = 1000


def test_files(config):
    files = sorted(glob.glob(os.path.join("./data/test_audio", "**", "*.wav"), recursive=True))
    short = []
    for f in files:
        try:
            d = sf.info(f).frames / sf.info(f).samplerate
        except Exception:
            d = 0.0
        if d <= config.MAX_DURATION:
            short.append(f)
    return short


def main():
    os.makedirs(OUT, exist_ok=True)
    config = v6.make_config()
    device = config.DEVICE
    wanted = sys.argv[1:] or ["test", "official_natural", "official_synthetic", "gold_heldout"]
    names = [n for n, _, _, _ in v6.MEMBERS]
    models = v6.load_models("./checkpoints")
    feats = {}
    for name, m in zip(names, models):
        if name in V4:
            m.classifier.register_forward_hook(lambda mod, inp, out, key=name: feats.__setitem__(key, inp[0].detach()))
    from transformers import AutoModel
    dasheng = AutoModel.from_pretrained("mispeech/dasheng-base", trust_remote_code=True).to(device).eval()
    d_mel = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512, hop_length=160,
                                             f_min=0, f_max=8000, n_mels=64, center=True),
        torchaudio.transforms.AmplitudeToDB(top_db=120)).to(device)
    mel80 = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80),
        torchaudio.transforms.AmplitudeToDB()).to(device)
    for set_name in wanted:
        if os.path.exists(f"{OUT}/{set_name}_emb.npy"):
            continue
        if set_name == "test":
            ds = v6.ShortClips(test_files(config), config)
        elif set_name == "gold_heldout":
            ds = gold_sets(config)["gold_heldout"]
        else:
            ds = OfficialValidationDataset(VAL_ROOT, config, subset=set_name.split("_")[1])
        is_test = set_name == "test"
        run_models = models if is_test else [m for n, m in zip(names, models) if n in V4]
        n = len(ds)
        emb = np.lib.format.open_memmap(f"{OUT}/{set_name}_emb.npy.part", "w+", np.float16, (n, len(V4), 250, 144))
        if is_test:
            logits = np.zeros((n, len(models), FRAMES), np.float16)
            mel = np.zeros((n, 80, FRAMES), np.float16)
            tok = np.lib.format.open_memmap(f"{OUT}/{set_name}_tok.npy.part", "w+", np.float16, (n, 250, 768))
        ids, pos = [], 0
        with torch.no_grad():
            for batch in tqdm(DataLoader(ds, batch_size=16, shuffle=False, num_workers=config.NUM_WORKERS), desc=set_name):
                x = batch[0].to(device)
                b = x.shape[0]
                lg = torch.stack([m(x) for m in run_models])  # (M, B, T)
                for j, key in enumerate(V4):
                    e = feats[key].transpose(1, 2)[..., :FRAMES]  # (B, 144, T)
                    emb[pos:pos + b, j] = F.avg_pool1d(e, 4).transpose(1, 2).cpu().numpy().astype(np.float16)
                if is_test:
                    logits[pos:pos + b] = lg.permute(1, 0, 2)[..., :FRAMES].cpu().numpy()
                    mm = mel80(x)[..., :FRAMES]
                    mel[pos:pos + b, :, :mm.shape[-1]] = mm.cpu().numpy()
                    tok[pos:pos + b] = dasheng(d_mel(x)).hidden_states.cpu().numpy().astype(np.float16)
                ids += [str(c) for c in batch[-1]] if is_test else [str(c) for c in batch[2]]
                pos += b
        assert pos == n
        emb.flush(); del emb
        os.replace(f"{OUT}/{set_name}_emb.npy.part", f"{OUT}/{set_name}_emb.npy")
        if is_test:
            tok.flush(); del tok
            os.replace(f"{OUT}/{set_name}_tok.npy.part", f"{OUT}/{set_name}_tok.npy")
            np.save(f"{OUT}/{set_name}_logits.npy", logits)
            np.save(f"{OUT}/{set_name}_mel.npy", mel)
        else:
            ref = json.load(open(f"{S}/sedt/{set_name}_ids.json"))
            assert ids == ref, f"{set_name}: order differs from scratchpad/sedt"
        json.dump(ids, open(f"{OUT}/{set_name}_ids.json", "w"))
        print(f"{set_name}: {n} clips cached", flush=True)


if __name__ == "__main__":
    main()
