"""Oct 14: the three clean VG models' frame logits on the test audio, windowed exactly as
temp/pipeline_v11.extract_items (clips <= 10 s padded to one 10 s window; longer clips in v6's
10 s / 2 s-overlap windows, logits averaged where windows overlap; T = samples // hop).

Output: work/verifier3/test_logits.pkl = {clip_id: (3, T) float16} in temp.verifier3.NEW order.
Usage: CUDA_VISIBLE_DEVICES=1 VAANI_DEVICE=cuda:0 python -m temp.cache_vg_test [--test-dir ./data/test_audio]
"""
import glob
import os
import pickle
import sys

import numpy as np
import torch

from src.dataset import pad_or_trim
from temp.cache_vg_logits import build
from temp.long_clips import windows
from temp.pipeline_v11 import path_items
from temp.verifier3 import NEW
import release.v6_inference as v6


@torch.no_grad()
def main():
    test_dir = sys.argv[sys.argv.index("--test-dir") + 1] if "--test-dir" in sys.argv else "./data/test_audio"
    paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.wav"), recursive=True))
    config = v6.make_config()
    hop, win = config.HOP_LENGTH, config.target_samples
    out = {}
    for mi, name in enumerate(NEW):
        model, cfg = build(name)
        for cid, wav in path_items(paths, config):
            n = wav.numel()
            short = n / config.SR <= config.MAX_DURATION
            if short:
                wav = pad_or_trim(wav, win)
                n = wav.numel()
            starts = [0] if short else windows(n, config.SR)
            T = n // hop
            acc, cnt = np.zeros(T, np.float32), np.zeros(T, np.float32)
            for s in starts:
                x = wav[s:s + win].unsqueeze(0).to(cfg.DEVICE)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(x).float()[0].cpu().numpy()
                a = s // hop
                b = min(a + len(lg), T)
                acc[a:b] += lg[:b - a]
                cnt[a:b] += 1
            out.setdefault(cid, np.zeros((len(NEW), T), np.float16))[mi] = (acc / np.maximum(cnt, 1)).astype(np.float16)
        print(f"{name}: {len(paths)} test clips", flush=True)
        del model
        torch.cuda.empty_cache()
    pickle.dump(out, open("./work/verifier3/test_logits.pkl", "wb"))
    print(f"saved {len(out)} clips -> work/verifier3/test_logits.pkl", flush=True)


if __name__ == "__main__":
    main()
