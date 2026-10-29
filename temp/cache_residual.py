"""Oct 14, residual Step 0 (run with the Track 2 env: miniconda3/envs/t2sep_bw/bin/python, GPU1):
public pretrained MossFormerGAN_SE_16K (ClearerVoice, Apache-2.0; NOT the Track 2 fine-tunes) on the
dumped 10 s clips -> enhanced speech s and residual r = x - s, saved as 80-band 10 ms log-mels on the
same transform/grid as work/refiner_<set>.npz["mel"] (MelSpectrogram n_fft 1024, hop 160, 80 mels, dB).
Output: work/residual/<set>_{res,spe}_mel.npy (N, 80, 1000) float16.
Usage: CUDA_VISIBLE_DEVICES=1 <t2sep_bw python> temp/cache_residual.py [set ...]"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
from clearvoice.models.mossformer_gan_se.generator import SyncANet
from clearvoice.utils.misc import istft, power_compress, power_uncompress, stft

CKPT = os.environ.get("MOSSFORMERGAN_CKPT", "MossFormerGAN_SE_16K/last_best_checkpoint.pt")
W = "./work/residual"
CFG = SimpleNamespace(win_type="hamming", win_len=400, win_inc=100, fft_len=400)


def load_model(dev):
    m = SyncANet(num_channel=64, num_features=CFG.fft_len // 2 + 1)
    ck = torch.load(CKPT, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in ck.get("model", ck).items()}
    missing, _ = m.load_state_dict(sd, strict=False)
    assert not missing, missing[:5]
    return m.to(dev).eval()


@torch.no_grad()
def enhance(m, x):
    T = x.shape[-1]
    nframe = int(np.ceil(T / CFG.win_inc)); pad = nframe * CFG.win_inc - T
    if pad:
        x = torch.cat([x, x[:, :pad]], dim=-1)
    norm = torch.sqrt(x.shape[-1] / (x.pow(2).sum(-1, keepdim=True) + 1e-8))
    spec = power_compress(stft(x * norm, CFG, center=True, periodic=True, onesided=True).to(torch.float32)).permute(0, 1, 3, 2)
    re, im = m(spec)
    out = istft(power_uncompress(re.permute(0, 1, 3, 2), im.permute(0, 1, 3, 2)).squeeze(1), CFG, center=True, periodic=True, onesided=True)
    return (out / norm)[:, :T]


def main():
    dev = "cuda:0"
    m = load_model(dev)
    mel_tf = torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80),
                                 torchaudio.transforms.AmplitudeToDB()).to(dev)
    for s in (sys.argv[1:] or ["gold_heldout", "official_natural", "official_synthetic"]):
        A = np.load(f"{W}/{s}_audio.npy", mmap_mode="r")
        res, spe = [], []
        for b in range(0, len(A), 2):
            x = torch.from_numpy(np.asarray(A[b:b + 2], np.float32) / 32767).to(dev)
            y = enhance(m, x)
            r = x - y
            spe.append(mel_tf(y)[..., :1000].half().cpu().numpy())
            res.append(mel_tf(r)[..., :1000].half().cpu().numpy())
        np.save(f"{W}/{s}_res_mel.npy", np.concatenate(res)); np.save(f"{W}/{s}_spe_mel.npy", np.concatenate(spe))
        print(s, len(A), flush=True)


if __name__ == "__main__":
    main()
