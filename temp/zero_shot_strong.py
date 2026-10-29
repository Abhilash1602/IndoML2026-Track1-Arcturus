"""Plan Step 0b: how much boundary knowledge do the AudioSet-Strong backbones
bring, before any fine-tuning? For each backbone: frozen 40 ms frame embeddings
-> a linear probe (logistic regression to "any noise") fitted ONLY on Gold
training clips (never validation) -> upsample to 10 ms -> fixed Viterbi decodes
-> official metric on the validation set. Our best single model
(dasheng_owA_t, cached logits) is scored with the same decodes for reference.

Usage: python -m temp.zero_shot_strong [ATST-F BEATs ASIT ...]   (name:stage, e.g. ATST-F:ssl)
"""
import multiprocessing as mp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import Config
from src.dataset import OfficialValidationDataset
from src.strong_backbones import build_strong
from src.utils import extract_events_from_mask, extract_events_viterbi, viterbi_decode
from temp.official_metric import macro_segment_dice, micro_event_f1
from train import make_gold_loaders

VAL_ROOT = "./data/official_validation_raw/validation"
SCRATCH = "./work"
N_GOLD = 3000
DECODES = [(0.95, 0.7), (0.9065, 0.8914), (0.9802, 0.9489), (0.998, 0.98)]
G = {}


@torch.no_grad()
def embed(model, loader, device):
    feats, masks = [], []
    for audio, mask, _ in tqdm(loader, leave=False):
        emb = model(model.mel_forward(audio.to(device)))  # (B, 250, D)
        feats.append(emb.half().cpu())
        masks.append(mask[:, :1000].float())
    return torch.cat(feats), torch.cat(masks)


def fit_probe(x, y, device, epochs=300):
    """x: (N, 250, D) embeddings, y: (N, 1000) 10 ms masks -> linear probe at 40 ms."""
    y40 = F.avg_pool1d(y.unsqueeze(1), 4).squeeze(1)  # (N, 250) soft targets
    x = x.float().reshape(-1, x.shape[-1]).to(device)
    y40 = y40.reshape(-1).to(device)
    mu, sd = x.mean(0), x.std(0) + 1e-5
    probe = torch.nn.Linear(x.shape[1], 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(probe((x - mu) / sd).squeeze(1), y40)
        loss.backward()
        opt.step()
    return probe, mu, sd, float(loss)


def score(args):
    subset, ps, pe = args
    probs, targets = G[subset]
    pred, gt = [], []
    for p, t in zip(probs, targets):
        st = viterbi_decode(p, p_stay_silence=ps, p_stay_event=pe)
        pred.append([(e["onset"], e["offset"]) for e in extract_events_viterbi(st, 0.01)])
        gt.append([(e["onset"], e["offset"]) for e in extract_events_from_mask(t.astype(np.int32), 0.01)])
    f1, _, _ = micro_event_f1(gt, pred)
    return subset, ps, pe, f1, macro_segment_dice(gt, pred, hop_sec=0.01)


def report(label):
    with mp.get_context("fork").Pool(8) as pool:
        res = pool.map(score, [(s, ps, pe) for s in ("natural", "synthetic") for ps, pe in DECODES])
    r = {(s, ps, pe): (f1, d) for s, ps, pe, f1, d in res}
    for ps, pe in DECODES:
        nf, nd = r[("natural", ps, pe)]
        sf, sd = r[("synthetic", ps, pe)]
        print(f"  {label:28s} {ps}/{pe}: natF1={nf:.4f} natDice={nd:.4f} syn={sf + sd:.4f} "
              f"eq={0.5 * (nf + nd + sf + sd):.4f}", flush=True)


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or ["ATST-F", "BEATs", "ASIT"]
    config = Config()
    device = config.DEVICE
    _, gold_train_loader, _ = make_gold_loaders(config)
    gold = Subset(gold_train_loader.dataset, range(N_GOLD))
    gold_loader = DataLoader(gold, batch_size=16, shuffle=False, num_workers=config.NUM_WORKERS)
    val_loaders = {s: DataLoader(OfficialValidationDataset(VAL_ROOT, config, subset=s), batch_size=16,
                                 shuffle=False, num_workers=config.NUM_WORKERS) for s in ("natural", "synthetic")}

    print("Reference: our best single model (dasheng_owA_t, fine-tuned, cached logits)", flush=True)
    C = dict(np.load(f"{SCRATCH}/ens_v2_val_logits.npz"))
    C.update(dict(np.load(f"{SCRATCH}/dasheng_ens_val_logits.npz")))
    for s in ("natural", "synthetic"):
        n = min(C[f"{s}__dasheng_owA_t"].shape[1], C[f"{s}__targets"].shape[1])
        G[s] = (1 / (1 + np.exp(-C[f"{s}__dasheng_owA_t"][:, :n])), C[f"{s}__targets"][:, :n])
    if "--no-ref" not in sys.argv:
        report("dasheng_owA_t (fine-tuned)")

    for spec in names:
        name, _, stage = spec.partition(":")
        stage = stage or "strong"
        model = build_strong(name, head_type=None, stage=stage).to(device).eval()
        gx, gy = embed(model, gold_loader, device)
        probe, mu, sd, loss = fit_probe(gx, gy, device)
        print(f"\n{name} [{stage}]: linear probe on {N_GOLD} Gold-train clips, final BCE {loss:.4f}", flush=True)
        for s, loader in val_loaders.items():
            vx, vy = embed(model, loader, device)
            with torch.no_grad():
                logit = probe((vx.float().to(device) - mu) / sd).squeeze(-1)  # (N, 250)
                p = torch.sigmoid(F.interpolate(logit.unsqueeze(1), size=1000, mode="linear",
                                                align_corners=True).squeeze(1)).cpu().numpy()
            G[s] = (p, vy.numpy())
        report(f"{name}-{stage} frozen+probe")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
