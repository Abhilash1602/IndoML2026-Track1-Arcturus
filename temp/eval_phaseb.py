"""Eval Phase B on gold_val + synth_val, sweep decode, for PhaseB-alone and
PhaseB + old baseline + old distilled."""
import numpy as np, torch
from datasets import Audio
from torch.utils.data import DataLoader
from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import VaaniDataset, SyntheticSEDDataset, load_full_dataset
from src.utils import (compute_segment_dice, compute_event_f1, extract_events_from_mask,
                       extract_events_viterbi, viterbi_decode)

PHASEB = "./checkpoints/dualbranch_baseline_synth.pth"
OLD_BASE = "./checkpoints/dualbranch_baseline.pth"
OLD_DIST = "./checkpoints/dualbranch_distilled.pth"

def load_model(path, cfg):
    m = DualBranchSED(cfg).to(cfg.DEVICE)
    m.load_state_dict(torch.load(path, map_location=cfg.DEVICE, weights_only=True)); m.eval()
    return m

def cache(models, loader, cfg):
    hop = cfg.HOP_LENGTH / cfg.SR
    P, M = [], []
    with torch.no_grad():
        for a, mk, _ in loader:
            a = a.to(cfg.DEVICE)
            p = torch.sigmoid(sum(m(a) for m in models)/len(models)).cpu().numpy()
            for pi, mi in zip(p, mk.numpy()):
                k = min(len(pi), len(mi)); P.append(pi[:k]); M.append(mi[:k])
    return P, M, hop

def sc(P, M, hop, ps, pe):
    d, f = [], []
    for pr, gt in zip(P, M):
        st = viterbi_decode(pr, p_stay_silence=ps, p_stay_event=pe)
        d.append(compute_segment_dice(st.astype(np.int32), gt.astype(np.int32)))
        f.append(compute_event_f1(extract_events_viterbi(st, hop),
                                  extract_events_from_mask(gt.astype(np.int32), hop)))
    return float(np.mean(d)), float(np.mean(f))

def main():
    cfg = Config()
    full = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold = full.filter(lambda r: r["annotationQuality"] == "verified_timestamps")
    sp = gold.train_test_split(test_size=0.2, seed=42)
    gv = DataLoader(VaaniDataset(sp["test"].cast_column("audio", Audio(decode=True)),
                    sp["test"].cast_column("audio", Audio(decode=False)), cfg),
                    batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    sv = DataLoader(SyntheticSEDDataset("data/synth_val", cfg, add_noise=False),
                    batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    pb = load_model(PHASEB, cfg)
    cfgs = {"PhaseB alone": [pb],
            "PhaseB+old2 ": [pb, load_model(OLD_BASE, cfg), load_model(OLD_DIST, cfg)]}
    grid = [(0.9986,0.6944),(0.99,0.85),(0.98,0.90),(0.97,0.90),(0.96,0.88),(0.95,0.85)]
    for name, ms in cfgs.items():
        gp, gm, hop = cache(ms, gv, cfg)
        spx, smx, _ = cache(ms, sv, cfg)
        print("\n=== %s ===" % name, flush=True)
        best = None
        for ps, pe in grid:
            gd, gf = sc(gp, gm, hop, ps, pe); sd, sf = sc(spx, smx, hop, ps, pe)
            bl = 0.6*(sd+sf) + 0.4*(gd+gf); tag = ""
            if best is None or bl > best[0]: best = (bl, ps, pe); tag = " <-- best"
            print("  (%.4f,%.4f) gold %.3f/%.3f/%.3f  synth %.3f/%.3f/%.3f%s" %
                  (ps, pe, gd, gf, gd+gf, sd, sf, sd+sf, tag), flush=True)
        print("  best: p_sil=%s p_evt=%s" % (best[1], best[2]), flush=True)

if __name__ == "__main__":
    main()
