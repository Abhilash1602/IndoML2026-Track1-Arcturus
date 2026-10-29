"""Oct 14, idea #2 step 2: a weak-label tagger (organisers'-baseline style attention MIL) on frozen pretrained
Dasheng tokens (work/tags/*_tok.npy), trained ONLY from clip-level NoiseCategory tags of all HF clips
except the old Gold held-out split (so its outputs are out-of-sample on Gold held-out; the official
validation set is not in HF). Outputs per-frame (25 Hz) and per-clip probabilities for 8 channels
(any-noise + 7 categories) on Gold held-out, validation and test -> evidence for the verifier.

Outputs: work/tags/{gold_heldout,official_natural,official_synthetic,test}_tagframe.npy (N, 250, 8) f16,
         ..._tagclip.npy (N, 8); work/tags/gold_heldout_rows.npy (HF row index of each Gold held-out clip).
Usage: CUDA_VISIBLE_DEVICES=0 python -m temp.train_tagger
"""
import os

import numpy as np
import torch
import torch.nn as nn
from datasets import Audio

from src.dataset import load_full_dataset

D = "./work/tags"
DEV = "cuda:0"


class Tagger(nn.Module):
    def __init__(self, d_in=768, d=256, n=8):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.rnn = nn.GRU(d, d, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.frame, self.att = nn.Linear(2 * d, n), nn.Linear(2 * d, n)

    def forward(self, x):
        h, _ = self.rnn(self.inp(x))
        f = torch.sigmoid(self.frame(h))
        a = torch.softmax(self.att(h), dim=1)
        return f, (f * a).sum(1).clamp(1e-6, 1 - 1e-6)


def heldout_rows():
    full = load_full_dataset().cast_column("audio", Audio(decode=False))
    q = full["annotationQuality"]
    gold_rows = [i for i, x in enumerate(q) if x == "verified_timestamps"]
    gold = full.filter(lambda r: r["annotationQuality"] == "verified_timestamps")
    gi = gold.add_column("_gi", list(range(len(gold)))).train_test_split(test_size=0.2, seed=42)["test"]["_gi"]
    return np.array([gold_rows[g] for g in gi])


@torch.no_grad()
def predict(model, tok, out_prefix):
    F, C = [], []
    for b in range(0, len(tok), 256):
        x = torch.from_numpy(np.asarray(tok[b:b + 256], np.float32)).to(DEV)
        f, c = model(x)
        F.append(f.half().cpu().numpy()); C.append(c.cpu().numpy())
    np.save(f"{out_prefix}_tagframe.npy", np.concatenate(F)); np.save(f"{out_prefix}_tagclip.npy", np.concatenate(C))


def main():
    torch.manual_seed(0)
    tok = np.load(f"{D}/hf_tok.npy", mmap_mode="r")
    tags = np.load(f"{D}/hf_tags.npy")
    held = heldout_rows()
    np.save(f"{D}/gold_heldout_rows.npy", held)
    train = np.setdiff1d(np.arange(len(tags)), held)
    print(f"tagger: {len(train)} training clips (Gold held-out {len(held)} excluded); tag rates {tags[train].mean(0).round(3)}", flush=True)
    model = Tagger().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    pos = torch.tensor(np.clip((1 - tags[train].mean(0)) / np.maximum(tags[train].mean(0), 1e-3), 1, 20), device=DEV)
    rng = np.random.default_rng(0)
    for ep in range(4):
        model.train()
        order = rng.permutation(train)
        losses = []
        for b in range(0, len(order) - 127, 128):
            idx = np.sort(order[b:b + 128])
            x = torch.from_numpy(np.asarray(tok[idx], np.float32)).to(DEV)
            y = torch.from_numpy(tags[idx]).to(DEV)
            _, c = model(x)
            loss = -(pos * y * torch.log(c) + (1 - y) * torch.log(1 - c)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            from sklearn.metrics import roc_auc_score
            c = torch.cat([model(torch.from_numpy(np.asarray(tok[held[b:b + 256]], np.float32)).to(DEV))[1].cpu()
                           for b in range(0, len(held), 256)]).numpy()
            aucs = [roc_auc_score(tags[held, j], c[:, j]) for j in range(8) if 0 < tags[held, j].mean() < 1]
        print(f"epoch {ep + 1}: loss {np.mean(losses):.4f} | Gold held-out clip-tag AUC mean {np.mean(aucs):.3f}", flush=True)
    torch.save(model.state_dict(), "./checkpoints/tagger_dasheng_mil.pth")
    predict(model, tok_rows(tok, held), f"{D}/gold_heldout")
    for s in ("official_natural", "official_synthetic", "test"):
        predict(model, np.load(f"{D}/{s}_tok.npy", mmap_mode="r"), f"{D}/{s}")
    print("saved tagger outputs", flush=True)


def tok_rows(tok, rows):
    """Gather rows in the given (Gold held-out cache) order."""
    return np.stack([np.asarray(tok[r]) for r in rows])


if __name__ == "__main__":
    main()
