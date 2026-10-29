"""Generate the Phase D (11M model) submission. generate_ensemble_submission builds
its own default-size Config() internally, which won't match this checkpoint's
bigger architecture -- so this duplicates its inference loop with the right config.
"""
import json, zipfile
import torch
from torch.utils.data import DataLoader
from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import pad_or_trim
from src.utils import extract_events_viterbi, viterbi_decode
from generate_submission import LocalTestDataset

config = Config()
config.CNN_CH = (48, 64, 128, 256)
config.CONF_DIM = 256
config.CONF_HEADS = 8
config.CONF_LAYERS = 6

model = DualBranchSED(config).to(config.DEVICE)
model.load_state_dict(torch.load("./checkpoints/dualbranch_big.pth", map_location=config.DEVICE, weights_only=True))
model.eval()

loader = DataLoader(LocalTestDataset("./data/test_audio", config), batch_size=config.BATCH_SIZE,
                    shuffle=False, num_workers=config.NUM_WORKERS)
hop = config.HOP_LENGTH / config.SR
entries = []
with torch.no_grad():
    from tqdm import tqdm
    for audio, clip_ids in tqdm(loader, desc="Inference"):
        audio = audio.to(config.DEVICE)
        probs = torch.sigmoid(model(audio)).cpu().numpy()
        for p, cid in zip(probs, clip_ids):
            states = viterbi_decode(p, p_stay_silence=0.9986, p_stay_event=0.6944)
            events = extract_events_viterbi(states, hop)
            entries.append({"clip_id": cid, "events": events})

with open("predictions_phased.jsonl", "w", encoding="utf-8") as fh:
    for e in entries:
        fh.write(json.dumps(e) + "\n")
with zipfile.ZipFile("submission_phased.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("predictions_phased.jsonl", arcname="predictions.jsonl")
print("done:", len(entries), "clips")
