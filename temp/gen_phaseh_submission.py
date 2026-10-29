"""Generate the Phase H submission using its Optuna-retuned decode params
(p_stay_silence=0.9955, p_stay_event=0.8222, real-validation-set eq-avg 1.4734 --
the first checkpoint since Phase F to beat it cleanly on both Natural and
Synthetic simultaneously, see STRATEGY.md's Phase H section).

Usage: python -m temp.gen_phaseh_submission
"""
import torch
from torch.utils.data import DataLoader

from src.architecture import DualBranchSED
from src.config import Config
from generate_submission import LocalTestDataset, generate_ensemble_submission

config = Config()
config.CNN_CH = (48, 64, 128, 256)
config.CONF_DIM = 256
config.CONF_HEADS = 8
config.CONF_LAYERS = 6

# generate_ensemble_submission builds Config() internally (default arch), which
# can't load Phase H's bigger checkpoint -- same issue as Phase D, same fix:
# build the model with the right overrides here and pass a pre-built model list
# by monkeypatching would be messier than just duplicating the small inference
# loop, matching temp/gen_phased_submission.py's precedent.
import json
import zipfile

from src.utils import extract_events_viterbi, viterbi_decode
from tqdm import tqdm

model = DualBranchSED(config).to(config.DEVICE)
model.load_state_dict(torch.load("./checkpoints/dualbranch_phaseh.pth", map_location=config.DEVICE, weights_only=True))
model.eval()

dataset = LocalTestDataset("./data/test_audio", config)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)

P_STAY_SILENCE, P_STAY_EVENT = 0.9955, 0.8222
entries, hop_sec = [], config.HOP_LENGTH / config.SR
with torch.no_grad():
    for audio, clip_ids in tqdm(loader, desc="Inference"):
        audio = audio.to(config.DEVICE)
        probabilities = torch.sigmoid(model(audio)).cpu().numpy()
        for probability, clip_id in zip(probabilities, clip_ids):
            states = viterbi_decode(probability, p_stay_silence=P_STAY_SILENCE, p_stay_event=P_STAY_EVENT)
            events = extract_events_viterbi(states, hop_sec)
            entries.append({"clip_id": clip_id, "events": events})

with open("predictions_phaseh.jsonl", "w", encoding="utf-8") as handle:
    for entry in entries:
        handle.write(json.dumps(entry) + "\n")
with zipfile.ZipFile("submission_phaseh.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write("predictions_phaseh.jsonl", arcname="predictions.jsonl")
print("Done -> submission_phaseh.zip")
