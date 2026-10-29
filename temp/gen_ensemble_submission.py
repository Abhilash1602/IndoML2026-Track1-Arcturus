"""Generate the Phase H + WavLM full1 ensemble submission using the Optuna-retuned
decode params (p_stay_silence=0.9909, p_stay_event=0.8566, real-validation-set
eq-avg 1.4818 -- beats Phase H alone's real Codabench 1.4815, see STRATEGY.md).

Usage: python -m temp.gen_ensemble_submission
"""
import json
import zipfile

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.utils import extract_events_viterbi, viterbi_decode
from generate_submission import LocalTestDataset

P_STAY_SILENCE, P_STAY_EVENT = 0.9909, 0.8566


def build_phaseh():
    config = Config()
    config.CNN_CH = (48, 64, 128, 256)
    config.CONF_DIM = 256
    config.CONF_HEADS = 8
    config.CONF_LAYERS = 6
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load("./checkpoints/dualbranch_phaseh.pth", map_location=config.DEVICE, weights_only=True))
    model.eval()
    return model, config


def build_wavlm_full1():
    config = Config()
    config.USE_WAVLM = True
    model = DualBranchSED(config).to(config.DEVICE)
    model.load_state_dict(torch.load("./checkpoints/dualbranch_wavlm_stage1_full.pth", map_location=config.DEVICE, weights_only=True))
    model.eval()
    return model, config


phaseh_model, config = build_phaseh()
wavlm_model, _ = build_wavlm_full1()
models = [phaseh_model, wavlm_model]

dataset = LocalTestDataset("./data/test_audio", config)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)

entries, hop_sec = [], config.HOP_LENGTH / config.SR
with torch.no_grad():
    for audio, clip_ids in tqdm(loader, desc="Inference"):
        audio = audio.to(config.DEVICE)
        probabilities = torch.sigmoid(sum(m(audio) for m in models) / len(models)).cpu().numpy()
        for probability, clip_id in zip(probabilities, clip_ids):
            states = viterbi_decode(probability, p_stay_silence=P_STAY_SILENCE, p_stay_event=P_STAY_EVENT)
            events = extract_events_viterbi(states, hop_sec)
            entries.append({"clip_id": clip_id, "events": events})

with open("predictions_ensemble_h_wavlm.jsonl", "w", encoding="utf-8") as handle:
    for entry in entries:
        handle.write(json.dumps(entry) + "\n")
with zipfile.ZipFile("submission_ensemble_h_wavlm.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write("predictions_ensemble_h_wavlm.jsonl", arcname="predictions.jsonl")
print("Done -> submission_ensemble_h_wavlm.zip")
