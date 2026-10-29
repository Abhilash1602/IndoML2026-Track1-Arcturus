"""Generate the Phase F submission using its Optuna-retuned decode params
(p_stay_silence=0.9989, p_stay_event=0.8355, corrected-metric eq-avg 1.1769 --
see temp/tune_phasef_decode.py; only +0.0017 over default, decode params remain
largely robust once the model itself is strong).

Usage: python -m temp.gen_phasef_submission
"""
from torch.utils.data import DataLoader

from generate_submission import LocalTestDataset, generate_ensemble_submission
from src.config import Config

config = Config()
dataset = LocalTestDataset("./data/test_audio", config)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

generate_ensemble_submission(
    ["./checkpoints/dualbranch_phasef.pth"],
    loader,
    output_jsonl="predictions_phasef.jsonl",
    output_zip="submission_phasef.zip",
    p_stay_silence=0.9989,
    p_stay_event=0.8355,
)
print("Done -> submission_phasef.zip")
