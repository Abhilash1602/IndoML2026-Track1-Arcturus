"""Generate the Phase E submission using its own Optuna-retuned decode params
(p_stay_silence=0.9963, p_stay_event=0.5257, eq-avg 1.1365 -- see
temp/tune_phasee_decode.py; only +0.0007 over the un-tuned default, confirming
decode params are largely robust once the model itself is strong, same pattern
as Phase B/D).

Usage: python -m temp.gen_phasee_submission
"""
from torch.utils.data import DataLoader

from generate_submission import LocalTestDataset, generate_ensemble_submission
from src.config import Config

config = Config()
dataset = LocalTestDataset("./data/test_audio", config)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

generate_ensemble_submission(
    ["./checkpoints/dualbranch_phasee.pth"],
    loader,
    output_jsonl="predictions_phasee.jsonl",
    output_zip="submission_phasee.zip",
    p_stay_silence=0.9963,
    p_stay_event=0.5257,
)
print("Done -> submission_phasee.zip")
