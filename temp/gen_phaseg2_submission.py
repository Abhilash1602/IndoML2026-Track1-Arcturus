"""Generate the Phase G2 submission using its Optuna-retuned decode params
(p_stay_silence=0.9950, p_stay_event=0.8639, real-validation-set eq-avg 1.4602 --
beats Phase F's own real-validation-set eq-avg of 1.4502 by +0.0100, see
STRATEGY.md's Phase G2 section).

Usage: python -m temp.gen_phaseg2_submission
"""
from torch.utils.data import DataLoader

from generate_submission import LocalTestDataset, generate_ensemble_submission
from src.config import Config

config = Config()
dataset = LocalTestDataset("./data/test_audio", config)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

generate_ensemble_submission(
    ["./checkpoints/dualbranch_phaseg2.pth"],
    loader,
    output_jsonl="predictions_phaseg2.jsonl",
    output_zip="submission_phaseg2.zip",
    p_stay_silence=0.9950,
    p_stay_event=0.8639,
)
print("Done -> submission_phaseg2.zip")
