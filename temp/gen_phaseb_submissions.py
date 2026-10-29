"""Generate the Phase B submissions. Decode params taken from temp/eval_phaseb.py."""
import sys
from torch.utils.data import DataLoader
from src.config import Config
from generate_submission import LocalTestDataset, generate_ensemble_submission

# args: p_sil_alone p_evt_alone p_sil_ens p_evt_ens
psa, pea, pse, pee = (float(x) for x in sys.argv[1:5])
cfg = Config()
loader = DataLoader(LocalTestDataset("./data/test_audio", cfg), batch_size=cfg.BATCH_SIZE,
                    shuffle=False, num_workers=cfg.NUM_WORKERS)

print(">> PhaseB alone (%.4f/%.4f)" % (psa, pea), flush=True)
generate_ensemble_submission(["./checkpoints/dualbranch_baseline_synth.pth"], loader,
                             output_jsonl="predictions_phaseb.jsonl",
                             output_zip="submission_phaseb.zip",
                             p_stay_silence=psa, p_stay_event=pea)

print(">> PhaseB + old2 ensemble (%.4f/%.4f)" % (pse, pee), flush=True)
generate_ensemble_submission(
    ["./checkpoints/dualbranch_baseline_synth.pth",
     "./checkpoints/dualbranch_baseline.pth",
     "./checkpoints/dualbranch_distilled.pth"], loader,
    output_jsonl="predictions_phaseb_ensemble.jsonl",
    output_zip="submission_phaseb_ensemble.zip",
    p_stay_silence=pse, p_stay_event=pee)
print("done", flush=True)
