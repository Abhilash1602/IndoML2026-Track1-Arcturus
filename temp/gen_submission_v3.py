"""Generate submission_v3.zip from the ensemble chosen after search v3 +
clip-bootstrap (scratchpad/v3_bootstrap.py): the submission_i2 members plus
dasheng_owA_t + dasheng_owB + beats_s2, logit-averaged, at submission_i2's own
real-verified decode 0.9802/0.9489 (so the only change vs the real 1.5460 is
the member set). Validation: eq 1.5270 vs reference 1.5179 (+0.0091).
dasheng_owA_t is a frozen copy (its run was still training).

Usage: python -m temp.gen_submission_v3
"""
import temp.search_ensemble_v3 as v3
import temp.search_ensemble_v2 as v2
from src.architecture import StandaloneBEATsSED, StandaloneDashengSED

NAMES = ("atst_stage2", "dasheng_s2_i2", "beats_smoke", "dasheng_owA_t", "dasheng_owB", "beats_s2")
MODE = "logit"
PS, PE = v3.DECODES["d1.5460"]

v2.MEMBERS["dasheng_owA_t"] = dict(cls=StandaloneDashengSED, path="./checkpoints/frozen_v3_dasheng_owA_t.pth", overrides={})
v2.MEMBERS["dasheng_owB"] = dict(cls=StandaloneDashengSED,
                                 path="./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth", overrides={})
v2.MEMBERS["beats_s2"] = dict(cls=StandaloneBEATsSED, path="./checkpoints/standalone_beats_stage2.pth", overrides={})

if __name__ == "__main__":
    assert all(n in v2.MEMBERS for n in NAMES)
    v3.generate_submission(NAMES, MODE, PS, PE)
