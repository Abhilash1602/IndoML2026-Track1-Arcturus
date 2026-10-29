"""Generate submission_v4.zip from the final search v3 winner
(scratchpad/search_v3_final.log): atst_stage2 + dasheng_owB + dasheng_owB_t +
beats_s2 + beats_owA (BEATs S2 + OWBCE-A student), logit-averaged, at the
real-verified 0.9802/0.9489 decode. Validation 1.5327 vs submission_v3's set
1.5270 (+0.0057; paired bootstrap P(>0)=0.96, 5th pct +0.0005 -- a smaller,
lower-confidence step than v3 over i2). Decode-stable: 1.5324 / 1.5327 / 1.5338.
All checkpoints come from finished runs.

Usage: python -m temp.gen_submission_v4
"""
import temp.search_ensemble_v2 as v2
import temp.search_ensemble_v3 as v3
from src.architecture import StandaloneBEATsSED, StandaloneDashengSED

NAMES = ("atst_stage2", "dasheng_owB", "dasheng_owB_t", "beats_s2", "beats_owA")
MODE = "logit"
PS, PE = v3.DECODES["d1.5460"]
v3.OUT_JSONL, v3.OUT_ZIP = "predictions_v4.jsonl", "submission_v4.zip"

v2.MEMBERS["dasheng_owB"] = dict(cls=StandaloneDashengSED,
                                 path="./checkpoints/standalone_dasheng_stage2_i2_owbceB.pth", overrides={})
v2.MEMBERS["dasheng_owB_t"] = dict(cls=StandaloneDashengSED,
                                   path="./checkpoints/standalone_dasheng_stage2_i2_owbceB_teacher.pth", overrides={})
v2.MEMBERS["beats_s2"] = dict(cls=StandaloneBEATsSED, path="./checkpoints/standalone_beats_stage2.pth", overrides={})
v2.MEMBERS["beats_owA"] = dict(cls=StandaloneBEATsSED, path="./checkpoints/standalone_beats_stage2_owbceA.pth",
                               overrides={})

if __name__ == "__main__":
    assert all(n in v2.MEMBERS for n in NAMES)
    v3.generate_submission(NAMES, MODE, PS, PE)
