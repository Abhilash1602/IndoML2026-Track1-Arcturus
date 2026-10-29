"""Loader for the AudioSet-Strong pretrained transformers of PretrainedSED
(github.com/fschmid56/PretrainedSED, MIT; vendored at src/pretrained_sed).

The vendored code uses bare top-level imports (`from config import ...`,
`from models.beats...`), so its directory is put first on sys.path, and
`models` / `config` are imported from it before anything else of ours with
those names can shadow them. Checkpoints are downloaded once into
src/pretrained_sed/resources/.

build_strong(name, head_type) -> PredictionsWrapper
    name: "ATST-F" | "BEATs" | "fpasst" | "M2D" | "ASIT"
    head_type="linear": returns (strong_logits (B, 447, 250), weak_logits) as released.
    head_type=None: returns frame embeddings (B, 250, embed_dim) at 40 ms.
Input to forward is model.mel_forward(waveform (B, samples) at 16 kHz).
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_sed")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import models.prediction_wrapper as _pw  # noqa: E402  (vendored package)
from data_util.audioset_classes import as_strong_train_classes  # noqa: E402

_pw.RESOURCES_FOLDER = os.path.join(_ROOT, "resources")  # absolute, independent of the cwd

CHECKPOINTS = {"ATST-F": "ATST-F_strong_1", "BEATs": "BEATs_strong_1", "fpasst": "fpasst_strong_1",
               "M2D": "M2D_strong_1", "ASIT": "ASIT_strong_1"}
AUDIOSET_STRONG_CLASSES = list(as_strong_train_classes)  # 447 names, strong-head column order


def build_strong(name, head_type="linear", stage="strong"):
    """stage: "strong" (AudioSet-Strong fine-tuned), "weak" (AudioSet clip labels) or "ssl"
    (self-supervised only) -- same architecture, for controlled comparisons."""
    if name == "ATST-F":
        from models.atstframe.ATSTF_wrapper import ATSTWrapper
        base, kwargs = ATSTWrapper(), {}
    elif name == "BEATs":
        from models.beats.BEATs_wrapper import BEATsWrapper
        base, kwargs = BEATsWrapper(), {}
    elif name == "fpasst":
        from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
        base, kwargs = FPaSSTWrapper(), {}
    elif name == "M2D":
        from models.m2d.M2D_wrapper import M2DWrapper
        base = M2DWrapper()
        kwargs = {"embed_dim": base.m2d.cfg.feature_d}
    elif name == "ASIT":
        from models.asit.ASIT_wrapper import ASiTWrapper
        base, kwargs = ASiTWrapper(), {}
    else:
        raise ValueError(f"unknown strong backbone {name!r}")
    ckpt = CHECKPOINTS[name] if stage == "strong" else CHECKPOINTS[name].replace("_strong_1", f"_{stage}")
    return _pw.PredictionsWrapper(base, checkpoint=ckpt, head_type=head_type, **kwargs)
