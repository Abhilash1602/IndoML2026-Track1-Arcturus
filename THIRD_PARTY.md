# Third-party components and pretrained dependencies

The code in this repository is MIT-licensed (see `LICENSE`). It builds on these public models and code, each under its own licence:

| Component | Used for | Source | Licence |
|---|---|---|---|
| ATST-Frame (`ATST-F_strong_1.pt`) and its model code (vendored in `src/atst_frame/`) | backbone of the 2 `DualBranchSED` members | [fschmid56/PretrainedSED](https://github.com/fschmid56/PretrainedSED) | MIT |
| BEATs iter3+ AS2M (`BEATs_iter3_plus_AS2M.pt`) | backbone of the 4 `StandaloneBEATsSED` members | [microsoft/unilm (BEATs)](https://github.com/microsoft/unilm/tree/master/beats); loaded through SpeechBrain | MIT |
| Dasheng-base (`mispeech/dasheng-base`) | backbone of the 4 `StandaloneDashengSED` members; frozen tokens for the DETR-style detector | [Hugging Face](https://huggingface.co/mispeech/dasheng-base) | Apache-2.0 |
| SpeechBrain | BEATs model code | [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain) | Apache-2.0 |
| OWBCE loss idea | onset/offset-weighted BCE | arXiv:2403.13254 (re-implemented in `src/loss.py`) | — |
| EPN/RED boundary refiner idea | in-domain boundary refiners | arXiv:2601.04178 (re-implemented in `src/refiner.py`) | — |
| Mean Teacher + ICT + layer-wise LR decay recipe | Stage-2 fine-tuning | CPJKU DCASE 2024 Task 4 system, arXiv:2407.12997 (re-implemented in `train.py` / `temp/train_*_stage2*.py`) | — |

**Data:** the Vaani Noise Event dataset ([ARTPARK-IISc/Vaani-Noise-Event-Dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset)) and the official Track 1 validation set. The organisers confirmed that the validation set may be used for fine-tuning. No other audio data is used by the final system, and no test audio is used for training.

**Experiment-only code that the final system does not use:**
- `temp/train_org_baseline.py` expects the organisers' baseline repository ([nagshubhadip/noise-event-detection-and-removal](https://github.com/nagshubhadip/noise-event-detection-and-removal)) under `third_party/org_baseline/`.
- `src/strong_backbones.py` expects the PretrainedSED repository under `src/pretrained_sed/`.
- `temp/cache_residual.py` needs ClearVoice (MossFormerGAN; path from the `MOSSFORMERGAN_CKPT` environment variable).

None of these are included or needed to reproduce the submission.
