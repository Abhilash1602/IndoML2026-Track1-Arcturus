# Team Arcturus: IndoML 2026 Datathon, Track 1 (Noise Event Detection)

Training and inference code for our Track 1 system. For every clip it predicts the onset and offset of each noise event (breathing, horns, animals, alarms, music, machines, crying babies) in Vaani field recordings.

**Final standing: 8th on the Phase 2 leaderboard** (leaderboard name: **asymptoticchase**).

| Phase 2 test set | Combined | Event F1 | Segment Dice |
|---|---|---|---|
| All 5,517 clips | **1.5893** | 0.7691 | 0.8202 |
| 3,437 natural recordings | | 0.4542 | 0.7178 |
| 2,080 synthetic mixtures | | 0.9953 | ≈ 0.990 |

- **Technical report** (3 pages, IEEE format): [`docs/Arcturus_track_1_8.pdf`](docs/Arcturus_track_1_8.pdf)
- **Checkpoints:** https://huggingface.co/asymptoticlearning/IndoML2026-Track1-Arcturus

## Method in one paragraph
A ten-member frame ensemble produces candidate events: 2 × ATST-Frame with a 10 ms CNN branch, 4 × Dasheng-base and 4 × BEATs, each with a Conformer head on a 10 ms grid. It is trained with:
- synthetic mixtures built from Vaani audio;
- backbone unfreezing with layer-wise learning-rate decay plus Mean-Teacher and interpolation consistency on 80k unlabelled Vaani clips;
- an onset/offset-weighted BCE loss;
- fine-tuning on the official validation set;
- one round of self-training whose teacher is the full system.

An event-level stage then decides which candidates to keep and which to merge. It uses five boundary refiners as voters, two gradient-boosted verifiers, and a DETR-style set detector as extra evidence. A confidence guard (an event with mean probability ≥ 0.70 is never dropped) lets one pipeline process natural and synthetic clips the same way. Details are in the reports.

## Repository structure
```
.
├── README.md
├── requirements_train.txt          # training environment
├── requirements_inference.txt      # inference environment
├── CHECKPOINTS.md                  # the 20 checkpoint files with sizes and sha256
├── configs/model_config.yaml       # final inference configuration (thresholds, blend weight, windows)
├── inference/infer.py              # inference entry point (Phase 2 format)
├── train/
│   ├── run_training.sh             # the full training chain, step by step
│   └── config.yaml                 # hyperparameters of the shipped models
├── checkpoints/README.md           # how to fetch and verify the checkpoints
├── scripts/
│   ├── run_inference.sh            # wrapper around inference/infer.py
│   ├── download_checkpoints.sh     # download checkpoints from Hugging Face + verify sha256
│   └── chain_*.sh                  # the original training/evaluation chains we ran
├── src/                            # models, losses, data loading, synthesis, edge correction
├── temp/                           # training, caching and evaluation scripts; pipeline_v11.py = full inference recipe
├── release/                        # frame-ensemble inference helpers (member list, windowing, Viterbi)
├── train.py                        # data loaders and batch mixing
├── tools/                          # verify_reproduction.py, make_manifest.py, fit_edge_table.py
└── docs/                           # technical report (PDF)
```

## Environment and hardware
| Item | Value |
|---|---|
| Python | 3.12.2 (development, training and inference) |
| PyTorch / CUDA | torch 2.13.0+cu130, torchaudio 2.11.0, CUDA 13.0, cuDNN 9.20; NVIDIA driver 610.43.02 |
| Training GPUs | NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) and NVIDIA A100 80GB PCIe (80 GB) |
| Inference | one GPU, about 9 GB of GPU memory, about 65 min for the 5,517 test clips; also runs on CPU (slower) |
| CPU / RAM | 2 × AMD EPYC 7763 (256 threads), 1 TB RAM |
| OS | Ubuntu 24.04.4 LTS |
| Training time | 2–6 GPU-hours per ensemble member; refiners, detector and gradient-boosted models take minutes |
| System packages | `libsndfile1` (used by `soundfile`); `git`, `curl` |

**Pretrained model dependencies:**

| Model | Version | Source | Role |
|---|---|---|---|
| ATST-Frame | `ATST-F_strong_1.pt` (AudioSet-Strong) | https://github.com/fschmid56/PretrainedSED | backbone of 2 members |
| BEATs | `BEATs_iter3_plus_AS2M.pt` | https://github.com/microsoft/unilm/tree/master/beats | backbone of 4 members |
| Dasheng | `mispeech/dasheng-base` | https://huggingface.co/mispeech/dasheng-base | backbone of 4 members; detector input |

All three are public. The two `.pt` backbone files are included in the checkpoint archive, and Dasheng is downloaded from Hugging Face on first use. Licences: MIT (ATST-Frame, BEATs), Apache-2.0 (Dasheng). See [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Setup
```bash
git clone <this repository> && cd <repository>
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.13.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements_inference.txt        # or requirements_train.txt for training
bash scripts/download_checkpoints.sh              # from Hugging Face, verifies sha256 (see checkpoints/README.md)
```

## Inference
```bash
python inference/infer.py <input_audio_dir> <output_dir> [--device cuda:0]
# or: bash scripts/run_inference.sh <input_audio_dir> <output_dir> [cuda:0]
```
- **Input:** a folder of `.wav` files, searched recursively. Any sample rate and length is accepted; audio is resampled to 16 kHz mono, and clips longer than 10 s are processed in 10 s windows with 5 s overlap.
- **Output:** `<output_dir>/predictions.jsonl` (the Phase 2 submission format) and `<output_dir>/submission.zip`, which contains that file at the zip root. One line per clip:
  ```json
  {"clip_id": "<wav file name without .wav>", "events": [{"onset": 1.23, "offset": 1.71}, ...]}
  ```
  Times are in seconds. A clip with no detected event gets `"events": []`.
- **Configuration:** `configs/model_config.yaml` (the settings of the submitted run).
- **Reproduction check:** `python tools/verify_reproduction.py <output_dir>/predictions.jsonl <submitted zip>`. Rerunning from raw audio gave 5,484 of 5,517 clips identical to our submitted file, 28 within 10 ms, and 5 with a borderline event decided differently because of GPU floating-point noise.

## Training
Hyperparameters are in `train/config.yaml`. Run the chain one step at a time:
```bash
pip install -r requirements_train.txt
bash train/run_training.sh data        # fixed 70/30 validation split (measurement)
bash train/run_training.sh synth       # Vaani-derived synthetic mixtures (15,000 clips)
bash train/run_training.sh stage1      # head training on frozen backbones
bash train/run_training.sh stage2      # backbone unfreezing with LLRD + Mean Teacher + ICT, OWBCE
bash train/run_training.sh valft       # fine-tuning with the official validation set
bash train/run_training.sh refiners    # 5 boundary refiners (validation set)
bash train/run_training.sh sedt        # DETR-style set detector (Gold)
bash train/run_training.sh postproc    # false-positive filter and verifier (GBMs)
bash train/run_training.sh edgetable   # edge-correction table (Gold held-out)
bash train/run_training.sh pseudo      # self-training labels from the full system (79,526 clips)
bash train/run_training.sh students    # self-trained students (the owbceVS_* members)
```
- **Data:** the Vaani Noise-Event dataset is loaded from Hugging Face (`ARTPARK-IISc/Vaani-Noise-Event-Dataset`, all three tiers). Place the official Track 1 validation set in `data/official_validation_raw/validation/`.
- **Outputs:** each step writes to `./checkpoints/` with the file names listed in `CHECKPOINTS.md`.
- **Reproducibility:** GPU training is not bit-deterministic. A retrain therefore gives statistically similar members, not byte-identical checkpoints; the submitted results use the checkpoints in `CHECKPOINTS.md`.

## Compliance
- No test audio or test labels were used for training.
- No external datasets were used. The only outside resources are the public pretrained backbones above.
- The official validation set was used for fine-tuning, as the organisers confirmed was allowed.
- Every clip is processed by the same pipeline; there is no natural/synthetic routing at inference.

## Licence
MIT ([`LICENSE`](LICENSE)). Pretrained models and vendored code keep their own licences ([`THIRD_PARTY.md`](THIRD_PARTY.md)).

## Team and contact
**Team Arcturus** (Codabench leaderboard name: asymptoticchase)
- Abhilash Jacob Mathew (team leader), HCLTech: abhilashjacob16@gmail.com
- Aman Kumar, KIET GAZIABAD: aman.kumar.cse@kiet.edu
