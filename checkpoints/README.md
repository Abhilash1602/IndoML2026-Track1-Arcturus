# Checkpoints

The final system loads the 20 files listed in [`../CHECKPOINTS.md`](../CHECKPOINTS.md) (4.3 GB in total, with sha256 checksums).
They are too large for GitHub and are hosted on Hugging Face:

**https://huggingface.co/asymptoticlearning/IndoML2026-Track1-Arcturus**

Install them from the repository root:
```bash
bash scripts/download_checkpoints.sh
```
This downloads the files into this folder (`checkpoints/*.pth`, `checkpoints/atst_frame/`, `checkpoints/beats/`, `checkpoints/postproc_v11.pkl`, `checkpoints/edge_correction_v5.json`) and verifies every checksum.

`mispeech/dasheng-base` is downloaded automatically from Hugging Face the first time the models are built.
