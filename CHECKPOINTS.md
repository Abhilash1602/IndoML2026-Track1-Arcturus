# Checkpoints used by the final Track 1 system

Loaded by `python inference/infer.py <input_dir> <output_dir>` (Phase 2 test: 1.5893, 8th place). Place them under `./checkpoints/` with these relative paths. `mispeech/dasheng-base` is downloaded from Hugging Face at first use.

| role | file | size (MB) | sha256 |
|---|---|---|---|
| member va_atst | `checkpoints/dualbranch_atst_frame_stage2_i2_owbceVS_atst_teacher.pth` | 352.4 | `1f374bba02555d4814be2f4529a1f6b20f6a73bf795527cfd640cd5d54836656` |
| member va_owB | `checkpoints/standalone_dasheng_stage2_i2_owbceVS_owB_teacher.pth` | 350.2 | `c26207ac3da6a29a836262ef3a436ec315f0618cedc32fe269938fb431053766` |
| member va_bS2 | `checkpoints/standalone_beats_stage2_owbceVS_bS2_teacher.pth` | 369.7 | `d79f4dbbcc53d63c4fe8fdaf060e9e99013ea243fc45724cd88f18e5c7c1934e` |
| member va_owBt | `checkpoints/standalone_dasheng_stage2_i2_owbceVS_owBt_teacher.pth` | 350.2 | `e541a327b86319490549caaf35b4f240d6a9a52edb278024f20c6e361a985dc8` |
| member va_bA | `checkpoints/standalone_beats_stage2_owbceVS_bA_teacher.pth` | 369.7 | `e2edfe4dd096bb4564657ff7bd8c687bdd6dac8cb1337b05be8c3451dec78012` |
| member atst_stage2 | `checkpoints/dualbranch_atst_frame_stage2.pth` | 352.3 | `245ea03e98d9a8c8bdfd70663cadca99717265c5171ebb90e04ceb43a0d55fbc` |
| member dasheng_owB | `checkpoints/standalone_dasheng_stage2_i2_owbceB.pth` | 350.2 | `bba863c849231c8ffe0e4c2da58a59c1566c98d250314ca22cfee654386633bb` |
| member dasheng_owB_t | `checkpoints/standalone_dasheng_stage2_i2_owbceB_teacher.pth` | 350.2 | `34edcc06bb0cbf833d11420cb75e6890babca9cb64cc4773298f6cab114f950a` |
| member beats_s2 | `checkpoints/standalone_beats_stage2.pth` | 369.7 | `4d252a5abdc0c4a7fd0c72a4ccd8823ff8f43c432dfda18dbd6dcb1b428d4b67` |
| member beats_owA | `checkpoints/standalone_beats_stage2_owbceA.pth` | 369.7 | `2c67b0f7b9cfccc26d0f4fbaa38795a899c3ba4adb003fc65ada4dd935d74a5b` |
| boundary refiner (voter 0) | `checkpoints/refiner_val_all_x0.pth` | 3.2 | `599b333ef8241ca5bf0e201aa67e7b0745196fea01864b6e0abb94a513b317c1` |
| boundary refiner (voter 1) | `checkpoints/refiner_val_train70_x0.pth` | 3.2 | `45d5fe77c924c00f7a873c31e64ee454aa12c22c2be9def15744f3c0d5be3c99` |
| boundary refiner (voter 2) | `checkpoints/refiner_val_all_x0_s1.pth` | 3.2 | `b8ba2266d90d4d27aa900d7ecfdab13b107c1fd6c6d44059b321fd2f3ed01282` |
| boundary refiner (voter 3) | `checkpoints/refiner_val_all_x0_s2.pth` | 3.2 | `73595bbbdaf9e479d7fb1be38d4733f06766bf5f643aa7066f7cee15249258d8` |
| boundary refiner (voter 4) | `checkpoints/refiner_val_all_x0_s3.pth` | 3.2 | `ed8a7281b81e783d49df8701568d7948b2c5119ab24672d6de3b2a630a007cef` |
| FP filter + verifier (GBMs, PCAs) | `checkpoints/postproc_v11.pkl` | 4.2 | `13d63d852c66f5043d050af7f57a78cd8d0038f65b5614bf347036327b9b5660` |
| DETR-style detector (verifier evidence) | `checkpoints/sedt_gold_anc_s0_alone.pth` | 24.5 | `d0b9f24bf37011696e9a4e3aad14fdae54b0712fc5f332f2b626c99dd6605bce` |
| gated edge-correction table | `checkpoints/edge_correction_v5.json` | 0.0 | `d6f397f15cd3f625343433feac7169a4b9540e8b96b46d21a12f3a9d7e196b7f` |
| backbone init (ATST-F) | `checkpoints/atst_frame/ATST-F_strong_1.pt` | 344.5 | `fbf2577958e3648d55ee8cea7e0e5260c4505fb0c13964e1ec81bc367cee5eda` |
| backbone init (BEATs) | `checkpoints/beats/BEATs_iter3_plus_AS2M.pt` | 361.5 | `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34` |
