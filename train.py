import collections
import copy
import gc
import os
import wandb
import numpy as np
import torch
from datasets import Audio
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import (
    BronzeInferenceDataset,
    SilverTrainingWrapper,
    SyntheticSEDDataset,
    VaaniDataset,
    load_full_dataset,
)
from src.loss import SEDLoss
from src.utils import (
    compute_event_f1,
    compute_segment_dice,
    extract_events_from_mask,
    extract_events_viterbi,
    viterbi_decode,
)

CHECKPOINT_DIR = "./checkpoints"
BASELINE_PATH = os.path.join(CHECKPOINT_DIR, "dualbranch_baseline.pth")
DISTILLED_PATH = os.path.join(CHECKPOINT_DIR, "dualbranch_distilled.pth")


def trainable_parameters(model):
    """Session 7d: DualBranchSED's frozen BEATs branch (90M params, requires_grad=False)
    would otherwise sit in every optimizer's param groups doing nothing but wasting
    memory on unused Adam momentum buffers -- filter it out at the source instead
    of special-casing each fit_* function's optimizer construction."""
    return (p for p in model.parameters() if p.requires_grad)


def apply_mixup(audio, masks, alpha=0.2, p=0.5, extra=None):
    """Mixup (Zhang et al. 2018), applied at the waveform level: blend two random
    clips' audio with a Beta-sampled weight. Labels combine with `max` (union), NOT
    a soft blend -- the org's official baseline does this with the explicit
    reasoning that a mixed clip genuinely contains both event sets, so a frame
    where an event definitely occurred shouldn't get a diluted target just because
    it came from the lower-weight side of the mix. (Session 7's soft-blend version
    is kept in git history; this replaced it after confirming SEDLoss's BCE/Dice/
    boundary terms need no other change -- none of them assume binary targets.)

    `extra`, if given (Session 9 plan, Part 1: the (B,T,7) category multi-hot
    target), is mixed with the identical union/max rule and returned as a third
    value -- optional and backward-compatible, every existing 2-return call
    site (`audio, masks = apply_mixup(...)`) is unaffected when omitted."""
    if audio.size(0) < 2 or torch.rand(1).item() > p:
        return (audio, masks) if extra is None else (audio, masks, extra)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(audio.size(0), device=audio.device)
    mixed_audio = lam * audio + (1 - lam) * audio[perm]
    mixed_masks = torch.maximum(masks, masks[perm])
    if extra is None:
        return mixed_audio, mixed_masks
    mixed_extra = torch.maximum(extra, extra[perm])
    return mixed_audio, mixed_masks, mixed_extra


def build_wandb_config_dict(config_cls):
    config_dict = {}
    for key, value in vars(config_cls).items():
        if key.startswith("__"):
            continue
        if isinstance(value, property):
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
            config_dict[key] = value
        else:
            config_dict[key] = str(value)
    return config_dict


def gold_fold_of(n, nf=5):
    """Oct 14 cross-fitting: fold of each Gold row (index in the verified-timestamps subset), fixed permutation."""
    import numpy as _np
    f = _np.empty(n, int)
    f[_np.random.default_rng(0).permutation(n)] = _np.arange(n) % nf
    return f


def val_fold_of(name, nf=5):
    """Oct 14 cross-fitting: fold of an official validation clip (md5 of its segmentFileName)."""
    import hashlib
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % nf


def make_gold_loaders(config):
    full_ds = load_full_dataset().cast_column("audio", Audio(decode=False))
    gold = full_ds.filter(lambda row: row["annotationQuality"] == "verified_timestamps")
    split = gold.train_test_split(test_size=0.2, seed=42)
    if os.environ.get("FOLD_K") is not None:  # cross-fitting: all Gold except fold FOLD_K (overrides GOLD_ALL)
        k, nf = int(os.environ["FOLD_K"]), int(os.environ.get("FOLD_N", "5"))
        f = gold_fold_of(len(gold), nf)
        split = {"train": gold.select([i for i in range(len(gold)) if f[i] != k]), "test": split["test"]}
    # GOLD_ALL=1: train on every Gold clip. The 20% hold-out predates the official
    # validation set, which now does all model selection; Gold is the only data in the
    # test set's (verified) annotation style, so holding a fifth of it out is a waste.
    gold_train = gold if (os.environ.get("GOLD_ALL") == "1" and os.environ.get("FOLD_K") is None) else split["train"]
    train = gold_train.cast_column("audio", Audio(decode=True))
    val = split["test"].cast_column("audio", Audio(decode=True))
    train_raw = gold_train.cast_column("audio", Audio(decode=False))
    val_raw = split["test"].cast_column("audio", Audio(decode=False))
    train_loader = DataLoader(VaaniDataset(train, train_raw, config), batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(VaaniDataset(val, val_raw, config), batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)
    return full_ds, train_loader, val_loader


SYNTH_TRAIN_DIR = "./data/synth_train"
SYNTH_VAL_DIR = "./data/synth_val"


def make_gold_synth_loaders(config, synth_train_dir=SYNTH_TRAIN_DIR, synth_val_dir=SYNTH_VAL_DIR,
                            synth_frac=0.5):
    """Mix real Gold (breathing-heavy, the Natural domain) with the synthetic
    loud-event set (src/synthesis.py). `synth_frac` is the synthetic share of each
    batch in expectation -- Phase B used 0.5 and solved the Synthetic subset
    (F1 0.44->0.89) at the cost of a small Natural F1 dip (0.41->0.38). Natural is
    now the entire remaining game, so lower synth_frac to protect it. Returns two
    val loaders -- Gold-val (Natural proxy) and synth-val (Synthetic proxy)."""
    full_ds, gold_train_loader, gold_val_loader = make_gold_loaders(config)
    synth_train = SyntheticSEDDataset(synth_train_dir, config)
    synth_val = SyntheticSEDDataset(synth_val_dir, config, add_noise=False)

    combined = ConcatDataset([gold_train_loader.dataset, synth_train])
    n_gold, n_synth = len(gold_train_loader.dataset), len(synth_train)
    gold_w, synth_w = (1.0 - synth_frac) / n_gold, synth_frac / n_synth
    weights = [gold_w] * n_gold + [synth_w] * n_synth
    num_samples = int(n_gold / max(1.0 - synth_frac, 1e-6))
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    train_loader = DataLoader(combined, batch_size=config.BATCH_SIZE, sampler=sampler,
                              num_workers=config.NUM_WORKERS)
    synth_val_loader = DataLoader(synth_val, batch_size=config.BATCH_SIZE, shuffle=False,
                                  num_workers=config.NUM_WORKERS)
    return full_ds, train_loader, gold_val_loader, synth_val_loader


# ---- Plan Phase V0/V1: the official validation set as training data --------------------
VAL_ROOT = "./data/official_validation_raw/validation"
VAL_SPLIT = "./data/val_split_s0.json"


def official_val_subset(config, subset, part):
    """OfficialValidationDataset restricted to a split part: "train70", "hold30" or "all"."""
    import json
    from torch.utils.data import Subset
    from src.dataset import OfficialValidationDataset
    ds = OfficialValidationDataset(VAL_ROOT, config, subset=subset)
    if part == "all":
        return ds
    if part == "fold":  # cross-fitting: every validation clip except fold FOLD_K
        k, nf = int(os.environ["FOLD_K"]), int(os.environ.get("FOLD_N", "5"))
        return Subset(ds, [i for i, r in enumerate(ds.rows) if val_fold_of(r["segmentFileName"], nf) != k])
    names = set(json.load(open(VAL_SPLIT))[part][subset])
    idx = [i for i, r in enumerate(ds.rows) if r["segmentFileName"] in names]
    assert len(idx) == len(names), f"{part}/{subset}: {len(idx)} of {len(names)} split names found"
    return Subset(ds, idx)


def eval_val_dataset(config, subset):
    """Per-epoch scoring set: hold30 whenever train70 is being trained on (no leakage)."""
    return official_val_subset(config, subset, "hold30" if os.environ.get("VAL_TRAIN") == "train70" else "all")


def make_gold_synth_val_loaders(config, synth_frac=0.3, val_part="train70",
                                val_nat_share=float(os.environ.get("VAL_NAT_SHARE", "0.3")),
                                val_syn_share=float(os.environ.get("VAL_SYN_SHARE", "0.5"))):
    """Gold + our synthetic + official-validation clips. Natural share of each batch
    (1 - synth_frac) is split Gold : val-Natural = (1 - val_nat_share) : val_nat_share;
    the synthetic share is split ours : val-synthetic = (1 - val_syn_share) : val_syn_share
    (val-synthetic comes from the organisers' own synthesiser, which ours does not match).
    Same return shape as make_gold_synth_loaders."""
    import json
    full_ds, gold_train_loader, gold_val_loader = make_gold_loaders(config)
    gold = gold_train_loader.dataset
    synth = SyntheticSEDDataset(SYNTH_TRAIN_DIR, config)
    vnat = official_val_subset(config, "natural", val_part)
    vsyn = official_val_subset(config, "synthetic", val_part)
    if val_part == "train70":
        hold = set(sum(json.load(open(VAL_SPLIT))["hold30"].values(), []))
        for d in (vnat, vsyn):
            assert not any(d.dataset.rows[i]["segmentFileName"] in hold for i in d.indices), "hold30 leaked"
    parts = [gold, synth, vnat, vsyn]
    shares = [(1 - synth_frac) * (1 - val_nat_share), synth_frac * (1 - val_syn_share),
              (1 - synth_frac) * val_nat_share, synth_frac * val_syn_share]
    silver_share = float(os.environ.get("SILVER_SHARE", "0"))
    if silver_share > 0:  # Option A: Silver "interior" human timestamps as extra Natural supervision
        silver = make_silver_interior_dataset(config, full_ds)
        parts.append(silver)
        shares[0] -= (1 - synth_frac) * silver_share  # taken from Gold's share of the Natural portion
        shares.append((1 - synth_frac) * silver_share)
        assert shares[0] > 0, "SILVER_SHARE + VAL_NAT_SHARE must be < 1"
    pseudo_dir, pseudo_share = os.environ.get("PSEUDO_LABELS"), float(os.environ.get("PSEUDO_SHARE", "0"))
    if pseudo_dir and pseudo_share > 0:  # iteration-4 self-training: Silver+Bronze labelled by the full v11g system
        import glob as _glob
        import numpy as _np
        labels = {}
        # PSEUDO_SOFT_MIX = lam (Oct 12, soft-target distillation): target = lam * teacher soft probability
        # + (1 - lam) * teacher final hard events; 0 (default) = hard events only, as before.
        lam = float(os.environ.get("PSEUDO_SOFT_MIX", "0"))
        for f in sorted(_glob.glob(os.path.join(pseudo_dir, "shard*of*.npz"))):
            d = _np.load(f)
            if lam > 0:
                mix = (lam * d["soft"].astype(_np.float32) + (1 - lam) * d["hard"].astype(_np.float32)).astype(_np.float16)
                labels.update(zip((str(x) for x in d["ids"]), mix))
            else:
                labels.update(zip((str(x) for x in d["ids"]), d["hard"]))
        pool = BronzeInferenceDataset(full_ds, config)
        missing = sum(1 for i in pool.indices if f"row_{i}" not in labels)
        assert missing == 0, f"pseudo-labels missing for {missing} of {len(pool)} Silver+Bronze clips"
        parts.append(SilverTrainingWrapper(pool, labels, audio_shape=(config.target_samples,),
                                           mask_shape=(config.target_samples // config.HOP_LENGTH,),
                                           use_boundary_head=getattr(config, "USE_BOUNDARY_HEAD", False)))
        shares[0] -= (1 - synth_frac) * pseudo_share  # taken from Gold's share of the Natural portion
        shares.append((1 - synth_frac) * pseudo_share)
        assert shares[0] > 0, "PSEUDO_SHARE + VAL_NAT_SHARE (+ SILVER_SHARE) must be < 1"
        print(f"[pseudo] {len(labels)} Silver+Bronze pseudo-labels from {pseudo_dir}, share {pseudo_share}", flush=True)
    audioset_share = float(os.environ.get("AUDIOSET_SHARE", "0"))
    if audioset_share > 0:  # AudioSet-Strong natural events, harmonised to Vaani style (temp/audioset_strong.py)
        aset = AudioSetStrongDataset(os.environ.get("AUDIOSET_NPZ", "./work/audioset_strong/audioset_strong_harmonised.npz"))
        parts.append(aset)
        shares[0] -= (1 - synth_frac) * audioset_share  # taken from Gold's share of the Natural portion
        shares.append((1 - synth_frac) * audioset_share)
        assert shares[0] > 0, "AUDIOSET_SHARE + other Natural shares must be < 1"
        print(f"[audioset] {len(aset)} harmonised AudioSet-Strong clips, share {audioset_share}", flush=True)
    weights = [w for ds, s in zip(parts, shares) for w in [s / len(ds)] * len(ds)]
    num_samples = int(len(gold) / max(1 - synth_frac, 1e-6))  # same epoch length as the Gold fine-tunes
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    train_loader = DataLoader(ConcatDataset(parts), batch_size=config.BATCH_SIZE, sampler=sampler,
                              num_workers=config.NUM_WORKERS)
    print(f"[val-train {val_part}] gold {len(gold)} synth {len(synth)} val-nat {len(vnat)} val-syn {len(vsyn)}; "
          f"batch shares {[round(s, 3) for s in shares]}; {len(train_loader)} batches/epoch", flush=True)
    return full_ds, train_loader, gold_val_loader, None


class AudioSetStrongDataset(torch.utils.data.Dataset):
    """AudioSet-Strong clips selected and harmonised to Vaani's annotation style (temp/audioset_strong.py):
    (audio float32 (160000,), merged any-noise mask float32 (1000,), id), with the same 50% light
    noise augmentation as VaaniDataset."""

    def __init__(self, path):
        z = np.load(path)
        self.audio, self.mask, self.ids = z["audio"], z["mask"], [str(x) for x in z["ids"]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        audio = torch.from_numpy(self.audio[i].astype(np.float32) / 32767.0)
        if torch.rand(1).item() < 0.5:
            audio = torch.clamp(audio + torch.randn_like(audio) * torch.empty(1).uniform_(0.001, 0.015).item(), -1.0, 1.0)
        return audio, torch.from_numpy(self.mask[i].astype(np.float32)), self.ids[i]


def make_silver_interior_dataset(config, full_ds):
    """Silver (unverified_timestamps, freelancer) clips whose events are all interior: no
    event starts at 0 or ends at the clip end (freelancers mark whole-clip spans for 30-39%
    of Silver events vs 5-9% in Gold) and the clip is <= 10 s. Same VaaniDataset targets
    as Gold (merged any-noise mask from NoiseSubCategoryTimeStamp)."""
    import ast

    def interior(row):
        if row["annotationQuality"] != "unverified_timestamps":
            return False
        try:
            dur = float(row["duration"])
            v = row["NoiseSubCategoryTimeStamp"]
            v = ast.literal_eval(v) if isinstance(v, str) else (v or [])
            ev = [(float(e["start"]), float(e["end"])) for e in v]
        except Exception:
            return False
        return (dur <= config.MAX_DURATION and len(ev) > 0 and
                all(t > s and s > 0.01 and abs(t - dur) > 0.02 for s, t in ev))

    silver = full_ds.filter(interior)
    ds = VaaniDataset(silver.cast_column("audio", Audio(decode=True)), silver, config)
    print(f"[silver-interior] {len(ds)} clips", flush=True)
    return ds


def make_train_loaders(config, synth_frac=0.3):
    """VAL_TRAIN = "" (default: Gold + synthetic only), "train70" or "all"."""
    part = os.environ.get("VAL_TRAIN", "")
    if part:
        return make_gold_synth_val_loaders(config, synth_frac, part)
    return make_gold_synth_loaders(config, synth_frac=synth_frac)


def evaluate_model(model, val_loader, criterion, config):
    """Checkpoint selection now decodes with Viterbi (fixed p_stay_silence/p_stay_event)
    instead of median-filter + a static threshold sweep — matching the decoder actually
    used at submission time (generate_submission.py), so the checkpoint judged "best"
    here is judged by the same method that ships, not a cruder proxy."""
    model.eval()
    total_loss, batch_count = 0.0, 0
    predictions, targets = [], []
    with torch.no_grad():
        for audio, masks, _ in val_loader:
            audio = audio.to(config.DEVICE, non_blocking=True)
            masks = masks.to(config.DEVICE, non_blocking=True)
            logits = model(audio)
            frames = min(logits.shape[1], masks.shape[1])
            logits, masks = logits[:, :frames], masks[:, :frames]
            total_loss += criterion(logits, masks).item()
            batch_count += 1
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(masks.cpu().numpy())

    all_probs = np.concatenate(predictions) if predictions else np.empty((0, 0))
    all_targets = np.concatenate(targets) if targets else np.empty((0, 0))
    hop = config.HOP_LENGTH / config.SR
    dice_scores, f1_scores = [], []
    for probability, target in zip(all_probs, all_targets):
        states = viterbi_decode(probability, p_stay_silence=0.95, p_stay_event=0.7)
        gt_mask = target.astype(np.int32)
        pred_events = extract_events_viterbi(states, hop)
        gt_events = extract_events_from_mask(gt_mask, hop)
        dice_scores.append(compute_segment_dice(states.astype(np.int32), gt_mask))
        f1_scores.append(compute_event_f1(pred_events, gt_events))
    dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return total_loss / max(batch_count, 1), dice, f1, dice + f1


def fit_baseline(model, train_loader, val_loader, config):
    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LR,
        steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS,
        pct_start=0.2,
    )
    best_score = -float("inf")
    epochs_since_improvement = 0
    config_dict = build_wandb_config_dict(Config)
    print("Commencing Phase 0 Baseline Training...")
    wandb.init(
        project="vaani-track1",
        name="phase0-baseline",
        config=config_dict,
        job_type="baseline_training",
    )
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(train_loader, desc=f"Baseline {epoch + 1}/{config.EPOCHS}"):
                audio, masks = audio.to(config.DEVICE), masks.to(config.DEVICE)
                audio, masks = apply_mixup(audio, masks)
                optimizer.zero_grad()
                logits = model(audio)
                frames = min(logits.shape[1], masks.shape[1])
                loss = criterion(logits[:, :frames], masks[:, :frames])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            validation_loss, val_dice, val_f1, score = evaluate_model(
                model, val_loader, criterion, config
            )
            train_loss = float(np.mean(losses)) if losses else 0.0
            print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, val_loss={validation_loss:.4f}, "
                  f"dice={val_dice:.4f}, f1={val_f1:.4f}, score={score:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": validation_loss,
                "val/dice": val_dice,
                "val/f1": val_f1,
                "val/combined_score": score,
                "train/learning_rate": scheduler.get_last_lr()[0],
            })
            if score > best_score:
                best_score = score
                epochs_since_improvement = 0
                torch.save(model.state_dict(), BASELINE_PATH)
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping: no improvement in {config.EARLY_STOP_PATIENCE} epochs.")
                    break
    finally:
        wandb.finish()


def generate_pseudo_labels(model, silver_loader, config):
    labels = {}
    model.eval()
    with torch.no_grad():
        for audio, clip_ids, _ in tqdm(silver_loader, desc="Pseudo-labeling"):
            probabilities = torch.sigmoid(model(audio.to(config.DEVICE))).cpu().numpy()
            for probability, clip_id in zip(probabilities, clip_ids):
                labels[clip_id] = viterbi_decode(
                    probability, p_stay_silence=0.95, p_stay_event=0.95
                ).astype(np.float32)
    return labels


def build_self_distillation_loader(full_ds, silver_bronze, pseudo_labels, gold_train_dataset, config):
    """Reconstructs Run A's proven recipe (Session 5 table: ~50% Gold / ~39% self-
    distilled Silver / ~11% self-distilled Bronze via WeightedRandomSampler) --
    this code path existed before Mean Teacher replaced it and is otherwise
    untouched (generate_pseudo_labels, SilverTrainingWrapper). Per-sample weights
    are derived from each tier's real size so the *expected* sampling proportions
    match Run A's ratio regardless of how large Silver+Bronze actually is."""
    silver_bronze_labeled = SilverTrainingWrapper(
        silver_bronze, pseudo_labels,
        audio_shape=(config.target_samples,),
        mask_shape=(config.target_samples // config.HOP_LENGTH,),
        use_boundary_head=getattr(config, "USE_BOUNDARY_HEAD", False),
    )
    combined = ConcatDataset([gold_train_dataset, silver_bronze_labeled])

    qualities = full_ds["annotationQuality"]
    tiers = [qualities[i] for i in silver_bronze.indices]
    n_gold = len(gold_train_dataset)
    n_silver = sum(1 for tier in tiers if tier == "unverified_timestamps")
    n_bronze = sum(1 for tier in tiers if tier == "no_timestamps")

    gold_weight = 0.50 / max(n_gold, 1)
    silver_weight = 0.39 / max(n_silver, 1)
    bronze_weight = 0.11 / max(n_bronze, 1)
    weights = [gold_weight] * n_gold
    weights.extend(silver_weight if tier == "unverified_timestamps" else bronze_weight for tier in tiers)

    # Original per-epoch sample count wasn't preserved in code; 3x Gold's size is a
    # reasonable middle ground -- long enough to actually draw from all three tiers
    # each epoch, short enough to keep epochs a similar order of magnitude to
    # fit_baseline's, unlike the old steps_per_epoch bug this project already paid
    # for once with Mean Teacher.
    num_samples = 3 * n_gold
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    return DataLoader(combined, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS)


def category_inverse_frequency_boost(category_labels, silver_bronze):
    """3.5-week re-plan, Must-do #1 Tier 0: compute a per-category inverse-
    frequency boost multiplier from temp/build_transcript_tag_labels.py's
    output (`category_labels`, a `{"row_<idx>": [category, ...]}` dict),
    restricted to whichever rows `silver_bronze` (a BronzeInferenceDataset)
    actually draws from -- so the boost reflects the true composition of
    this run's Silver+Bronze pool, not the full dataset. Standard inverse-
    frequency: rarest category gets the largest multiplier, most common
    (human_non_speech) anchors at 1.0. Clips with no recognized tag (~0.4%,
    see Tier 0's coverage report) get no boost."""
    counts = collections.Counter()
    for row_index in silver_bronze.indices:
        for category in category_labels.get(f"row_{row_index}", []):
            counts[category] += 1
    if not counts:
        return {}
    max_count = max(counts.values())
    return {category: max_count / n for category, n in counts.items()}


def make_full_mix_loaders(config, teacher, full_ds, gold_train_dataset,
                          synth_train_dir=SYNTH_TRAIN_DIR, synth_val_dir=SYNTH_VAL_DIR,
                          gold_frac=0.35, silver_bronze_frac=0.40, synth_frac=0.25,
                          category_labels=None, pseudo_labels=None):
    """Phase E: Gold (real labels) + self-distilled Silver+Bronze (`teacher`
    pseudo-labels the 79,526-clip pool via Viterbi -- Session 5 found trusting
    Silver's own approximate timestamps hurts boundary-tight F1 more than the
    model's own more-precise pseudo-labels do) + the synthetic loud-event set
    (keeps Synthetic solved), three-way WeightedRandomSampler mix. `teacher`
    should be the strongest available real-tested model (Phase B) -- Silver+Bronze
    is all Natural-domain real field recordings, so this is meant to give the
    model far more Natural-domain training volume than Gold's 11,111 clips alone
    can, without re-introducing Silver's raw boundary noise.

    3.5-week re-plan, Must-do #1: `category_labels`, if given (load
    data/transcript_tag_labels.json), up-weights individual Silver/Bronze
    samples within their tier's total mass by inverse category frequency
    (category_inverse_frequency_boost above) -- a direct, additive fix for
    Gold's diagnosed breathing/human_non_speech overrepresentation, since
    Silver/Bronze's transcript-derived composition is more diverse (see
    Tier 0's coverage report). Multi-category clips get the MAX boost among
    their categories (any rare category present is enough to up-weight).
    Renormalized so the tier's total sampling mass still sums to exactly
    `silver_bronze_frac`, matching gold_frac/synth_frac's own exact-sum
    convention -- only the *distribution within* the tier changes.
    Default (None) is byte-identical to the pre-existing uniform-weight
    behavior, so every existing caller (temp/train_phaseh.py etc.) is
    unaffected."""
    silver_bronze = BronzeInferenceDataset(full_ds, config)
    silver_bronze_loader = DataLoader(silver_bronze, batch_size=config.BATCH_SIZE, shuffle=False,
                                      num_workers=config.NUM_WORKERS)
    # `pseudo_labels`, if given, is a precomputed {clip_id: frame mask} dict keyed
    # by BronzeInferenceDataset's `row_<idx>` ids (e.g. an ensemble's labels from
    # temp/gen_ensemble_pseudo_labels.py) and replaces labeling with `teacher`.
    if pseudo_labels is None:
        print(f"Pseudo-labeling {len(silver_bronze)} Silver+Bronze clips...", flush=True)
        pseudo_labels = generate_pseudo_labels(teacher, silver_bronze_loader, config)
    else:
        missing = sum(1 for i in silver_bronze.indices if f"row_{i}" not in pseudo_labels)
        assert missing == 0, f"precomputed pseudo_labels missing {missing} Silver+Bronze clips"
        print(f"Using {len(pseudo_labels)} precomputed pseudo-labels (teacher unused)", flush=True)
    silver_bronze_labeled = SilverTrainingWrapper(
        silver_bronze, pseudo_labels,
        audio_shape=(config.target_samples,),
        mask_shape=(config.target_samples // config.HOP_LENGTH,),
        use_boundary_head=getattr(config, "USE_BOUNDARY_HEAD", False),
    )
    synth_train = SyntheticSEDDataset(synth_train_dir, config)
    synth_val = SyntheticSEDDataset(synth_val_dir, config, add_noise=False)

    combined = ConcatDataset([gold_train_dataset, silver_bronze_labeled, synth_train])
    n_gold, n_sb, n_synth = len(gold_train_dataset), len(silver_bronze_labeled), len(synth_train)
    if category_labels is not None:
        boost = category_inverse_frequency_boost(category_labels, silver_bronze)
        sb_raw_weights = []
        for row_index in silver_bronze.indices:
            cats = category_labels.get(f"row_{row_index}", [])
            sb_raw_weights.append(max((boost.get(c, 1.0) for c in cats), default=1.0))
        raw_total = sum(sb_raw_weights)
        sb_weights = [w * silver_bronze_frac / raw_total for w in sb_raw_weights]
        print(f"Category-stratified Silver+Bronze weighting: boost range "
              f"{min(boost.values()) if boost else 1.0:.2f}-{max(boost.values()) if boost else 1.0:.2f}x "
              f"across {len(boost)} categories", flush=True)
    else:
        sb_weights = [silver_bronze_frac / n_sb] * n_sb
    weights = ([gold_frac / n_gold] * n_gold
              + sb_weights
              + [synth_frac / n_synth] * n_synth)
    num_samples = int(n_gold / max(gold_frac, 1e-6))
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    train_loader = DataLoader(combined, batch_size=config.BATCH_SIZE, sampler=sampler,
                              num_workers=config.NUM_WORKERS)
    synth_val_loader = DataLoader(synth_val, batch_size=config.BATCH_SIZE, shuffle=False,
                                  num_workers=config.NUM_WORKERS)
    return train_loader, synth_val_loader


def fit_self_distilled(model, self_distill_loader, val_loader, config):
    """Run A's proven recipe (Session 5's best real Codabench result, 1.0899
    Combined / 0.4265 Event F1): supervised SEDLoss against a WeightedRandomSampler
    mix of Gold's real labels and self-distilled Silver+Bronze pseudo-labels
    (~50/39/11), on a fresh model -- matching the original convention of NOT
    warm-starting this phase, unlike this session's Mean Teacher (which needs a
    competent starting point for its consistency loss; this is plain supervised
    training against pseudo-labels, no such requirement). Adds this session's
    training hygiene the original run never had: gradient clipping and early
    stopping."""
    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=config.DISTILLED_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.DISTILLED_LR,
        steps_per_epoch=len(self_distill_loader),
        epochs=config.EPOCHS,
        pct_start=0.2,
    )
    best_score = -float("inf")
    epochs_since_improvement = 0
    config_dict = build_wandb_config_dict(Config)
    print(">> Starting Phase 2 Self-Distillation Training (Run A recipe)...")
    wandb.init(
        project="vaani-track1",
        name="phase2-self-distillation",
        config=config_dict,
        job_type="self_distillation_training",
    )
    try:
        for epoch in range(config.EPOCHS):
            model.train()
            losses = []
            for audio, masks, _ in tqdm(self_distill_loader, desc=f"SelfDistill {epoch + 1}/{config.EPOCHS}"):
                audio, masks = audio.to(config.DEVICE), masks.to(config.DEVICE)
                audio, masks = apply_mixup(audio, masks)
                optimizer.zero_grad()
                logits = model(audio)
                frames = min(logits.shape[1], masks.shape[1])
                loss = criterion(logits[:, :frames], masks[:, :frames])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            validation_loss, val_dice, val_f1, score = evaluate_model(
                model, val_loader, criterion, config
            )
            train_loss = float(np.mean(losses)) if losses else 0.0
            print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, val_loss={validation_loss:.4f}, "
                  f"dice={val_dice:.4f}, f1={val_f1:.4f}, score={score:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": validation_loss,
                "val/dice": val_dice,
                "val/f1": val_f1,
                "val/combined_score": score,
                "train/learning_rate": scheduler.get_last_lr()[0],
            })
            if score > best_score:
                best_score = score
                epochs_since_improvement = 0
                torch.save(model.state_dict(), DISTILLED_PATH)
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping: no improvement in {config.EARLY_STOP_PATIENCE} epochs.")
                    break
    finally:
        wandb.finish()


def _cycle_batch(loader, iterator):
    """Draw one batch from iterator, transparently restarting it when the
    (typically shorter) loader is exhausted, so two loaders of different
    lengths can be stepped through in lockstep."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def fit_distilled(student, gold_loader, silver_bronze_loader, val_loader, config):
    """Phase 2, now with a real Mean Teacher instead of one-shot static pseudo-labeling:
    Gold gets the usual supervised SEDLoss against its real labels; Silver+Bronze (no
    reliable temporal labels — Session 5 found Silver's real-but-unverified timestamps
    underperform self-distillation, so it stays grouped with Bronze here) gets an MSE
    consistency loss against a continuously EMA-updated teacher's soft predictions.
    (The auxiliary clip-level NoiseCategory tag loss from the previous run was removed:
    a full-pool check found NoiseCategory truthy on 100% of the 79,526 Silver+Bronze
    rows, so BCEWithLogitsLoss against it had no negative examples to discriminate
    against and was just converging to the trivial always-predict-positive solution
    -- consistent with tag_loss collapsing to ~0.0000 by epoch 12 of that run.)

    The consistency weight ramps up over the first MEAN_TEACHER_RAMPUP_EPOCHS epochs
    (Tarvainen & Valpola 2017's sigmoid schedule) instead of applying full strength
    from step 1: at epoch 1 both student and its own EMA-teacher are barely trained,
    so their "soft predictions" are closer to noise than signal, and forcing the
    student to match that noise at full weight fights the real supervised loss --
    exactly the failure mode observed when a fresh-init run's val score cratered at
    epoch 2 before slowly recovering. `student` is also expected to be warm-started
    from the trained Phase 0 baseline by the caller now, for the same reason: Mean
    Teacher assumes a reasonably-competent starting point, unlike one-shot static
    pseudo-labeling which specifically wanted a from-scratch signal-only test."""
    criterion = SEDLoss(config)
    criterion.to(config.DEVICE)

    teacher = copy.deepcopy(student)
    teacher.to(config.DEVICE)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(trainable_parameters(student), lr=config.DISTILLED_LR)
    # Fix A: drive the loop off Gold (the full, reliably-labeled pass) instead of
    # max(gold, silver_bronze) -- the old code replayed Gold ~9x per nominal "epoch"
    # since Silver+Bronze is ~9x larger, which is what caused the severe overfitting
    # (val score peaked at epoch 4, declined every epoch after) and ~9x the compute.
    steps_per_epoch = len(gold_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        pct_start=0.3,
    )
    best_score = -float("inf")
    epochs_since_improvement = 0
    config_dict = build_wandb_config_dict(Config)
    print(">> Starting Phase 2 Joint Distillation Training (Mean Teacher)...")
    wandb.init(
        project="vaani-track1",
        name="phase2-distillation",
        config=config_dict,
        job_type="distillation_training",
    )
    # Fix B: silver_iter lives outside the epoch loop so _cycle_batch's restart-on-
    # exhaustion is the only thing that reshuffles it -- happening naturally every
    # ~9 epochs (len(silver_bronze_loader) / steps_per_epoch), guaranteeing a full
    # pass over Silver+Bronze before any repeats. Re-creating it fresh every epoch
    # (the old code) would only ever sample its first steps_per_epoch batches from a
    # freshly-reshuffled pool each time, statistically missing ~16% of the data
    # entirely over a 15-epoch run.
    silver_iter = iter(silver_bronze_loader)
    try:
        for epoch in range(config.EPOCHS):
            # Sigmoid ramp-up (Tarvainen & Valpola 2017): 0 at epoch 0, reaching
            # ~MEAN_TEACHER_CONSISTENCY_WEIGHT by MEAN_TEACHER_RAMPUP_EPOCHS.
            rampup_phase = min(1.0, epoch / max(1, config.MEAN_TEACHER_RAMPUP_EPOCHS))
            consistency_weight = config.MEAN_TEACHER_CONSISTENCY_WEIGHT * float(
                np.exp(-5.0 * (1.0 - rampup_phase) ** 2)
            )
            student.train()
            losses, consistency_losses = [], []
            gold_iter = iter(gold_loader)
            for _ in tqdm(range(steps_per_epoch), desc=f"Distillation {epoch + 1}/{config.EPOCHS}"):
                (gold_audio, gold_masks, _), gold_iter = _cycle_batch(gold_loader, gold_iter)
                (sb_audio, sb_clip_ids, _sb_tags), silver_iter = _cycle_batch(silver_bronze_loader, silver_iter)

                gold_audio, gold_masks = gold_audio.to(config.DEVICE), gold_masks.to(config.DEVICE)
                sb_audio = sb_audio.to(config.DEVICE)

                optimizer.zero_grad()

                gold_logits = student(gold_audio)
                frames = min(gold_logits.shape[1], gold_masks.shape[1])
                supervised_loss = criterion(gold_logits[:, :frames], gold_masks[:, :frames])

                sb_logits = student(sb_audio)
                sb_probs = torch.sigmoid(sb_logits)
                with torch.no_grad():
                    teacher_probs = torch.sigmoid(teacher(sb_audio))
                consistency_loss = torch.nn.functional.mse_loss(sb_probs, teacher_probs)

                loss = (
                    supervised_loss
                    + consistency_weight * consistency_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                        teacher_param.data.mul_(config.MEAN_TEACHER_EMA_DECAY).add_(
                            student_param.data, alpha=1 - config.MEAN_TEACHER_EMA_DECAY
                        )

                losses.append(supervised_loss.item())
                consistency_losses.append(consistency_loss.item())

            validation_loss, val_dice, val_f1, score = evaluate_model(
                student, val_loader, criterion, config
            )
            train_loss = float(np.mean(losses)) if losses else 0.0
            mean_consistency = float(np.mean(consistency_losses)) if consistency_losses else 0.0
            print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, consistency={mean_consistency:.4f}, "
                  f"consistency_weight={consistency_weight:.4f}, val_loss={validation_loss:.4f}, "
                  f"dice={val_dice:.4f}, f1={val_f1:.4f}, score={score:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/consistency_loss": mean_consistency,
                "train/consistency_weight": consistency_weight,
                "val/loss": validation_loss,
                "val/dice": val_dice,
                "val/f1": val_f1,
                "val/combined_score": score,
                "train/learning_rate": scheduler.get_last_lr()[0],
            })
            if score > best_score:
                best_score = score
                epochs_since_improvement = 0
                torch.save(student.state_dict(), DISTILLED_PATH)
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping: no improvement in {config.EARLY_STOP_PATIENCE} epochs.")
                    break
    finally:
        wandb.finish()


def main():
    """Phase 2 uses Run A's proven self-distillation recipe (fit_self_distilled),
    not Mean Teacher (fit_distilled, still defined above and available for
    Priority 2 of the win-the-hackathon plan): two consecutive real Codabench
    submissions showed Mean Teacher failing to beat a plain Gold-only baseline,
    while this recipe is this project's actual best real result (Combined 1.0899,
    Event F1 0.4265) -- reviving it on the now-reverted plain-Conv2d architecture
    plus Freq-MixStyle (baked into DualBranchSED.forward) is the higher-leverage
    move than continuing to tune Mean Teacher."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config = Config()
    full_ds, train_loader, val_loader = make_gold_loaders(config)
    baseline = DualBranchSED(config).to(config.DEVICE)
    fit_baseline(baseline, train_loader, val_loader, config)
    # Reload the actual best-scoring checkpoint (fit_baseline only keeps the
    # in-memory `baseline` at whatever epoch training happened to end on, which
    # early stopping may have run past) before using it to pseudo-label.
    baseline.load_state_dict(torch.load(BASELINE_PATH, map_location=config.DEVICE))

    silver_bronze = BronzeInferenceDataset(full_ds, config)
    silver_bronze_loader = DataLoader(
        silver_bronze, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )
    print(f"Silver+Bronze (self-distillation pool) samples: {len(silver_bronze)}")
    pseudo_labels = generate_pseudo_labels(baseline, silver_bronze_loader, config)

    self_distill_loader = build_self_distillation_loader(
        full_ds, silver_bronze, pseudo_labels, train_loader.dataset, config
    )
    # Fresh init, matching Run A's original convention -- unlike Mean Teacher,
    # this is plain supervised training against pseudo-labels, so it doesn't need
    # a warm-started, already-competent starting point the way a consistency loss
    # against one's own EMA-teacher does.
    distilled = DualBranchSED(config).to(config.DEVICE)
    fit_self_distilled(distilled, self_distill_loader, val_loader, config)
    del baseline, distilled
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
