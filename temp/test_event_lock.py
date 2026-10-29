"""Step 0 checks for src/event_lock.py: the lock never changes frames (Dice exactly equal), and
feeding the reference events as the "event model" recovers the oracle-cut ceiling on validation.

Usage: python -m temp.test_event_lock
"""
import numpy as np

from src.edge_correction import correct_events, edge_contrast, load_table
from src.event_lock import frames_equal, lock_partition
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S


def test_random_frames_equal():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = rng.integers(0, 5)
        starts = np.sort(rng.uniform(0, 9, n)).round(2)
        regions = [(float(s), float(round(min(s + rng.uniform(0.05, 2), 10), 2))) for s in starts]
        regions = [r for k, r in enumerate(regions) if k == 0 or r[0] >= regions[k - 1][1]]
        ev = [(float(x), float(x + rng.uniform(0.02, 1))) for x in rng.uniform(0, 10, rng.integers(0, 6))]
        out = lock_partition(regions, ev, c_min=0.0)
        assert frames_equal(out, regions), (regions, ev, out)
        assert all(b - a >= 0.03 - 1e-9 for a, b in out if (a, b) not in regions), out
    print("random inputs: frames always identical")


def test_oracle_on_validation():
    table = load_table("./checkpoints/edge_correction_v5.json")
    d = np.load(f"{S}/refiner_official_natural.npz")
    P = 1 / (1 + np.exp(-d["logits"].astype(np.float32).mean(1)))
    refs = refs_of(d["mask"].astype(np.float32))
    vit = viterbi_events(P)
    base = [[(round(a, 2), round(b, 2)) for a, b in correct_events(e, p, table)] for e, p in zip(vit, P)]
    locked = [lock_partition(b, r) for b, r in zip(base, refs)]
    assert all(frames_equal(x, y) for x, y in zip(base, locked))
    r0, r1 = per_clip(refs, base), per_clip(refs, locked)
    dd = r1[:, 3].mean() - r0[:, 3].mean()
    print(f"oracle lock (references as event model): Natural F1 {f1_only(r0):.4f} -> {f1_only(r1):.4f} "
          f"({f1_only(r1) - f1_only(r0):+.4f}); Dice change {dd:+.6f}")
    assert dd == 0.0 and f1_only(r1) - f1_only(r0) > 0.1


if __name__ == "__main__":
    test_random_frames_equal()
    test_oracle_on_validation()
    print("event-lock checks passed")
