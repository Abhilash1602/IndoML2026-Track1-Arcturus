from collections import Counter
from datasets import Audio
from src.dataset import load_full_dataset

ds = load_full_dataset().cast_column("audio", Audio(decode=False)).remove_columns(["audio"])
print("loaded, rows:", len(ds), flush=True)
q = ds["annotationQuality"]
nc = ds["NoiseCategory"]
bronze_tags = []
for i in range(len(q)):
    if q[i] == "no_timestamps":
        v = nc[i]
        if isinstance(v, list): bronze_tags.extend(v)
        elif v: bronze_tags.append(v)
print("Bronze NoiseCategory dist:", dict(Counter(bronze_tags).most_common()), flush=True)

ts = ds["NoiseSubCategoryTimeStamp"]
sc = Counter()
for i in range(len(q)):
    if q[i] == "unverified_timestamps":
        sc[min(len(ts[i] or []), 6)] += 1
print("Silver event-count dist (0..6+):", dict(sorted(sc.items())), flush=True)

# Also: Gold event tag full distribution (for the synthesis event bank)
gc = Counter()
for i in range(len(q)):
    if q[i] == "verified_timestamps":
        for ev in (ts[i] or []):
            gc[(ev.get("category") or "?", ev.get("tag") or "?")] += 1
print("Gold (category, tag) counts:", dict(gc.most_common(25)), flush=True)
