import json, os, numpy as np
from datasets import Audio
from src.dataset import load_full_dataset
from src.edge_correction import correct_events
from temp.crossfit_refiner import all_data
from temp.eval_crossfit_refiner import clip, durations
from temp.eval_recipe_refiners import T, presence, seeds, recipe
from temp.train_refiner import f1_only, per_clip, refs_of, viterbi_events
from temp.train_refiner_val import S
from temp.verifier_recipe import W, mapping
full = load_full_dataset().cast_column("audio", Audio(decode=False))
spk = lambda fn: os.path.basename(fn).split("_")[5]
tr = set(spk(a["path"]) for a in full["audio"])
ids = {sub: [str(x) for x in np.load(f"{S}/refiner_official_{sub}.npz")["ids"]] for sub in ("natural", "synthetic")}
mp = mapping()
seen = np.array([spk(ids[n.split("_")[1]][j] if ids[n.split("_")[1]][j].endswith(".wav") else ids[n.split("_")[1]][j] + ".wav") in tr for n, j in mp])
L, _, Y, _, nat = all_data(); N, sec = len(L), durations(); refs = refs_of(Y)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
P = 1 / (1 + np.exp(-(W * L.mean(1) + (1 - W) * lg(presence("crossfit", N)))))
base = [clip(correct_events(e, p, T), s) for e, p, s in zip(viterbi_events(P), P, sec)]
Rs = seeds("crossfit", N, P, sec)
rec = [recipe(x, Rs, i) for i, x in enumerate(base)]
r = per_clip(refs, rec)
print("example id:", ids["natural"][0])
for name, m in (("Natural, seen speaker", nat & seen), ("Natural, unseen speaker", nat & ~seen), ("Synthetic, seen", ~nat & seen), ("Synthetic, unseen", ~nat & ~seen)):
    rr = r[m]; tp, fp, fn = rr[:, 0].sum(), rr[:, 1].sum(), rr[:, 2].sum()
    print(f"{name:24s} clips {m.sum():5d} | F1 {2*tp/(2*tp+fp+fn):.4f} P {tp/(tp+fp):.3f} R {tp/(tp+fn):.3f} pred/ref {(tp+fp)/(tp+fn):.3f} | Dice {rr[:, 3].mean():.4f}")
