"""3.5-week re-plan: shape/gradient/determinism/unfreeze checks for
StandaloneBEATsSED, matching this session's established discipline
(e.g. temp/verify_standalone_atst.py) before any real training.

Usage: python -m temp.verify_standalone_beats
"""
import torch

from src.architecture import StandaloneBEATsSED
from src.config import Config

ok = True


def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    ok &= cond


config = Config()
config.CONF_DIM, config.CONF_HEADS, config.CONF_LAYERS = 256, 8, 6
config.MAX_DURATION = 10.0
model = StandaloneBEATsSED(config)

# 1. Frozen by default.
n_frozen = sum(1 for p in model.beats.parameters() if not p.requires_grad)
n_total = sum(1 for p in model.beats.parameters())
check("BEATs backbone frozen by default (all params require_grad=False)",
      n_frozen == n_total and n_total > 0)

# 2. Shape check.
audio = torch.randn(2, config.target_samples)
model.eval()
with torch.no_grad():
    out = model(audio)
expected_frames = config.target_samples // config.HOP_LENGTH
check("forward() returns a bare (B,T) tensor",
      torch.is_tensor(out) and out.shape == (2, expected_frames))

# 3. Gradient flow into the new head only (frozen backbone gets none).
model.train()
out = model(audio)
target = torch.zeros_like(out)
target[:, 10] = 1.0
loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
loss.backward()
head_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                for p in model.temporal_proj.parameters())
classifier_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                      for p in model.classifier.parameters())
dasheng_grad = any(p.grad is not None for p in model.beats.parameters())
check("gradient flows into temporal_proj", head_grad)
check("gradient flows into classifier", classifier_grad)
check("NO gradient reaches the frozen BEATs backbone", not dasheng_grad)

# 4. Eval-mode determinism.
model.eval()
with torch.no_grad():
    out1 = model(audio)
    out2 = model(audio)
check("eval-mode determinism (identical outputs on repeated forward passes)",
      torch.equal(out1, out2))

# 5. train() keeps the frozen backbone in eval mode.
model.train()
check("train() keeps BEATs backbone in eval mode while frozen",
      not model.beats.training)

# 6. unfreeze_beats() correctness.
model.unfreeze_beats()
n_trainable = sum(1 for p in model.beats.parameters() if p.requires_grad)
check("unfreeze_beats() makes all BEATs params trainable", n_trainable == n_total)
model.train()
check("train() now lets BEATs follow the parent's train mode (post-unfreeze)",
      model.beats.training)

# 7. Gradient now reaches the (unfrozen) backbone too.
model.train()
out = model(audio)
loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
loss.backward()
dasheng_grad_after_unfreeze = any(
    p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.beats.parameters())
check("gradient reaches BEATs backbone after unfreeze_beats()", dasheng_grad_after_unfreeze)

# 8. LLRD grouping covers every BEATs parameter exactly once.
groups = model.separate_params()
n_numel = sum(p.numel() for p in model.beats.parameters())
check("separate_params(): 12 groups covering every BEATs param exactly once",
      len(groups) == 12 and sum(p.numel() for g in groups for p in g) == n_numel
      and len({id(p) for g in groups for p in g}) == sum(len(g) for g in groups))

print()
print(f"params: total={sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
      f"BEATs={n_total/1e6:.2f}M, new-head={sum(p.numel() for n,p in model.named_parameters() if not n.startswith('beats'))/1e6:.2f}M")
print("ALL CHECKS PASSED" if ok else "AT LEAST ONE CHECK FAILED -- do not proceed to training")
