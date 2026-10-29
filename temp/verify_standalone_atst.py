"""3.5-week re-plan, Must-do #2 verification: shape/gradient/determinism/
unfreeze checks for StandaloneATSTSED, matching this session's established
discipline (e.g. temp/verify_boundary_head.py) before any real training.

Usage: python -m temp.verify_standalone_atst
"""
import torch

from src.architecture import StandaloneATSTSED
from src.config import Config

ok = True


def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    ok &= cond


config = Config()
config.CONF_DIM, config.CONF_HEADS, config.CONF_LAYERS = 256, 8, 6
config.MAX_DURATION = 10.0
model = StandaloneATSTSED(config)

# 1. Frozen by default -- ATST-Frame params should not require grad.
n_atst_frozen = sum(1 for p in model.atst_frame.parameters() if not p.requires_grad)
n_atst_total = sum(1 for p in model.atst_frame.parameters())
check("ATST-Frame backbone frozen by default (all params require_grad=False)",
      n_atst_frozen == n_atst_total and n_atst_total > 0)

# 2. Shape check.
audio = torch.randn(2, config.target_samples)
model.eval()
with torch.no_grad():
    out = model(audio)
expected_frames = config.target_samples // config.HOP_LENGTH
check("forward() returns a bare (B,T) tensor",
      torch.is_tensor(out) and out.shape == (2, expected_frames))

# 3. Gradient flow into the new temporal head / classifier (frozen backbone
# should get NO gradient at this stage).
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
atst_grad = any(p.grad is not None for p in model.atst_frame.parameters())
check("gradient flows into temporal_proj", head_grad)
check("gradient flows into classifier", classifier_grad)
check("NO gradient reaches the frozen ATST-Frame backbone", not atst_grad)

# 4. Eval-mode determinism.
model.eval()
with torch.no_grad():
    out1 = model(audio)
    out2 = model(audio)
check("eval-mode determinism (identical outputs on repeated forward passes)",
      torch.equal(out1, out2))

# 5. train() keeps the frozen backbone in eval mode (dropout/etc. off).
model.train()
check("train() keeps ATST-Frame backbone in eval mode while frozen",
      not model.atst_frame.training)

# 6. unfreeze_atst_frame() correctness.
model.unfreeze_atst_frame()
n_atst_trainable = sum(1 for p in model.atst_frame.parameters() if p.requires_grad)
check("unfreeze_atst_frame() makes all ATST-Frame params trainable",
      n_atst_trainable == n_atst_total)
model.train()
check("train() now lets ATST-Frame follow the parent's train mode (post-unfreeze)",
      model.atst_frame.training)

# 7. separate_params() still works for LLRD (reused directly from ATSTWrapper).
atst_groups = model.atst_frame.separate_params()
check("separate_params() returns 12 LLRD groups", len(atst_groups) == 12)
n_grouped = sum(p.numel() for g in atst_groups for p in g)
n_total_params = sum(p.numel() for p in model.atst_frame.parameters())
check("LLRD grouping covers every ATST-Frame parameter (no drops)",
      n_grouped == n_total_params)

# 8. Gradient now reaches the (unfrozen) backbone too.
model.train()
out = model(audio)
loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
loss.backward()
atst_grad_after_unfreeze = any(
    p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.atst_frame.parameters())
check("gradient reaches ATST-Frame backbone after unfreeze_atst_frame()", atst_grad_after_unfreeze)

print()
print(f"params: total={sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
      f"ATST-Frame={n_total_params/1e6:.2f}M, new-head={sum(p.numel() for n,p in model.named_parameters() if not n.startswith('atst_frame'))/1e6:.2f}M")
print("ALL CHECKS PASSED" if ok else "AT LEAST ONE CHECK FAILED -- do not proceed to training")
