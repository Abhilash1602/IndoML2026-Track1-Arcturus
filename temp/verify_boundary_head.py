"""Session 9 plan, Part 2 Stage 2A verification: shape/gradient/determinism/
backward-compat checks for the boundary head, matching the discipline used
for every other architecture addition this session (e.g. Part 1's category
head). Run before any real training.

Usage: python -m temp.verify_boundary_head
"""
import torch

from src.architecture import DualBranchSED
from src.config import Config
from src.dataset import _build_targets
from src.loss import OnsetOffsetLoss

ok = True


def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    ok &= cond


# 1. _build_targets: boundary_mask is None when the flag is off (backward compat).
config_off = Config()
config_off.SR, config_off.HOP_LENGTH = 16000, 160
events = [{"start": 1.0, "end": 2.0, "category": "animal"}]
total_frames = 1000
mask_off, cat_off, boundary_off = _build_targets(events, total_frames, config_off)
check("boundary_mask is None when USE_BOUNDARY_HEAD is off", boundary_off is None)

# 2. _build_targets: boundary_mask is a correct (T,2) single-frame impulse when on.
config_on = Config()
config_on.SR, config_on.HOP_LENGTH = 16000, 160
config_on.USE_BOUNDARY_HEAD = True
mask_on, cat_on, boundary_on = _build_targets(events, total_frames, config_on)
check("boundary_mask has shape (T,2)", boundary_on is not None and tuple(boundary_on.shape) == (total_frames, 2))
onset_frame = int(1.0 * 16000 / 160)
offset_frame = int(2.0 * 16000 / 160)
check("onset impulse at the exact start frame, exactly one 1", boundary_on[:, 0].sum().item() == 1.0
      and boundary_on[onset_frame, 0].item() == 1.0)
check("offset impulse at the exact end frame, exactly one 1", boundary_on[:, 1].sum().item() == 1.0
      and boundary_on[offset_frame, 1].item() == 1.0)
check("mask unaffected by USE_BOUNDARY_HEAD (byte-identical primary target)",
      torch.equal(mask_off, mask_on))

# 3. An event that runs off the end of the clip gets no offset impulse.
config_edge = Config()
config_edge.SR, config_edge.HOP_LENGTH = 16000, 160
config_edge.USE_BOUNDARY_HEAD = True
events_edge = [{"start": 9.0, "end": 15.0}]  # end frame far past total_frames=1000 (=10s @ 100Hz)
_, _, boundary_edge = _build_targets(events_edge, total_frames, config_edge)
check("event running off the clip end contributes no offset impulse",
      boundary_edge[:, 1].sum().item() == 0.0)
check("...but still contributes its onset impulse", boundary_edge[:, 0].sum().item() == 1.0)

# 4. Architecture: forward() shape/gradient/backward-compat.
config_arch_off = Config()
config_arch_off.CNN_CH = (16, 24, 32, 48)
config_arch_off.CONF_DIM, config_arch_off.CONF_HEADS, config_arch_off.CONF_LAYERS = 64, 4, 2
model_off = DualBranchSED(config_arch_off)
audio = torch.randn(2, config_arch_off.target_samples)
out_off = model_off(audio)
check("forward() returns a bare tensor when USE_BOUNDARY_HEAD is off (backward compat)",
      torch.is_tensor(out_off))

config_arch_on = Config()
config_arch_on.CNN_CH = (16, 24, 32, 48)
config_arch_on.CONF_DIM, config_arch_on.CONF_HEADS, config_arch_on.CONF_LAYERS = 64, 4, 2
config_arch_on.USE_BOUNDARY_HEAD = True
model_on = DualBranchSED(config_arch_on)
main_logits, boundary_logits = model_on(audio)
check("forward() returns (main_logits, boundary_logits) when on",
      main_logits.shape == (2, main_logits.shape[1]) and boundary_logits.shape == (2, main_logits.shape[1], 2))

# gradient flows into boundary_head and the shared trunk
loss_fn = OnsetOffsetLoss()
targets = torch.zeros_like(boundary_logits)
targets[:, 5, 0] = 1.0
targets[:, 20, 1] = 1.0
loss = loss_fn(boundary_logits, targets)
loss.backward()
boundary_head_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                          for p in model_on.boundary_head.parameters())
trunk_grad = model_on.temporal_proj.weight.grad is not None and model_on.temporal_proj.weight.grad.abs().sum().item() > 0
check("gradient flows into boundary_head", boundary_head_grad)
check("gradient flows into the shared trunk (temporal_proj)", trunk_grad)

# eval-mode determinism
model_on.eval()
with torch.no_grad():
    out1 = model_on(audio)
    out2 = model_on(audio)
check("eval-mode determinism (identical outputs on repeated forward passes)",
      torch.equal(out1[0], out2[0]) and torch.equal(out1[1], out2[1]))

print()
print("ALL CHECKS PASSED" if ok else "AT LEAST ONE CHECK FAILED -- do not proceed to training")
