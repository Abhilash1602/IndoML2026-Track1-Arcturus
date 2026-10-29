"""3.5-week re-plan: shape/gradient/determinism/unfreeze checks for
StandaloneDashengSED, matching this session's established discipline
(e.g. temp/verify_standalone_atst.py) before any real training.

Usage: python -m temp.verify_standalone_dasheng
"""
import torch

from src.architecture import StandaloneDashengSED
from src.config import Config

ok = True


def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    ok &= cond


config = Config()
config.CONF_DIM, config.CONF_HEADS, config.CONF_LAYERS = 256, 8, 6
config.MAX_DURATION = 10.0
model = StandaloneDashengSED(config)

# 1. Frozen by default.
n_frozen = sum(1 for p in model.dasheng.parameters() if not p.requires_grad)
n_total = sum(1 for p in model.dasheng.parameters())
check("Dasheng backbone frozen by default (all params require_grad=False)",
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
dasheng_grad = any(p.grad is not None for p in model.dasheng.parameters())
check("gradient flows into temporal_proj", head_grad)
check("gradient flows into classifier", classifier_grad)
check("NO gradient reaches the frozen Dasheng backbone", not dasheng_grad)

# 4. Eval-mode determinism.
model.eval()
with torch.no_grad():
    out1 = model(audio)
    out2 = model(audio)
check("eval-mode determinism (identical outputs on repeated forward passes)",
      torch.equal(out1, out2))

# 5. train() keeps the frozen backbone in eval mode.
model.train()
check("train() keeps Dasheng backbone in eval mode while frozen",
      not model.dasheng.training)

# 6. unfreeze_dasheng() correctness.
model.unfreeze_dasheng()
n_trainable = sum(1 for p in model.dasheng.parameters() if p.requires_grad)
check("unfreeze_dasheng() makes all Dasheng params trainable", n_trainable == n_total)
model.train()
check("train() now lets Dasheng follow the parent's train mode (post-unfreeze)",
      model.dasheng.training)

# 7. Gradient now reaches the (unfrozen) backbone too.
model.train()
out = model(audio)
loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
loss.backward()
dasheng_grad_after_unfreeze = any(
    p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.dasheng.parameters())
check("gradient reaches Dasheng backbone after unfreeze_dasheng()", dasheng_grad_after_unfreeze)

# 8. mel_forward matches DashengFeatureExtractor's own output shape/params.
from transformers import AutoFeatureExtractor
fe = AutoFeatureExtractor.from_pretrained("mispeech/dasheng-base", trust_remote_code=True)
ref = fe(audio, sampling_rate=16000, return_tensors="pt")["input_values"]
ours = model.mel_forward(audio)
check("mel_forward() output shape matches DashengFeatureExtractor's own output",
      ours.shape == ref.shape)
check("mel_forward() output numerically matches DashengFeatureExtractor's own output",
      torch.allclose(ours, ref, atol=1e-3))

print()
print(f"params: total={sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
      f"Dasheng={n_total/1e6:.2f}M, new-head={sum(p.numel() for n,p in model.named_parameters() if not n.startswith('dasheng'))/1e6:.2f}M")
print("ALL CHECKS PASSED" if ok else "AT LEAST ONE CHECK FAILED -- do not proceed to training")
