"""Session 9 plan, Part 2 Stage 2A mandatory hard gate: a hand-built synthetic
unit test for OnsetOffsetLoss (src/loss.py). No model, no audio -- this is the
only ground truth available since no reference implementation of
arXiv:2601.04178's OOL exists. Do not proceed to touching real training data
until every check here passes.

Paper's confirmed equation (see src/loss.py's OnsetOffsetLoss docstring):
L = -(1/CT) * sum_{c,t} [ (1-p)^alpha * log(p) if y=1 else p^alpha * log(1-p) ],
alpha=2 (focusing exponent, no separate class-balance term).

Usage: python -m temp.test_onset_offset_loss_unit
"""
import math

import torch

from src.loss import OnsetOffsetLoss


def check(name, actual, expected, tol=1e-5):
    ok = abs(actual - expected) < tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: actual={actual:.7f} expected={expected:.7f} (tol={tol})")
    return ok


def main():
    all_ok = True
    loss_fn = OnsetOffsetLoss(gamma=2.0)

    # Case 1: T=6, all p=0.5 (logit=0), single onset impulse at frame 2.
    # Hand computation: p_t=0.5 for every frame regardless of target (since
    # p=1-p=0.5). Per-frame term = -(1-0.5)^2 * log(0.5) = -(0.25 * -0.6931472)
    # = 0.1732868, identical for all 6 frames (2 pos-shaped, 4 neg-shaped
    # elements collapse to the same value at p=0.5) -> mean = 0.1732868.
    logits1 = torch.zeros(1, 6, 1)
    targets1 = torch.zeros(1, 6, 1)
    targets1[0, 2, 0] = 1.0
    expected1 = -((1 - 0.5) ** 2 * math.log(0.5))
    loss1 = loss_fn(logits1, targets1).item()
    all_ok &= check("Case 1 (uniform p=0.5, single onset impulse)", loss1, expected1)

    # Case 2: T=2, one confident-correct positive (p=0.9, y=1) and one
    # confident-WRONG negative (p=0.9, y=0) -- demonstrates focal down-
    # weighting of easy correct predictions vs a large penalty for a
    # confidently wrong one.
    #   frame0 (y=1,p=0.9): p_t=0.9, term=-(0.1^2 * log(0.9))=0.00105360...
    #   frame1 (y=0,p=0.9): p_t=0.1, term=-(0.9^2 * log(0.1))=1.86509412...
    #   mean = (0.00105360 + 1.86509412) / 2
    p_pos, p_neg = 0.9, 0.9
    logit_pos = math.log(p_pos / (1 - p_pos))
    logit_neg = math.log(p_neg / (1 - p_neg))
    logits2 = torch.tensor([[[logit_pos], [logit_neg]]])
    targets2 = torch.tensor([[[1.0], [0.0]]])
    term_pos = -((1 - p_pos) ** 2 * math.log(p_pos))
    term_neg = -((p_neg) ** 2 * math.log(1 - p_neg))
    expected2 = (term_pos + term_neg) / 2
    loss2 = loss_fn(logits2, targets2).item()
    all_ok &= check("Case 2 (confident-correct pos vs confident-wrong neg)", loss2, expected2)
    all_ok &= check("Case 2 sanity: confidently-wrong term >> confidently-correct term",
                     1.0 if term_neg > 100 * term_pos else 0.0, 1.0, tol=0.5)

    # Case 3: perfect predictions (p->1 for y=1, p->0 for y=0) should drive
    # loss toward 0, confirming the focal weighting genuinely vanishes for
    # confidently-correct predictions rather than just being small.
    logits3 = torch.tensor([[[20.0], [-20.0]]])  # sigmoid(+-20) ~= 1.0 / ~0.0
    targets3 = torch.tensor([[[1.0], [0.0]]])
    loss3 = loss_fn(logits3, targets3).item()
    all_ok &= check("Case 3 (near-perfect predictions -> near-zero loss)", loss3, 0.0, tol=1e-6)

    # Case 4: gradient descent sanity -- the plan's own explicit gate
    # criterion ("confirm the loss decreases under manual gradient steps").
    # Start from a deliberately bad prediction (p~0.1 for a y=1 target) and
    # confirm a few manual SGD steps monotonically reduce the loss.
    torch.manual_seed(0)
    logits4 = torch.nn.Parameter(torch.tensor([[[math.log(0.1 / 0.9)]]]))
    targets4 = torch.tensor([[[1.0]]])
    prev = float("inf")
    monotonic = True
    for step in range(10):
        loss4 = loss_fn(logits4, targets4)
        loss4.backward()
        with torch.no_grad():
            logits4 -= 0.5 * logits4.grad
        logits4.grad = None
        val = loss4.item()
        if val >= prev - 1e-9:
            monotonic = False
        prev = val
    all_ok &= check("Case 4 (10 manual SGD steps monotonically decrease loss)",
                     1.0 if monotonic else 0.0, 1.0)
    print(f"  final loss after 10 steps: {prev:.7f} (started from p=0.1 target y=1, "
          f"predicted prob now {torch.sigmoid(logits4).item():.4f})")

    print()
    if all_ok:
        print("ALL CHECKS PASSED -- OnsetOffsetLoss matches the paper's literal "
              "equation. Safe to proceed to real-data integration (Stage 2A next steps).")
    else:
        print("AT LEAST ONE CHECK FAILED -- do NOT proceed to real training data "
              "until this is resolved.")
    return all_ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
