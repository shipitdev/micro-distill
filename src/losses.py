"""Hinton-style soft-target distillation."""

import torch.nn.functional as F
from torch import nn


class DistillationLoss(nn.Module):
    """Blend 30% hard-label CE with 70% temperature-scaled soft-target KL."""

    def __init__(self, temperature=4.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, labels):
        t = self.temperature
        # The teacher is a fixed target; no gradient should update it here.
        soft_targets = F.softmax(teacher_logits.detach() / t, dim=1)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / t, dim=1), soft_targets, reduction="batchmean"
        ) * t**2
        return 0.3 * F.cross_entropy(student_logits, labels) + 0.7 * soft_loss
