"""Small checks for the two failure-prone parts of the experiment."""

from random import Random

import torch

from src.dataset import comparison_key, remove_one_letter, split_requests, training_typo_copies
from src.losses import DistillationLoss
from src.models import StudentNet, TeacherNet


def test_split_keeps_matching_requests_together():
    rows = [
        {"utterance": text, "intent": intent}
        for intent in ("a", "b")
        for text in (f"{intent} first?", f"{intent} FIRST", f"{intent} second", f"{intent} third")
    ]
    names, train, validation, test = split_requests(rows)
    assert names == ["a", "b"]
    split_keys = [{comparison_key(rows[i]["utterance"]) for i in ids} for ids in (train, validation, test)]
    assert split_keys[0].isdisjoint(split_keys[1])
    assert split_keys[0].isdisjoint(split_keys[2])
    assert split_keys[1].isdisjoint(split_keys[2])
    assert len(train) + len(validation) + len(test) == len(rows)
    assert remove_one_letter("cancel order", Random(7)) != "cancel order"
    copies = training_typo_copies(rows, train)
    assert all(comparison_key(text) not in {comparison_key(r["utterance"]) for r in rows} for _, text in copies)


def test_distillation_updates_only_student():
    torch.manual_seed(42)
    teacher, student = TeacherNet(), StudentNet()
    inputs = torch.randn(3, 384)
    labels = torch.tensor([0, 1, 2])
    with torch.no_grad():
        teacher_scores = teacher(inputs)
    loss = DistillationLoss()(student(inputs), teacher_scores, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(weight.grad is None for weight in teacher.parameters())
    assert any(weight.grad is not None for weight in student.parameters())
