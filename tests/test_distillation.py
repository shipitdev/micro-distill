import torch

from src.benchmark import count_parameters
from src.dataset import AgentQueryDataset
from src.losses import DistillationLoss
from src.models import StudentNet, TeacherNet


def test_shapes_and_loss():
    dataset = AgentQueryDataset(2)
    x, y = dataset.embeddings, dataset.labels
    teacher, student = TeacherNet(), StudentNet()
    assert count_parameters(teacher) == 139972
    assert count_parameters(student) == 12452
    target = teacher(x).detach()
    loss = DistillationLoss()(student(x), target, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None for p in student.parameters())
