"""Train teacher, baseline student, and distilled student on synthetic routes."""

import argparse

import torch
from tabulate import tabulate
from torch import nn
from torch.utils.data import DataLoader

from src.benchmark import count_parameters, evaluate_accuracy, measure_latency, measure_model_size_kb
from src.dataset import AgentQueryDataset
from src.losses import DistillationLoss
from src.models import StudentNet, TeacherNet


def train_model(model, loader, epochs, device, teacher=None):
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = DistillationLoss() if teacher else nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0.0
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embeddings)
            if teacher:
                # Frozen teacher targets should not create an autograd graph.
                with torch.no_grad():
                    teacher_logits = teacher(embeddings)
                loss = criterion(logits, teacher_logits, labels)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"{model.__class__.__name__} epoch {epoch + 1}/{epochs}: loss {total / len(loader):.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples_per_class < 1:
        parser.error("epochs and samples-per-class must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train = DataLoader(AgentQueryDataset(args.samples_per_class, args.seed), batch_size=64, shuffle=True)
    held_out = DataLoader(AgentQueryDataset(100, args.seed + 1), batch_size=128)
    teacher = train_model(TeacherNet(), train, args.epochs, device)
    teacher.eval()
    baseline = train_model(StudentNet(), train, args.epochs, device)
    distilled = train_model(StudentNet(), train, args.epochs, device, teacher)

    rows = []
    for name, model in (("Teacher", teacher), ("Baseline student", baseline), ("Distilled student", distilled)):
        accuracy = evaluate_accuracy(model, held_out, device)
        p50, p95 = measure_latency(model)
        rows.append((name, count_parameters(model), f"{measure_model_size_kb(model):.1f}",
                     f"{p50:.4f}", f"{p95:.4f}", f"{accuracy:.1%}"))
    print(tabulate(rows, headers=("Model", "Parameters", "Weights KiB", "p50 ms", "p95 ms", "Accuracy"), tablefmt="github"))


if __name__ == "__main__":
    main()
