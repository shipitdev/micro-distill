"""CPU inference and held-out accuracy measurements."""

import statistics
import time

import torch


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def measure_model_size_kb(model):
    """Float32 parameter storage in KiB; excludes Python and runtime overhead."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024


def measure_latency(model, passes=1000):
    """Single-query CPU forward latency after warm-up, in milliseconds."""
    model = model.cpu().eval()
    query = torch.randn(1, 384)
    with torch.inference_mode():
        for _ in range(100):
            model(query)
        samples = []
        for _ in range(passes):
            start = time.perf_counter_ns()
            model(query)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    samples.sort()
    return statistics.median(samples), samples[int(0.95 * (passes - 1))]


def evaluate_accuracy(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for embeddings, labels in loader:
            predictions = model(embeddings.to(device)).argmax(dim=1)
            correct += (predictions == labels.to(device)).sum().item()
            total += len(labels)
    return correct / total
