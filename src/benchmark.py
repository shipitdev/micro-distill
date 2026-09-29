"""Measure accuracy, model storage, and warm CPU inference latency."""

from collections import Counter
from statistics import median
from time import perf_counter_ns

import torch


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def measure_model_size_kib(model) -> float:
    """Parameter storage only; excludes Python, tokenizers, and runtime overhead."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024


def measure_latency(call, warmups: int = 10, passes: int = 100) -> tuple[float, float]:
    """Return p50/p95 milliseconds for a warmed, single-request CPU call."""
    if passes < 1:
        raise ValueError("passes must be positive")
    for _ in range(warmups):
        call()
    times = []
    for _ in range(passes):
        start = perf_counter_ns()
        call()
        times.append((perf_counter_ns() - start) / 1_000_000)
    times.sort()
    return median(times), times[int(0.95 * (passes - 1))]


def predict(model, embeddings) -> list[int]:
    model.eval()
    with torch.inference_mode():
        return model(embeddings).argmax(dim=1).tolist()


def accuracy(predictions: list[int], answers: list[int]) -> float:
    if not answers or len(predictions) != len(answers):
        raise ValueError("predictions and answers must have the same nonzero length")
    return sum(guess == answer for guess, answer in zip(predictions, answers)) / len(answers)


def per_intent_recall(predictions: list[int], answers: list[int], names: list[str]):
    """Return (correct, total) for every source intent, including rare ones."""
    total = Counter(answers)
    correct = Counter(answer for guess, answer in zip(predictions, answers) if guess == answer)
    return {name: (correct[index], total[index]) for index, name in enumerate(names)}


def common_confusions(predictions: list[int], answers: list[int], names: list[str], limit=5):
    counts = Counter((names[answer], names[guess]) for guess, answer in zip(predictions, answers) if guess != answer)
    return [(correct, guessed, count) for (correct, guessed), count in counts.most_common(limit)]
