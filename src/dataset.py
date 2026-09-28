"""Reproducible synthetic query embeddings for an isolated routing experiment."""

import torch
from torch.utils.data import Dataset

ROUTES = ("RAG document search", "SQL database query", "general chit-chat", "sensitive action")


class AgentQueryDataset(Dataset):
    """Four separated semantic centroids with per-query Gaussian variation.

    These are synthetic vectors, not embeddings from real text. Accuracy on them
    measures learning this controlled task and cannot establish production fidelity.
    """

    def __init__(self, samples_per_class=500, seed=42, noise=0.35):
        generator = torch.Generator().manual_seed(seed)
        centroids = torch.randn(4, 384, generator=torch.Generator().manual_seed(0))
        centroids = torch.nn.functional.normalize(centroids, dim=1) * 8
        self.labels = torch.arange(4).repeat_interleave(samples_per_class)
        self.embeddings = centroids[self.labels] + noise * torch.randn(
            len(self.labels), 384, generator=generator
        )
        order = torch.randperm(len(self.labels), generator=generator)
        self.embeddings, self.labels = self.embeddings[order], self.labels[order]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
