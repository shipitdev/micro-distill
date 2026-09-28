"""MLPs operating on 384-dimensional query embeddings."""

from torch import nn


class TeacherNet(nn.Module):
    """Four linear layers: 384 → 256 → 128 → 64 → four route logits."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, embeddings):
        return self.layers(embeddings)


class StudentNet(nn.Module):
    """Two linear layers: 384 → 32 → four route logits."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(384, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, embeddings):
        return self.layers(embeddings)
