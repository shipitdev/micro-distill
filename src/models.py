"""Intent classifiers that read 384-number sentence embeddings."""

from torch import nn


class StudentNet(nn.Module):
    """One linear layer: 384 embedding numbers → 27 Bitext intent logits."""

    def __init__(self, num_intents: int = 27):
        super().__init__()
        self.layer = nn.Linear(384, num_intents)

    def forward(self, embeddings):
        return self.layer(embeddings)


class TeacherNet(nn.Module):
    """Three linear layers: 384 → 256 → 128 → 27 intent logits."""

    def __init__(self, num_intents: int = 27):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_intents),
        )

    def forward(self, embeddings):
        return self.layers(embeddings)
