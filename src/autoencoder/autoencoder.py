import torch
import torch.nn as nn


class MNISTAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super(MNISTAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.Sigmoid(),
            nn.Linear(500, 300),
            nn.Sigmoid(),
            nn.Linear(300, 100),
            nn.Sigmoid(),
            nn.Linear(100, encoding_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 100),
            nn.Sigmoid(),
            nn.Linear(100, 300),
            nn.Sigmoid(),
            nn.Linear(300, 500),
            nn.Sigmoid(),
            nn.Linear(500, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.decoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
