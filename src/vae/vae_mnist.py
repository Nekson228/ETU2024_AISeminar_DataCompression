import torch
import torch.nn as nn


class VaeEncoderMnist(nn.Module):
    def __init__(self, latent_dim: int):
        super(VaeEncoderMnist, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=-1)
        return mu, log_var


class VaeDecoderMnist(nn.Module):
    def __init__(self, latent_dim: int):
        super(VaeDecoderMnist, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
