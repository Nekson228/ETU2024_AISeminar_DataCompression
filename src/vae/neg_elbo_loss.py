import torch
import torch.nn.functional as F


def negative_elbo_loss(x_reconstructed: torch.Tensor, x: torch.Tensor,
                       mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute the Negative ELBO loss.

    Parameters:
    - recon_x: Reconstructed input (output of the decoder).
    - x: Original input.
    - mu: Mean from the encoder's latent space.
    - logvar: Log variance from the encoder's latent space.

    Returns:
    - loss: Negative ELBO loss.
    """
    l_rec = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')

    l_reg = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return l_rec + l_reg
