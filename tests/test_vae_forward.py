import torch

from src.models import ColorVAE


def test_vae_forward():
    model = ColorVAE(latent_dim=16)
    x = torch.randn(2, 3, 64, 64)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == x.shape
    assert mu.shape[0] == x.shape[0]
    assert logvar.shape == mu.shape
