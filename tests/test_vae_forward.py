import sys
from pathlib import Path
import torch

# Make sure the project root (where "src" lives) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ColorVAE



def test_vae_forward():
    model = ColorVAE(latent_dim=16)
    x = torch.randn(2, 3, 64, 64)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == x.shape
    assert mu.shape[0] == x.shape[0]
    assert logvar.shape == mu.shape
