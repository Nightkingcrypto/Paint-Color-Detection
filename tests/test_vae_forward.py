import sys
from pathlib import Path

# --- Ensure project root and 'src' folder are on sys.path ---
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]          # repo root
SRC_DIR = PROJECT_ROOT / "src"          # src folder

for p in (PROJECT_ROOT, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import torch

try:
    from src.models import ColorVAE
except ModuleNotFoundError:
    # Fallback if src is added directly
    from models import ColorVAE


def test_vae_forward():
    """Simple smoke test: VAE forward pass runs and shapes are correct."""
    model = ColorVAE(latent_dim=16)
    x = torch.randn(2, 3, 64, 64)
    x_hat, mu, logvar = model(x)

    assert x_hat.shape == x.shape
    assert mu.shape[0] == x.shape[0]
    assert logvar.shape == mu.shape
