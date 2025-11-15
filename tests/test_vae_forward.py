from pathlib import Path
import importlib.util
import torch


# --- Load ColorVAE directly from src/models/vae.py using the file path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAE_PATH = PROJECT_ROOT / "src" / "models" / "vae.py"

assert VAE_PATH.is_file(), f"VAE file not found at {VAE_PATH}"

spec = importlib.util.spec_from_file_location("vae_module", VAE_PATH)
vae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vae_module)

ColorVAE = vae_module.ColorVAE


def test_vae_forward():
    """Simple smoke test: VAE forward pass runs and shapes are correct."""
    model = ColorVAE(latent_dim=16)
    x = torch.randn(2, 3, 64, 64)
    x_hat, mu, logvar = model(x)

    assert x_hat.shape == x.shape
    assert mu.shape[0] == x.shape[0]
    assert logvar.shape == mu.shape


