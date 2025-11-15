import argparse
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader
from loguru import logger

from .config import (
    DATASET_ROOT,
    IMAGE_SIZE,
    LATENT_DIM,
    BATCH_SIZE,
    VAE_EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
)
from .dataset import ColorFolderDataset
from .models import ColorVAE, vae_loss
from .mlflow_utils import mlflow_run
from .utils import set_seed


def train_vae(device: str = None):
    set_seed()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = ColorFolderDataset(DATASET_ROOT, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = ColorVAE(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "vae.pt"

    run_params = {
        "image_size": IMAGE_SIZE,
        "latent_dim": LATENT_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": VAE_EPOCHS,
        "lr": LEARNING_RATE,
    }

    with mlflow_run("train_vae", run_params):
        for epoch in range(1, VAE_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            recon_sum = 0.0
            kl_sum = 0.0
            for batch_idx, (x, _) in enumerate(dataloader):
                x = x.to(device)
                optimizer.zero_grad()
                x_hat, mu, logvar = model(x)
                loss, recon, kl = vae_loss(x, x_hat, mu, logvar)
                loss.backward()
                optimizer.step()

                batch_size_actual = x.size(0)
                epoch_loss += loss.item() * batch_size_actual
                recon_sum += recon.item() * batch_size_actual
                kl_sum += kl.item() * batch_size_actual

            avg_loss = epoch_loss / len(dataset)
            avg_recon = recon_sum / len(dataset)
            avg_kl = kl_sum / len(dataset)

            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "recon_loss": avg_recon,
                    "kl_div": avg_kl,
                },
                step=epoch,
            )

            logger.info(
                f"Epoch {epoch}/{VAE_EPOCHS} - loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}"
            )

        # Save model locally and in MLflow
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, artifact_path="vae_model")
        logger.info(f"Saved VAE model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()
    train_vae(device=args.device)
