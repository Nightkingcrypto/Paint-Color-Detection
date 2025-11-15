import argparse

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from loguru import logger

from .config import (
    DATASET_ROOT,
    IMAGE_SIZE,
    LATENT_DIM,
    BATCH_SIZE,
    GAN_EPOCHS,
    GAN_LR,
    BETAS,
    MODELS_DIR,
)
from .dataset import ColorFolderDataset
from .models import Generator, Discriminator
from .mlflow_utils import mlflow_run
from .utils import set_seed


def train_gan(device: str = None):
    """Train a small GAN purely for data augmentation / VAE-GAN demonstration.

    This is **not** critical for the colour matching logic, but you can show
    generated samples in the report and video.
    """
    set_seed()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = ColorFolderDataset(DATASET_ROOT, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    gen = Generator(latent_dim=LATENT_DIM).to(device)
    disc = Discriminator().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=GAN_LR, betas=BETAS)
    opt_d = torch.optim.Adam(disc.parameters(), lr=GAN_LR, betas=BETAS)

    criterion = nn.BCELoss()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    gen_path = MODELS_DIR / "gan_generator.pt"
    disc_path = MODELS_DIR / "gan_discriminator.pt"

    run_params = {
        "latent_dim": LATENT_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": GAN_EPOCHS,
        "lr": GAN_LR,
    }

    with mlflow_run("train_gan", run_params):
        for epoch in range(1, GAN_EPOCHS + 1):
            gen.train()
            disc.train()
            g_loss_sum = 0.0
            d_loss_sum = 0.0

            for x, _ in dataloader:
                x = x.to(device)
                b_size = x.size(0)

                # Train Discriminator: real
                real_labels = torch.ones(b_size, device=device)
                fake_labels = torch.zeros(b_size, device=device)

                opt_d.zero_grad()
                out_real = disc(x)
                loss_real = criterion(out_real, real_labels)

                # Train Discriminator: fake
                z = torch.randn(b_size, LATENT_DIM, device=device)
                fake = gen(z).detach()
                out_fake = disc(fake)
                loss_fake = criterion(out_fake, fake_labels)

                d_loss = loss_real + loss_fake
                d_loss.backward()
                opt_d.step()

                # Train Generator
                opt_g.zero_grad()
                z = torch.randn(b_size, LATENT_DIM, device=device)
                fake = gen(z)
                out_fake2 = disc(fake)
                g_loss = criterion(out_fake2, real_labels)
                g_loss.backward()
                opt_g.step()

                d_loss_sum += d_loss.item() * b_size
                g_loss_sum += g_loss.item() * b_size

            avg_d = d_loss_sum / len(dataset)
            avg_g = g_loss_sum / len(dataset)

            mlflow.log_metrics({"d_loss": avg_d, "g_loss": avg_g}, step=epoch)
            logger.info(
                f"Epoch {epoch}/{GAN_EPOCHS} - D_loss={avg_d:.4f}, G_loss={avg_g:.4f}"
            )

        torch.save(gen.state_dict(), gen_path)
        torch.save(disc.state_dict(), disc_path)
        mlflow.pytorch.log_model(gen, artifact_path="gan_generator")
        mlflow.pytorch.log_model(disc, artifact_path="gan_discriminator")
        logger.info(f"Saved GAN models to {MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()
    train_gan(device=args.device)
