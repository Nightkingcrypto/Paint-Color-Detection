import pickle
from collections import Counter

import torch
from torch.utils.data import DataLoader
from loguru import logger

from .config import DATASET_ROOT, IMAGE_SIZE, LATENT_DIM, BATCH_SIZE, MODELS_DIR, EMBEDDINGS_PATH
from .dataset import ColorFolderDataset
from .models import ColorVAE
from .utils import set_seed, save_label_counts


def create_color_embeddings(device: str = None):
    """Compute one latent embedding per colour label.

    We encode all images with the VAE encoder and average the latent vectors
    for each label. This gives a robust representation independent of brightness.
    """
    set_seed()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = ColorFolderDataset(DATASET_ROOT, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ColorVAE(latent_dim=LATENT_DIM).to(device)
    vae_path = MODELS_DIR / "vae.pt"
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE model not found at {vae_path}. Train it first.")
    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.eval()

    all_latents = []
    all_labels = []
    label_counts = Counter()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mu, logvar = model.encoder(x)
            z = mu  # we use mean as embedding
            all_latents.append(z.cpu())
            labels_batch = [dataset.idx2label[int(i)] for i in y]
            all_labels.extend(labels_batch)
            label_counts.update(labels_batch)

    all_latents = torch.cat(all_latents, dim=0)

    # Average latent per label
    label_to_vec = {}
    for label in label_counts.keys():
        indices = [i for i, l in enumerate(all_labels) if l == label]
        vecs = all_latents[indices]
        label_to_vec[label] = vecs.mean(dim=0).numpy()

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(label_to_vec, f)

    save_label_counts(label_counts)
    logger.info(f"Saved embeddings for {len(label_to_vec)} colours to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    create_color_embeddings()
