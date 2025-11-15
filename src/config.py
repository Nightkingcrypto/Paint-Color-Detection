from pathlib import Path

# === PATHS ===
# TODO: Change this to your actual dataset path on your machine.
# Each subfolder inside this root must be a color name like '0N05 Fawn'
DATASET_ROOT = Path(r"C:\\Users\\kaThi\\OneDrive\\Desktop\\color_vae_gan_mlops\\Dataset")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
EMBEDDINGS_PATH = MODELS_DIR / "color_embeddings.pkl"
LABEL_COUNTS_PATH = MODELS_DIR / "label_counts.json"

# === TRAINING HYPERPARAMETERS ===
IMAGE_SIZE = 64
LATENT_DIM = 32
BATCH_SIZE = 64
VAE_EPOCHS = 25
GAN_EPOCHS = 25
LEARNING_RATE = 1e-3
GAN_LR = 2e-4
BETAS = (0.5, 0.999)

# === MLFLOW ===
MLFLOW_EXPERIMENT = "color_vae_gan"
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()  # local folder with proper file:// URI

# === MISC ===
SEED = 45
