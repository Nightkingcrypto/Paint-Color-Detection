from datetime import datetime
import csv
import io
import pickle
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from src.config import (
    EMBEDDINGS_PATH,
    MODELS_DIR,
    LOGS_DIR,
    IMAGE_SIZE,
    LATENT_DIM,
)
from src.models import ColorVAE

app = FastAPI(title="Color Detector VAE+GAN")

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_vae(device: str):
    model = ColorVAE(latent_dim=LATENT_DIM).to(device)
    vae_path = MODELS_DIR / "vae.pt"
    if not vae_path.exists():
        raise FileNotFoundError("VAE model not found. Train it first (src/train_vae.py).")
    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.eval()
    return model


def load_embeddings():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Embeddings file not found. Run `python -m src.embeddings` after training the VAE."
        )
    with open(EMBEDDINGS_PATH, "rb") as f:
        label_to_vec = pickle.load(f)
    return label_to_vec


def preprocess_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    return transform(img).unsqueeze(0)  # [1, C, H, W]


def compute_latent(model: ColorVAE, x: torch.Tensor, device: str):
    x = x.to(device)
    with torch.no_grad():
        mu, logvar = model.encoder(x)
    return mu.cpu().numpy()[0]


def rank_colours(latent_vec: np.ndarray, label_to_vec: dict, top_k: int = 3):
    names: List[str] = []
    dists: List[float] = []
    for label, vec in label_to_vec.items():
        v = np.asarray(vec)
        dist = np.linalg.norm(latent_vec - v)
        names.append(label)
        dists.append(dist)
    order = np.argsort(dists)[:top_k]
    return [(names[i], float(dists[i])) for i in order]


def log_prediction(top3: List[tuple]):
    csv_path = LOGS_DIR / "predictions.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "top1_color", "top2_color", "top3_color"])
        timestamp = datetime.utcnow().isoformat()
        row = [timestamp] + [c[0] for c in top3]
        writer.writerow(row)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_vae(device)
        label_to_vec = load_embeddings()
        x = preprocess_image(contents)
        latent = compute_latent(model, x, device)
        top3 = rank_colours(latent, label_to_vec, top_k=3)
        log_prediction(top3)
        logger.info(f"Prediction top-3: {top3}")
    except Exception as e:
        logger.exception(e)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": None,
                "error": str(e),
            },
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": top3,
            "filename": file.filename,
        },
    )
