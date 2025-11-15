# Color Detector with VAE + GAN (MLOps Coursework)

This project implements a **color detector** using a **Variational Autoencoder (VAE)** and a **GAN** on top of a paint-color dataset.
Each color has its own folder (for example `0N05 Fawn`) and contains a *key* image plus several brightness variations.

The model learns a latent representation of each color using a VAE. A simple GAN is used to generate extra color samples in the
same latent space, helping the model become more robust to brightness changes. At inference time, the app returns the **top‑3
closest colors** in the latent space for a user‑uploaded image.

The project is organised and instrumented with **MLOps practices**:

- Conda environment and `requirements.txt`
- Reproducible training scripts with **MLflow** experiment tracking
- Version control with **Git** (you will create the repo)
- Explanation and config for **DVC** for data versioning
- **GitHub Actions** workflow for CI (pytest)
- **FastAPI** + **Docker** for deployment
- Simple **logging** & **model drift** analysis script
- Jupyter Notebook report template

You should connect this project to your own GitHub repository and your local dataset path.

---

## 1. Project structure

```text
color_vae_gan_mlops/
├─ README.md
├─ environment.yml
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/                 # (optional) you can copy a small sample here
│  └─ processed/
├─ logs/
│  ├─ app.log
│  └─ predictions.csv
├─ models/
│  └─ (saved models will be stored here)
├─ notebooks/
│  └─ report_template.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ dataset.py
│  ├─ utils.py
│  ├─ mlflow_utils.py
│  ├─ embeddings.py
│  ├─ monitor_drift.py
│  ├─ train_vae.py
│  ├─ train_gan.py
│  └─ models/
│     ├─ __init__.py
│     ├─ vae.py
│     └─ gan.py
├─ app/
│  ├─ main.py              # FastAPI app
│  ├─ templates/
│  │  └─ index.html
│  └─ static/
│     └─ style.css
├─ tests/
│  ├─ test_imports.py
│  └─ test_vae_forward.py
├─ docker/
│  └─ Dockerfile
├─ .github/
│  └─ workflows/
│     └─ ci.yml
└─ docs/
   └─ dvc_usage.md
```

---

## 2. Quickstart

1. **Create conda env**

```bash
conda create -n color_vae_gan python=3.11 -y
conda activate color_vae_gan
pip install -r requirements.txt
```

2. **Set your dataset path**

Edit `src/config.py` and change `DATASET_ROOT` to your actual folder, e.g.:

```python
DATASET_ROOT = r"F:\Dekstop\Colors\Dataset"
```

3. **Train VAE with MLflow**

```bash
conda activate color_vae_gan
python -m src.train_vae
```

Start MLflow UI in another terminal:

```bash
mlflow ui
```

4. **Train GAN (optional but recommended)**

```bash
python -m src.train_gan
```

5. **Create color embeddings**

```bash
python -m src.embeddings
```

6. **Run the FastAPI app**

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in the browser, upload a color patch image, and see the top‑3 predicted colors.

7. **Run tests (for CI and locally)**

```bash
pytest
```

8. **Docker (for deployment)**

```bash
docker build -t color-vae-gan -f docker/Dockerfile .
docker run -p 8000:8000 color-vae-gan
```

---

For details on each script and how to use Git, DVC, MLflow and GitHub Actions for your coursework report,
see the comments inside the code and the notebook template in `notebooks/report_template.ipynb`.
