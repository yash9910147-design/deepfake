# Deepfake Lens — Xception

An image deepfake detector powered by a trained Xception model. The browser
interface and FastAPI inference service can run locally or be deployed
separately. The app does not train a model during inference.

## Setup

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon, the requirements include TensorFlow Metal GPU support.

## Install the trained model

Model weights are intentionally not stored in Git. Copy your trained Xception
HDF5 file to the expected local path:

```bash
mkdir -p models
cp /path/to/xception_deepfake_finetuned.h5 models/xception_deepfake.h5
```

The model must accept 224 × 224 RGB images and return one sigmoid output.
The supplied project model uses `[0, 1]` pixel scaling and its sigmoid output
represents the **Real** class; inference converts that value to a Fake score for
display.

## Run locally

```bash
source .venv/bin/activate
python app.py
```

Open <http://127.0.0.1:8000>. Upload an image and select Xception to get a
prediction. The interface shows whether the model file is available.

Alternatively, run the API server with:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Command-line inference

```bash
python predict.py --image /path/to/image.jpg
```

## Deployment

The Vercel configuration serves the static frontend from `static/`. A separate
FastAPI backend is required for predictions; the frontend reports when that
backend is not connected. The backend must have the trained model file and set
`MODEL_DIR` to the directory containing it. Set `CORS_ORIGINS` to a
comma-separated list of trusted frontend origins if the browser calls the API
directly from another domain.

The HDF5 model is intentionally excluded from Git. Do not add it to the public
repository to deploy it; provide it to the backend host using that provider's
private storage or secure file-transfer mechanism.

## Training and evaluation

Training is explicit and is never started by the web app:

```bash
python train.py --epochs 10 --fine-tune-epochs 5 --batch-size 32
python evaluate.py
```

Use `--resume` to resume a compatible interrupted training run. Evaluation
uses the dataset test split and writes reports and plots under `results/`.

## Dataset

The training scripts download `manjilkarki/deepfake-and-real-images` with
KaggleHub unless `--data-root` points to a local copy. Kaggle authentication
may be required depending on the account and environment.

## Repository contents

This repository is Xception-only. The legacy exploratory notebook and trained
model binaries remain local and are intentionally excluded from Git.
