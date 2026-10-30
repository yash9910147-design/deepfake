from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from config import MODEL_DIR, MODEL_SPECS
from predict import predict_image

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
MAX_IMAGE_BYTES = 15 * 1024 * 1024

app = FastAPI(
    title="Deepfake Lens",
    description="Local image inference using the Xception deepfake classifier.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _model_available(model_key: str) -> bool:
    model_path = MODEL_DIR / MODEL_SPECS[model_key]["filename"]
    return model_path.is_file() and model_path.stat().st_size > 0


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
async def get_models() -> dict:
    return {
        "models": [
            {
                "key": model_key,
                "name": spec["display_name"],
                "available": _model_available(model_key),
            }
            for model_key, spec in MODEL_SPECS.items()
        ]
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    selected_models: str = Form(...),
) -> dict:
    try:
        model_keys = json.loads(selected_models)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400, detail="Selected model list must be valid JSON."
        ) from error
    if (
        not isinstance(model_keys, list)
        or not model_keys
        or any(not isinstance(key, str) or key not in MODEL_SPECS for key in model_keys)
    ):
        raise HTTPException(status_code=400, detail="Select one or more valid models.")

    content = await image.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 15 MB or smaller.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    try:
        with Image.open(BytesIO(content)) as uploaded_image:
            uploaded_image.verify()
        with Image.open(BytesIO(content)) as uploaded_image:
            image_copy = uploaded_image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail="The uploaded file is not a readable image."
        ) from error

    results, loaded_count = await run_in_threadpool(
        predict_image,
        image_copy,
        MODEL_DIR,
        model_keys,
    )
    return {
        "results": results,
        "loaded_count": loaded_count,
        "message": (
            f"{loaded_count} model(s) returned a prediction."
            if loaded_count
            else "No selected trained model is currently available."
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
