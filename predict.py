from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import IMAGE_SIZE, MODEL_DIR, MODEL_SPECS
from models import is_compatible_xception_model


def _preprocess(image: Image.Image):
    import tensorflow as tf

    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = tf.keras.utils.img_to_array(image)
    tensor = tf.convert_to_tensor(array[None, ...], dtype=tf.float32)
    return tensor / 255.0


@lru_cache(maxsize=len(MODEL_SPECS))
def _load_model_cached(model_path: str, modified_ns: int):
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    if not is_compatible_xception_model(model):
        raise ValueError("incompatible Xception model architecture or input/output shape")
    return model


def predict_image(
    image: Image.Image,
    models_dir: str | Path = MODEL_DIR,
    selected_models: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict[str, str | float]], int]:
    """Run the available trained Xception model without starting training."""
    models_path = Path(models_dir).expanduser().resolve()
    model_keys = tuple(
        MODEL_SPECS.keys() if selected_models is None else selected_models
    )
    unknown_models = sorted(set(model_keys) - set(MODEL_SPECS))
    if unknown_models:
        raise ValueError(f"Unknown model selection: {', '.join(unknown_models)}")

    image = image.convert("RGB")
    results: list[dict[str, str | float]] = []
    fake_probabilities = []
    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        model_path = models_path / spec["filename"]
        if not model_path.is_file():
            results.append({
                "model": spec["display_name"],
                "label": "Unavailable",
                "predicted_probability": "-",
                "fake_probability": "-",
                "status": f"Model file missing: {model_path}",
            })
            continue

        try:
            model = _load_model_cached(
                str(model_path),
                model_path.stat().st_mtime_ns,
            )
            real_probability = float(
                model.predict(_preprocess(image), verbose=0)[0][0]
            )
            probability = 1.0 - real_probability
        except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
            results.append({
                "model": spec["display_name"],
                "label": "Unavailable",
                "predicted_probability": "-",
                "fake_probability": "-",
                "status": f"Could not load or use model: {error}",
            })
            continue

        label = "Fake" if probability >= 0.5 else "Real"
        fake_probabilities.append(probability)
        results.append({
            "model": spec["display_name"],
            "label": label,
            "predicted_probability": f"{max(probability, 1.0 - probability):.4f}",
            "fake_probability": f"{probability:.4f}",
            "status": "Ready",
        })

    return results, len(fake_probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an image is Real or Fake.")
    parser.add_argument("--image", required=True, type=Path, help="Path to an image.")
    parser.add_argument("--models-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image file does not exist: {args.image}")
    try:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        parser.error(f"Could not read image {args.image}: {error}")

    results, loaded_count = predict_image(image, args.models_dir)
    for result in results:
        print(
            f"{result['model']}: {result['label']} "
            f"(predicted-label probability: {result['predicted_probability']}; "
            f"Fake probability: {result['fake_probability']}; "
            f"{result['status']}; not a calibrated confidence)"
        )
    if loaded_count == 0:
        raise SystemExit("No usable trained models were found; train models before inference.")


if __name__ == "__main__":
    main()
