from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import (
    CHECKPOINT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FINE_TUNE_EPOCHS,
    MODEL_DIR,
    MODEL_SPECS,
    RANDOM_SEED,
    RESULTS_DIR,
)
from data_loader import (
    build_tf_dataset,
    DatasetSplits,
    prepare_splits,
    print_split_summary,
    resolve_dataset_root,
)
from models import build_model, compile_model, is_compatible_xception_model


class BestValidationCheckpoint:
    def __new__(
        cls,
        tf,
        filepath: Path,
        state_path: Path,
        state: dict,
        initial_best: float = float("inf"),
    ):
        class Callback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best = initial_best
                self.epoch_started = None

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_started = time.monotonic()

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                if self.epoch_started is not None:
                    state["training_duration_seconds"] = float(
                        state.get("training_duration_seconds", 0.0)
                        + time.monotonic()
                        - self.epoch_started
                    )
                value = logs.get("val_loss")
                if value is not None and value < self.best:
                    self.best = float(value)
                    self.model.save(filepath)
                    state["best_val_loss"] = self.best
                    print(f"\nSaved best validation-loss checkpoint: {filepath}")
                _atomic_json_write(state_path, state)

        return Callback()


def _atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _valid_checkpoint(tf, path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except (OSError, ValueError, TypeError):
        return False
    return is_compatible_xception_model(model)


def _class_weights(samples) -> dict[int, float]:
    counts = Counter(sample.label for sample in samples)
    total = sum(counts.values())
    return {
        1: total / (2 * counts[0]),
        0: total / (2 * counts[1]),
    }


def _data_fingerprint(splits) -> str:
    import hashlib

    digest = hashlib.sha256()
    for split_name in ("train", "validation", "test"):
        digest.update(split_name.encode("utf-8"))
        for sample in sorted(getattr(splits, split_name), key=lambda item: item.content_hash):
            digest.update(sample.content_hash.encode("ascii"))
            digest.update(str(sample.label).encode("ascii"))
    return digest.hexdigest()


def _stratified_subset(samples, fraction: float, seed: int):
    if fraction >= 1.0:
        return samples
    labels = [sample.label for sample in samples]
    if min(Counter(labels).values()) < 2:
        raise ValueError(
            "Each class needs at least two samples to create a stratified training subset."
        )
    _, selected = train_test_split(
        list(samples),
        test_size=fraction,
        random_state=seed,
        stratify=labels,
    )
    selected_labels = {sample.label for sample in selected}
    if selected_labels != {0, 1}:
        raise ValueError(
            "The selected subset does not contain both Real and Fake classes; "
            "increase --subset-fraction."
        )
    return tuple(sorted(selected, key=lambda sample: str(sample.path)))


def _callbacks(tf, checkpoint_callback, backup_dir: Path, patience: int):
    backup_dir.mkdir(parents=True, exist_ok=True)
    return [
        checkpoint_callback,
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir)),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def _set_fine_tuning(backbone) -> None:
    if backbone is None:
        return
    import tensorflow as tf

    backbone.trainable = True
    unfrozen_layers = 30
    cutoff = max(0, len(backbone.layers) - unfrozen_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )


def _save_history(history_data: dict, model_key: str, phase: str, history) -> None:
    epochs = history.epoch
    for metric, values in history.history.items():
        history_data.setdefault(model_key, {}).setdefault(metric, [])
        if phase == "fine_tune" and history_data[model_key][metric]:
            history_data[model_key][metric].append(None)
        history_data[model_key][metric].extend(
            float(value) for value in values[: len(epochs)]
        )


def train_one_model(
    tf,
    model_key: str,
    splits,
    batch_size: int,
    epochs: int,
    fine_tune_epochs: int,
    seed: int,
    resume: bool,
    force: bool,
    history_data: dict,
) -> float | None:
    spec = MODEL_SPECS[model_key]
    model_path = MODEL_DIR / spec["filename"]
    if _valid_checkpoint(tf, model_path) and not (resume or force):
        print(f"{spec['display_name']}: verified checkpoint exists; skipping training.")
        return None
    if model_path.exists() and not _valid_checkpoint(tf, model_path):
        warnings.warn(
            f"Existing checkpoint is invalid or incompatible and will be replaced: {model_path}",
            RuntimeWarning,
        )

    model_checkpoint_dir = CHECKPOINT_DIR / model_key
    state_path = model_checkpoint_dir / "training_state.json"
    state = {}
    if resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    if resume and _valid_checkpoint(tf, model_path) and not state:
        raise ValueError(
            "The existing Xception model has no matching resumable training state. "
            "It can be used for inference; use --force to start a new training run."
        )
    run_config = {
        "architecture": model_key,
        "batch_size": batch_size,
        "seed": seed,
        "data_fingerprint": _data_fingerprint(splits),
        "image_size": [224, 224],
    }
    if resume and state.get("run_config") not in (None, run_config):
        raise ValueError(
            f"Cannot resume {spec['display_name']}: the dataset, batch size, seed, or "
            "architecture differs from the saved training state. Use --force to start over."
        )
    if state.get("fine_tune_complete") and not force:
        if _valid_checkpoint(tf, model_path):
            print(f"{spec['display_name']}: compatible completed training state; skipping.")
            return None
        warnings.warn(
            f"Completed training state has no valid model file and will be restarted: {model_path}",
            RuntimeWarning,
        )
        state = {}
    if force:
        for phase in ("head", "fine_tune"):
            backup = model_checkpoint_dir / f"{phase}_backup"
            if backup.exists():
                shutil.rmtree(backup)
        state = {}
        if state_path.exists():
            state_path.unlink()
        if model_path.exists():
            model_path.unlink()
    elif not resume and state_path.is_file() and not state.get("fine_tune_complete"):
        for phase in ("head", "fine_tune"):
            backup = model_checkpoint_dir / f"{phase}_backup"
            if backup.exists():
                shutil.rmtree(backup)
        state = {}
    model, backbone = build_model()
    if resume and _valid_checkpoint(tf, model_path):
        saved_model = tf.keras.models.load_model(model_path, compile=False)
        model.set_weights(saved_model.get_weights())

    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    state["run_config"] = run_config
    _atomic_json_write(state_path, state)
    initial_best = float(state.get("best_val_loss", float("inf")))
    checkpoint_callback = BestValidationCheckpoint(
        tf, model_path, state_path, state, initial_best
    )
    class_weights = _class_weights(splits.train)
    train_data = build_tf_dataset(
        splits.train, model_key, batch_size, training=True, seed=seed
    )
    validation_data = build_tf_dataset(
        splits.validation, model_key, batch_size, training=False, seed=seed
    )
    started = time.monotonic()

    if not state.get("head_complete"):
        compile_model(model, learning_rate=1e-4)
        head_history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            class_weight=class_weights,
            shuffle=False,
            callbacks=_callbacks(
                tf, checkpoint_callback, model_checkpoint_dir / "head_backup", patience=3
            ),
            verbose=1,
        )
        _save_history(history_data, model_key, "head", head_history)
        if not model_path.is_file():
            raise RuntimeError(f"{spec['display_name']} produced no best checkpoint.")
        state["head_complete"] = True
        state["best_val_loss"] = checkpoint_callback.best
        _atomic_json_write(state_path, state)
        best_model = tf.keras.models.load_model(model_path, compile=False)
        model.set_weights(best_model.get_weights())

    if backbone is not None and not state.get("fine_tune_complete"):
        _set_fine_tuning(backbone)
        compile_model(model, learning_rate=1e-5)
        fine_history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            shuffle=False,
            callbacks=_callbacks(
                tf,
                checkpoint_callback,
                model_checkpoint_dir / "fine_tune_backup",
                patience=2,
            ),
            verbose=1,
        )
        _save_history(history_data, model_key, "fine_tune", fine_history)
        state["fine_tune_complete"] = True
        state["best_val_loss"] = checkpoint_callback.best
        _atomic_json_write(state_path, state)
    elif backbone is None:
        state["fine_tune_complete"] = True
        _atomic_json_write(state_path, state)

    if not _valid_checkpoint(tf, model_path):
        raise RuntimeError(f"Saved model failed compatibility verification: {model_path}")
    duration = float(state.get("training_duration_seconds", time.monotonic() - started))
    print(f"{spec['display_name']} training time: {duration / 60:.2f} minutes")
    return duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Xception deepfake classifier.")
    parser.add_argument("--data-root", type=Path, help="Use a local dataset directory.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=DEFAULT_FINE_TUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=1.0,
        help="Fraction of train and validation data to use, stratified by class (default: 1.0).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--resume", action="store_true", help="Resume from compatible checkpoints.")
    parser.add_argument("--force", action="store_true", help="Start fresh and overwrite model files.")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.fine_tune_epochs < 1
        or args.batch_size < 1
        or not 0 < args.subset_fraction <= 1
    ):
        parser.error(
            "Epoch counts and batch size must be positive; --subset-fraction must be in (0, 1]."
        )
    if args.resume and args.force:
        parser.error("--resume and --force cannot be used together.")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as error:
            warnings.warn(f"Could not enable GPU memory growth: {error}", RuntimeWarning)
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU") or "none; using CPU")

    dataset_root = resolve_dataset_root(args.data_root)
    splits = prepare_splits(dataset_root, seed=args.seed)
    if args.subset_fraction < 1:
        splits = DatasetSplits(
            train=_stratified_subset(splits.train, args.subset_fraction, args.seed),
            validation=_stratified_subset(
                splits.validation, args.subset_fraction, args.seed
            ),
            test=splits.test,
            source=(
                f"{splits.source}; {args.subset_fraction:.0%} stratified train/validation "
                "subset (full test split)"
            ),
            skipped_corrupt=splits.skipped_corrupt,
            skipped_unclassified=splits.skipped_unclassified,
        )
    print(f"Dataset root: {dataset_root}")
    print_split_summary(splits)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = RESULTS_DIR / "training_history.json"
    history_data = (
        json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    )
    times_path = RESULTS_DIR / "training_times.json"
    training_times = (
        json.loads(times_path.read_text(encoding="utf-8")) if times_path.exists() else {}
    )

    for model_key in MODEL_SPECS:
        print(f"\n{'=' * 20} {MODEL_SPECS[model_key]['display_name']} {'=' * 20}")
        tf.keras.backend.clear_session()
        if args.force:
            history_data.pop(model_key, None)
            training_times.pop(model_key, None)
        duration = train_one_model(
            tf,
            model_key,
            splits,
            args.batch_size,
            args.epochs,
            args.fine_tune_epochs,
            args.seed,
            args.resume,
            args.force,
            history_data,
        )
        if duration is not None:
            training_times[model_key] = duration
            _atomic_json_write(times_path, training_times)
            _atomic_json_write(history_path, history_data)

    print("Training run finished. Evaluate with: python evaluate.py")


if __name__ == "__main__":
    main()
