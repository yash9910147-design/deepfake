from __future__ import annotations

import argparse
import hashlib
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from config import CLASS_NAMES, CLASS_TO_LABEL, DATASET_ID, IMAGE_SIZE, RANDOM_SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_ALIASES = {
    "real": CLASS_TO_LABEL["real"],
    "real_image": CLASS_TO_LABEL["real"],
    "real_images": CLASS_TO_LABEL["real"],
    "fake": CLASS_TO_LABEL["fake"],
    "fake_image": CLASS_TO_LABEL["fake"],
    "fake_images": CLASS_TO_LABEL["fake"],
    "deepfake": CLASS_TO_LABEL["fake"],
    "deepfakes": CLASS_TO_LABEL["fake"],
}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    split: str | None
    content_hash: str


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    test: tuple[Sample, ...]
    source: str
    skipped_corrupt: int
    skipped_unclassified: int


def _normalize_component(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _label_for_path(path: Path) -> int | None:
    for component in reversed(path.parts[:-1]):
        label = LABEL_ALIASES.get(_normalize_component(component))
        if label is not None:
            return label
    return None


def _split_for_path(path: Path) -> str | None:
    for component in path.parts[:-1]:
        split = SPLIT_ALIASES.get(_normalize_component(component))
        if split is not None:
            return split
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def discover_samples(dataset_root: str | Path) -> tuple[list[Sample], int, int]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    samples: list[Sample] = []
    skipped_corrupt = 0
    skipped_unclassified = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = _label_for_path(path)
        if label is None:
            skipped_unclassified += 1
            continue
        if not _inspect_image(path):
            skipped_corrupt += 1
            warnings.warn(f"Skipping unreadable image: {path}", RuntimeWarning)
            continue
        samples.append(
            Sample(
                path=path,
                label=label,
                split=_split_for_path(path),
                content_hash=_sha256(path),
            )
        )

    if not samples:
        raise ValueError(
            f"No readable images in recognized Real/Fake folders were found under {root}. "
            "Expected folders named Real/Real Images and Fake/Fake Images."
        )
    labels = {sample.label for sample in samples}
    if labels != {0, 1}:
        raise ValueError(
            "Both Real and Fake classes must be present. "
            f"Found: {[CLASS_NAMES[label] for label in sorted(labels)]}"
        )
    if skipped_unclassified:
        warnings.warn(
            f"Ignored {skipped_unclassified} image(s) outside recognized Real/Fake folders.",
            RuntimeWarning,
        )
    return samples, skipped_corrupt, skipped_unclassified


def _unique_content_groups(samples: Iterable[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.content_hash].append(sample)
    for content_hash, group in groups.items():
        if len({sample.label for sample in group}) > 1:
            paths = ", ".join(str(sample.path) for sample in group[:3])
            raise ValueError(
                "Identical image content was assigned conflicting Real/Fake labels: "
                f"{paths}"
            )
    return groups


def _has_both_classes(samples: Iterable[Sample]) -> bool:
    return {sample.label for sample in samples} == {0, 1}


def _existing_splits(samples: list[Sample]) -> DatasetSplits | None:
    by_declared_split = {
        split: [sample for sample in samples if sample.split == split]
        for split in ("train", "validation", "test")
    }
    if not all(_has_both_classes(group) for group in by_declared_split.values()):
        return None

    groups = _unique_content_groups(samples)
    selected: dict[str, Sample] = {}
    split_priority = {"train": 0, "validation": 1, "test": 2, None: 3}
    duplicate_count = 0
    for content_hash, group in groups.items():
        ordered = sorted(
            group,
            key=lambda sample: (
                split_priority[sample.split],
                str(sample.path),
            ),
        )
        selected[content_hash] = ordered[0]
        duplicate_count += len(group) - 1
    if duplicate_count:
        warnings.warn(
            f"Removed {duplicate_count} exact duplicate image file(s) to prevent split leakage.",
            RuntimeWarning,
        )

    result: dict[str, list[Sample]] = {"train": [], "validation": [], "test": []}
    for sample in selected.values():
        result[sample.split or "train"].append(sample)
    if not all(_has_both_classes(group) for group in result.values()):
        return None
    return DatasetSplits(
        train=tuple(sorted(result["train"], key=lambda sample: str(sample.path))),
        validation=tuple(sorted(result["validation"], key=lambda sample: str(sample.path))),
        test=tuple(sorted(result["test"], key=lambda sample: str(sample.path))),
        source="existing dataset splits",
        skipped_corrupt=0,
        skipped_unclassified=0,
    )


def _stratified_splits(
    samples: list[Sample], skipped_corrupt: int, skipped_unclassified: int, seed: int
) -> DatasetSplits:
    groups = _unique_content_groups(samples)
    representatives = [group[0] for group in groups.values()]
    group_ids = [sample.content_hash for sample in representatives]
    labels = [sample.label for sample in representatives]
    if min(Counter(labels).values()) < 3:
        raise ValueError("At least three unique images per class are needed to create three splits.")

    train_validation_ids, test_ids = train_test_split(
        group_ids,
        test_size=0.1,
        random_state=seed,
        stratify=labels,
    )
    label_by_hash = {sample.content_hash: sample.label for sample in representatives}
    train_validation_labels = [label_by_hash[content_hash] for content_hash in train_validation_ids]
    train_ids, validation_ids = train_test_split(
        train_validation_ids,
        test_size=1 / 9,
        random_state=seed,
        stratify=train_validation_labels,
    )
    sample_by_hash = {sample.content_hash: sample for sample in representatives}
    return DatasetSplits(
        train=tuple(sample_by_hash[value] for value in train_ids),
        validation=tuple(sample_by_hash[value] for value in validation_ids),
        test=tuple(sample_by_hash[value] for value in test_ids),
        source="reproducible stratified, content-grouped split",
        skipped_corrupt=skipped_corrupt,
        skipped_unclassified=skipped_unclassified,
    )


def prepare_splits(dataset_root: str | Path, seed: int = RANDOM_SEED) -> DatasetSplits:
    samples, skipped_corrupt, skipped_unclassified = discover_samples(dataset_root)
    existing = _existing_splits(samples)
    if existing is not None:
        return DatasetSplits(
            train=existing.train,
            validation=existing.validation,
            test=existing.test,
            source=existing.source,
            skipped_corrupt=skipped_corrupt,
            skipped_unclassified=skipped_unclassified,
        )
    return _stratified_splits(samples, skipped_corrupt, skipped_unclassified, seed)


def resolve_dataset_root(dataset_root: str | Path | None = None) -> Path:
    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Provided dataset directory does not exist: {root}")
        return root
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "kagglehub is not installed. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    try:
        path = kagglehub.dataset_download(DATASET_ID)
    except Exception as error:
        raise RuntimeError(
            f"Could not download Kaggle dataset {DATASET_ID!r}. Check network access and "
            "Kaggle authentication (KAGGLE_API_TOKEN, KAGGLE_USERNAME/KAGGLE_KEY, or "
            "~/.kaggle/kaggle.json)."
        ) from error
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"KaggleHub returned a missing dataset directory: {root}")
    print(f"Path to dataset files: {root}")
    return root


def print_split_summary(splits: DatasetSplits) -> None:
    print(f"Split source: {splits.source}")
    for split_name in ("train", "validation", "test"):
        samples = getattr(splits, split_name)
        counts = Counter(sample.label for sample in samples)
        print(
            f"{split_name:10s}: {len(samples):7d} images "
            f"(Real={counts[0]}, Fake={counts[1]})"
        )
    if splits.skipped_corrupt:
        print(f"Skipped corrupt images: {splits.skipped_corrupt}")
    if splits.skipped_unclassified:
        print(f"Skipped unclassified images: {splits.skipped_unclassified}")


def build_tf_dataset(
    samples: tuple[Sample, ...],
    architecture: str,
    batch_size: int,
    training: bool,
    seed: int = RANDOM_SEED,
):
    import tensorflow as tf

    paths = [str(sample.path) for sample in samples]
    labels = [sample.label for sample in samples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(samples),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def decode_and_preprocess(path, label):
        contents = tf.io.read_file(path)
        image = tf.io.decode_image(contents, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        image = tf.cast(image, tf.float32)
        if training:
            image = tf.image.random_flip_left_right(image, seed=seed)
            image = rotation(image, training=True)
            image = zoom(image, training=True)
        if architecture != "xception":
            raise ValueError(f"Unknown architecture: {architecture}")
        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        real_probability_target = 1.0 - tf.cast(label, tf.float32)
        return image, tf.reshape(real_probability_target, (1,))

    rotation = tf.keras.layers.RandomRotation(
        factor=0.035,
        fill_mode="reflect",
        seed=seed,
    )
    zoom = tf.keras.layers.RandomZoom(
        height_factor=0.08,
        width_factor=0.08,
        fill_mode="reflect",
        seed=seed + 1,
    )
    dataset = dataset.map(
        decode_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/inspect the deepfake dataset.")
    parser.add_argument("--data-root", type=Path, help="Use a local dataset directory.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    root = resolve_dataset_root(args.data_root)
    splits = prepare_splits(root, seed=args.seed)
    print(f"Dataset root: {root}")
    print_split_summary(splits)
    first_sample = splits.train[0]
    with Image.open(first_sample.path) as image:
        print(
            f"Sample: {first_sample.path} | label={CLASS_NAMES[first_sample.label]} "
            f"| source size={image.size} | target size={IMAGE_SIZE}"
        )


if __name__ == "__main__":
    main()
