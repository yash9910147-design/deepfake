import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models"))).expanduser()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_ID = "manjilkarki/deepfake-and-real-images"
IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_FINE_TUNE_EPOCHS = 5
RANDOM_SEED = 42

CLASS_NAMES = ("Real", "Fake")
CLASS_TO_LABEL = {"real": 0, "fake": 1}

MODEL_SPECS = {
    "xception": {
        "display_name": "Xception",
        "filename": "xception_deepfake.h5",
        "preprocessing": "rescale_0_1",
        "positive_class": "Real",
    },
}
