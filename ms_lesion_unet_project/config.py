from pathlib import Path

# Edit this path to match your dataset root.
DATA_ROOT = Path(r"D:\Xử lý ảnh\Brai MIR")
OUTPUT_ROOT = Path(r"D:\Xử lý ảnh\brain_mri_ms_output")

# First experiment: FLAIR image -> LesionSeg-FLAIR mask.
IMAGE_MODALITY = "FLAIR"
MASK_PREFIX = "LesionSeg"

# Patient-level split (must never split slices from one patient across sets).
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 2D U-Net training settings.
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # Windows-safe default
THRESHOLD = 0.50

# Preprocessing / postprocessing.
MIN_COMPONENT_PIXELS = 4
USE_KMEANS_POSTPROCESS = False  # Optional experiment only; may remove real small lesions.
KMEANS_CLUSTERS = 3
