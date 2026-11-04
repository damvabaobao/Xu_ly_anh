# -*- coding: utf-8 -*-
"""
Cấu hình project MRI MS:
- Phân đoạn tổn thương trên lát cắt 2D.
- Phân loại bệnh nhân MS / không MS.
- Tiền xử lý ảnh: Gaussian, Median, CLAHE, Otsu, K-Means.
- ROI và thống kê Mean, Standard Deviation, Histogram.

Các biến cũ được giữ lại để tương thích với code U-Net hiện tại.
"""

from pathlib import Path


# 1. ĐƯỜNG DẪN DỮ LIỆU
MS_DATA_ROOT = Path(r"D:\XU_LY_ANH\Brai MIR")
NON_MS_DATA_ROOT = Path(r"D:\XU_LY_ANH\non_brain")

# DATA_ROOT vẫn là thư mục MS vì pipeline phân đoạn cần ground-truth mask.
DATA_ROOT = MS_DATA_ROOT
OUTPUT_ROOT = Path(r"D:\XU_LY_ANH\brain_mri_ms_output")

AUDIT_DIR = OUTPUT_ROOT / "01_audit"
OVERLAY_DIR = OUTPUT_ROOT / "02_mask_overlay"
PREPROCESS_DIR = OUTPUT_ROOT / "03_preprocessing"
CLASSIFICATION_DIR = OUTPUT_ROOT / "04_classification"
YOLO_DIR = OUTPUT_ROOT / "05_yolo_segmentation"
REPORT_DIR = OUTPUT_ROOT / "06_reports"


# 2. MODALITY VÀ MASK
IMAGE_MODALITY = "FLAIR"
MASK_PREFIX = "LesionSeg"
AVAILABLE_MODALITIES = ("FLAIR", "T1", "T2")
EXCLUDE_FILENAME_TOKENS = ("acq-localizer", "localizer")
MASK_THRESHOLD = 0.5


# 3. CHIA DỮ LIỆU THEO BỆNH NHÂN
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
PATIENT_LEVEL_SPLIT = True


# 4. CHUẨN HÓA VÀ KÍCH THƯỚC ẢNH 2D
IMAGE_SIZE = 256
NORMALIZATION_METHOD = "robust_zscore"
NORMALIZATION_PERCENTILES = (1.0, 99.0)
NORMALIZATION_CLIP = 5.0
NORMALIZATION_OUTPUT_SCALE = 5.0
IMAGE_RESIZE_ORDER = 1
MASK_RESIZE_ORDER = 0
MASK_INTERPOLATION = "nearest"


# 5. LỌC NHIỄU VÀ TĂNG TƯƠNG PHẢN
USE_GAUSSIAN = True
GAUSSIAN_SIGMAS = (0.5, 1.0)

USE_MEDIAN = True
MEDIAN_KERNEL_SIZES = (3, 5)

USE_CLAHE = True
CLAHE_CLIP_LIMITS = (0.01, 0.03)
CLAHE_TILE_GRID_SIZE = (8, 8)


# 6. OTSU VÀ K-MEANS
USE_OTSU = True
USE_KMEANS = True
KMEANS_CLUSTERS = 3
KMEANS_RANDOM_STATE = SEED
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
KMEANS_FEATURES = ("intensity",)
KMEANS_USE_SPATIAL_FEATURES = False

# Hậu xử lý K-Means tùy chọn trong code cũ; mặc định tắt.
USE_KMEANS_POSTPROCESS = False


# 7. ROI VÀ THỐNG KÊ
# Không mặc định dùng hình chữ nhật cố định ở trung tâm ảnh.
ROI_MODE = "brain_mask"  # "brain_mask", "manual", hoặc "lesion_mask"
ROI_MARGIN_PIXELS = 0
SAVE_ROI_OVERLAY = True
SAVE_ROI_HISTOGRAM = True
HISTOGRAM_BINS = 64
ROI_STATISTICS = ("pixel_count", "mean", "std", "min", "max", "median")
REPORT_REFERENCE_LESION_STATS = True


# 8. 2D U-NET: giữ tên biến tương thích code hiện tại
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # Windows-safe default
THRESHOLD = 0.50
MIN_COMPONENT_PIXELS = 4


# 9. YOLO SEGMENTATION
# Dùng khi xây dựng module YOLO-Seg riêng; code U-Net hiện tại chưa đọc các biến này.
YOLO_MODEL = "yolo11n-seg.pt"
YOLO_IMAGE_SIZE = 640
YOLO_EPOCHS = 100
YOLO_BATCH_SIZE = 8
YOLO_WORKERS = 0
YOLO_DEVICE = "cpu"  # "auto", "cpu", hoặc 0 cho GPU đầu tiên
YOLO_CONFIDENCE = 0.25
YOLO_IOU = 0.50
YOLO_EXPORT_FORMAT = "polygon"
YOLO_SKIP_EMPTY_MASKS = False


# 10. PHÂN LOẠI MS / KHÔNG MS
CLASS_NAMES = ("non_ms", "ms")
CLASS_TO_ID = {"non_ms": 0, "ms": 1}
CLASSIFICATION_INPUT_MODALITY = "FLAIR"
CLASSIFICATION_IMAGE_SIZE = IMAGE_SIZE
CLASSIFICATION_BATCH_SIZE = 16
CLASSIFICATION_EPOCHS = 50
CLASSIFICATION_LEARNING_RATE = 1e-3
USE_CLASS_WEIGHTS = True
CLASSIFICATION_METRICS = (
    "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "f1", "roc_auc",
)


# 11. ĐÁNH GIÁ PHÂN ĐOẠN
SEGMENTATION_METRICS = ("dice", "iou", "precision", "recall", "specificity")
EVALUATE_ON_PATIENT_SPLIT = True


# 12. ĐẦU RA VÀ TÁI LẬP KẾT QUẢ
SAVE_PREPROCESS_VARIANTS = True
SAVE_TRAINING_CURVES = True
SAVE_PREDICTED_NIFTI_MASK = True
SAVE_PREDICTED_PROBABILITY = True
SAVE_SLICE_OVERLAYS = True
RESEARCH_ONLY_NOTICE = True
