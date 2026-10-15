# -*- coding: utf-8 -*-

"""

MRI MS research pipeline

- Patient-level audit/split (MS vs non-MS)

- NIfTI FLAIR preprocessing: Gaussian, median, CLAHE, Otsu brain ROI, K-means K=3

- MS/non-MS classification (2D slice classifier; patient-level aggregation)

- YOLO segmentation dataset export from LesionSeg-FLAIR masks and optional YOLO training

- NIfTI lesion prediction, ROI overlays, mean/std/histogram reports



Important:

1) Research prototype, not a clinical diagnostic tool.

2) Otsu/K-means are preprocessing/ROI aids, not ground-truth lesion labels.

3) Use only MS patients' expert LesionSeg-FLAIR masks for segmentation supervision.

4) Split by patient before extracting slices to avoid leakage.

"""

from __future__ import annotations



import argparse

import csv

import json

import re

import warnings

from pathlib import Path



import nibabel as nib

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cv2

from scipy import ndimage as ndi

from nibabel.processing import resample_from_to

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,

                             recall_score, f1_score, confusion_matrix, roc_auc_score)

from skimage.transform import resize

from skimage.morphology import remove_small_objects, binary_closing, disk

import torch

from torch import nn

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm



try:

    import config

except ImportError as exc:

    raise ImportError("Đặt ms_pipeline.py cùng thư mục với config.py.") from exc



MODALITIES = ("FLAIR", "T1", "T2")

SEED = int(getattr(config, "SEED", 42))

np.random.seed(SEED)

torch.manual_seed(SEED)





def cfg(name, default):

    return getattr(config, name, default)





def nii_files(root: Path):

    if not root.exists():

        return []

    return sorted(p for p in root.rglob("*") if p.is_file() and

                  (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz")))





def modality_from_name(name: str):

    u = name.upper().replace("-", "_")

    # "T1 ACP" is an acquisition variant, not the base T1 sequence.

    for mod in MODALITIES:

        if re.search(rf"(^|[^A-Z0-9]){mod}([^A-Z0-9]|$)", u):

            return mod

    for mod in MODALITIES:

        if mod in u:

            return mod

    return None





def is_mask(path: Path):

    u = path.name.upper()

    return "LESIONSEG" in u or "LESION_SEG" in u or "LESION-SEG" in u





def patient_id(path: Path, root: Path):

    try:

        rel = path.relative_to(root)

        if len(rel.parts) > 1:

            return rel.parts[0]

    except ValueError:

        pass

    m = re.search(r"patient[-_ ]?\d+", str(path), re.I)

    return m.group(0) if m else path.parent.name





def inventory(root: Path):

    if not root.exists():

        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {root}")

    result = {}

    for p in nii_files(root):

        # Exclude localizers and T1 acquisition/localizer variants from modality matching.

        if any(t in p.name.lower() for t in cfg("EXCLUDE_FILENAME_TOKENS", ("localizer",))):

            continue

        mod = modality_from_name(p.name)

        if not mod:

            continue

        pid = patient_id(p, root)

        kind = "mask" if is_mask(p) else "image"

        result.setdefault(pid, {}).setdefault(kind, {}).setdefault(mod, []).append(p)

    return result





def choose(paths):

    return sorted(paths, key=lambda p: (len(p.name), p.name.lower()))[0] if paths else None





def load_nii(path: Path):

    nii = nib.load(str(path))

    arr = np.asarray(nii.dataobj, dtype=np.float32)

    if arr.ndim == 2:

        arr = arr[:, :, None]

    if arr.ndim != 3:

        raise ValueError(f"NIfTI phải là 2D/3D, nhận shape={arr.shape}: {path}")

    return nii, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)





def align_mask(image_nii, mask_path: Path):

    mask_nii = nib.load(str(mask_path))

    if image_nii.shape[:3] == mask_nii.shape[:3] and np.allclose(image_nii.affine, mask_nii.affine, atol=1e-3):

        return np.asarray(mask_nii.dataobj, dtype=np.float32), False

    aligned = resample_from_to(mask_nii, (image_nii.shape[:3], image_nii.affine), order=0, mode="constant", cval=0)

    return np.asarray(aligned.dataobj, dtype=np.float32), True





def find_image_mask(records, pid, modality="FLAIR"):

    group = records.get(pid, {})

    return choose(group.get("image", {}).get(modality, [])), choose(group.get("mask", {}).get(modality, []))





def write_json(path: Path, data):

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")





def write_csv(path: Path, rows):

    path.parent.mkdir(parents=True, exist_ok=True)

    fields = sorted(set(k for r in rows for k in r)) if rows else ["status"]

    with path.open("w", newline="", encoding="utf-8-sig") as f:

        w = csv.DictWriter(f, fieldnames=fields)

        w.writeheader()

        w.writerows(rows)
        w.writerows(rows)


def write_image_unicode(path: Path, image):

    encoded_ok, encoded = cv2.imencode(path.suffix, image)

    if not encoded_ok:

        raise OSError(f"OpenCV không mã hóa được ảnh: {path}")

    encoded.tofile(str(path))



def audit(output: Path):

    ms_root = Path(cfg("MS_DATA_ROOT", cfg("DATA_ROOT", ".")))

    non_root = Path(cfg("NON_MS_DATA_ROOT", ""))

    records_by_class = {}

    rows = []

    summary = {"MS": {}, "non_MS": {}}

    for label, root in (("MS", ms_root), ("non_MS", non_root)):

        if not root.exists():

            warnings.warn(f"Thư mục {label} không tồn tại hoặc config chưa đúng: {root}")

            continue

        rec = inventory(root)

        records_by_class[label] = rec

        summary[label] = {"root": str(root), "patients": len(rec), "nifti_files": len(nii_files(root))}

        for pid, group in sorted(rec.items()):

            row = {"class": label, "patient": pid, "root": str(root)}

            for mod in MODALITIES:

                imgs, masks = group.get("image", {}).get(mod, []), group.get("mask", {}).get(mod, [])

                row[f"{mod}_image_count"] = len(imgs)

                row[f"{mod}_mask_count"] = len(masks)

                row[f"{mod}_image"] = str(choose(imgs) or "")

                row[f"{mod}_mask"] = str(choose(masks) or "")

                if choose(imgs):

                    try:

                        nii, arr = load_nii(choose(imgs))

                        row[f"{mod}_shape"] = str(arr.shape)

                        row[f"{mod}_spacing_mm"] = str(tuple(float(x) for x in nii.header.get_zooms()[:3]))

                        if choose(masks):

                            mask, resampled = align_mask(nii, choose(masks))

                            row[f"{mod}_mask_resampled"] = resampled

                            row[f"{mod}_mask_empty"] = bool(np.count_nonzero(mask > 0) == 0)

                            row[f"{mod}_mask_voxels"] = int(np.count_nonzero(mask > 0))

                    except Exception as e:

                        row[f"{mod}_error"] = str(e)

            rows.append(row)

    output.mkdir(parents=True, exist_ok=True)

    write_csv(output / "dataset_audit.csv", rows)

    write_json(output / "audit_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Audit CSV: {output / 'dataset_audit.csv'}")

    return records_by_class





def patient_splits(records_by_class, output: Path):

    all_rows = []

    for label in ("non_MS", "MS"):

        for pid in records_by_class.get(label, {}):

            all_rows.append({"patient": pid, "class": label, "label": 1 if label == "MS" else 0})

    if len({r["class"] for r in all_rows}) < 2:

        raise RuntimeError("Cần cấu hình đúng cả MS_DATA_ROOT và NON_MS_DATA_ROOT.")

    labels = [r["label"] for r in all_rows]

    # Stratify only when every class has enough patients for the requested split.

    strat = labels if min(labels.count(0), labels.count(1)) >= 3 else None

    train_val, test = train_test_split(all_rows, test_size=cfg("TEST_RATIO", .15),

                                       random_state=SEED, stratify=strat)

    tv_labels = [r["label"] for r in train_val]

    strat_tv = tv_labels if min(tv_labels.count(0), tv_labels.count(1)) >= 2 else None

    val_frac = cfg("VAL_RATIO", .15) / (cfg("TRAIN_RATIO", .70) + cfg("VAL_RATIO", .15))

    train, val = train_test_split(train_val, test_size=val_frac, random_state=SEED, stratify=strat_tv)

    split = {"train": train, "val": val, "test": test}

    write_json(output / "patient_split_classification.json", split)

    print("Patient-level split:", {k: len(v) for k, v in split.items()})

    return split





def normalize_volume(arr):

    arr = np.asarray(arr, dtype=np.float32)

    finite = np.isfinite(arr)

    nonzero = finite & (arr != 0)

    vals = arr[nonzero] if nonzero.any() else arr[finite]

    out = np.zeros_like(arr, dtype=np.float32)

    if vals.size == 0:

        return out

    lo, hi = np.percentile(vals, cfg("NORMALIZATION_PERCENTILES", (1, 99)))

    clipped = np.clip(arr, lo, hi)

    vals2 = clipped[nonzero] if nonzero.any() else clipped[finite]

    mean, std = float(vals2.mean()), float(vals2.std())

    if std < 1e-6:

        std = 1.0

    out[finite] = np.clip((clipped[finite] - mean) / std, -5, 5) / 5

    return out.astype(np.float32)





def scale_u8(img):

    a = np.asarray(img, dtype=np.float32)

    finite = np.isfinite(a)

    if not finite.any():

        return np.zeros(a.shape, np.uint8)

    lo, hi = np.percentile(a[finite], [1, 99])

    if hi <= lo:

        lo, hi = float(a[finite].min()), float(a[finite].max() + 1e-6)

    return np.uint8(np.clip((a - lo) / (hi - lo), 0, 1) * 255)





def preprocess_slice(slice2d, output_dir: Path | None = None, tag="slice"):

    """Return enhanced image, brain ROI, K=3 labels and Otsu threshold mask."""

    raw = np.nan_to_num(np.asarray(slice2d, dtype=np.float32))

    u8 = scale_u8(raw)

    if cfg("USE_GAUSSIAN", True):

        sigma = float(tuple(cfg("GAUSSIAN_SIGMAS", (0.5, 1.0)))[0])

        u8 = cv2.GaussianBlur(u8, (0, 0), sigmaX=sigma)

    if cfg("USE_MEDIAN", True):

        k = int(tuple(cfg("MEDIAN_KERNEL_SIZES", (3, 5)))[0])

        if k % 2 == 0:

            k += 1

        u8 = cv2.medianBlur(u8, k)

    if cfg("USE_CLAHE", True):

        clip = float(tuple(cfg("CLAHE_CLIP_LIMITS", (.01, .03)))[0])

        # OpenCV's CLAHE clipLimit is not the same scale as skimage's; use a conservative value.

        clahe = cv2.createCLAHE(clipLimit=max(1.0, clip * 100.0), tileGridSize=tuple(cfg("CLAHE_TILE_GRID_SIZE", (8, 8))))

        u8 = clahe.apply(u8)

    # Otsu is used to estimate foreground/brain ROI, not lesion truth.

    _, otsu = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    roi = otsu > 0

    roi = ndi.binary_fill_holes(roi)

    roi = remove_small_objects(roi.astype(bool), min_size=max(32, int(roi.size * .005)))

    labels = np.zeros(u8.shape, dtype=np.uint8)

    vals = u8[roi]

    if vals.size >= 3 and cfg("USE_KMEANS", True):

        km = KMeans(n_clusters=int(cfg("KMEANS_CLUSTERS", 3)),

                    random_state=int(cfg("KMEANS_RANDOM_STATE", SEED)),

                    n_init=int(cfg("KMEANS_N_INIT", 10)),

                    max_iter=int(cfg("KMEANS_MAX_ITER", 300)))

        lab = km.fit_predict(vals.reshape(-1, 1))

        # Re-label clusters by increasing intensity: 1=darkest, 3=brightest.

        order = np.argsort(km.cluster_centers_.ravel())

        mapping = {int(old): int(new + 1) for new, old in enumerate(order)}

        labels[roi] = np.array([mapping[int(v)] for v in lab], dtype=np.uint8)

    enhanced = u8.astype(np.float32) / 255.0

    if output_dir is not None:

        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 4, figsize=(14, 4))

        ax[0].imshow(raw.T, cmap="gray", origin="lower"); ax[0].set_title("Original")

        ax[1].imshow(enhanced.T, cmap="gray", origin="lower"); ax[1].set_title("Gaussian/Median/CLAHE")

        ax[2].imshow(roi.T, cmap="gray", origin="lower"); ax[2].set_title("Otsu brain ROI")

        ax[3].imshow(labels.T, cmap="viridis", origin="lower", vmin=0, vmax=3); ax[3].set_title("K-means K=3")

        for a in ax: a.axis("off")

        fig.tight_layout()

        fig.savefig(output_dir / f"{tag}_preprocess.png", dpi=150)

        plt.close(fig)

    return enhanced, roi.astype(bool), labels, (otsu > 0)





def make_yolo_input(slice2d):
    """
    Build a reproducible 3-channel pseudo-RGB input for YOLO:
    R = denoised/CLAHE-enhanced FLAIR, G = K-means K=3 labels,
    B = Otsu-derived foreground/brain ROI.
    These channels are engineered representations, NOT new independent data.
    """
    enhanced, brain_roi, kmeans_labels, _ = preprocess_slice(slice2d)
    ch0 = np.uint8(np.clip(enhanced, 0.0, 1.0) * 255.0)
    # labels 0=outside ROI, 1..3=dark-to-bright cluster
    ch1 = np.zeros(kmeans_labels.shape, dtype=np.uint8)
    for label in (1, 2, 3):
        ch1[kmeans_labels == label] = int(label * 255 / 3)
    ch2 = (brain_roi.astype(np.uint8) * 255)
    return np.stack([ch0, ch1, ch2], axis=-1), brain_roi


def save_roi_report(image2d, roi, output_dir: Path, tag, lesion_mask=None):

    output_dir.mkdir(parents=True, exist_ok=True)

    valid = np.asarray(roi, bool) & np.isfinite(image2d)

    vals = np.asarray(image2d, np.float32)[valid]

    if vals.size == 0:

        stats = {"pixel_count": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}

    else:

        stats = {"pixel_count": int(vals.size), "mean": float(vals.mean()), "std": float(vals.std()),

                 "min": float(vals.min()), "max": float(vals.max()), "median": float(np.median(vals))}

    write_json(output_dir / f"{tag}_roi_stats.json", stats)

    fig, ax = plt.subplots(figsize=(7, 4))

    if vals.size:

        ax.hist(vals, bins=int(cfg("HISTOGRAM_BINS", 64)))

    ax.set_title(f"ROI intensity histogram: {tag}")

    ax.set_xlabel("Intensity"); ax.set_ylabel("Pixel count")

    fig.tight_layout(); fig.savefig(output_dir / f"{tag}_roi_histogram.png", dpi=160); plt.close(fig)

    display = scale_u8(image2d)

    rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)

    # ROI boundary in yellow; lesion mask boundary in red.

    if np.any(valid):

        boundary = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

        rgb[boundary] = [255, 220, 0]

    if lesion_mask is not None and np.any(lesion_mask):

        boundary = cv2.morphologyEx(np.asarray(lesion_mask, np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

        rgb[boundary] = [255, 0, 0]

    cv2.imwrite(str(output_dir / f"{tag}_roi_overlay.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return stats





def resize2d(a, size, order=1):

    return resize(a, (size, size), order=order, preserve_range=True,

                  anti_aliasing=order > 0, mode="reflect").astype(np.float32)





class SliceClassificationDataset(Dataset):

    def __init__(self, rows, subset, records_by_class, size):

        self.items = []

        self.size = size

        for r in rows:

            label = int(r["label"])

            group_name = "MS" if label else "non_MS"

            group = records_by_class.get(group_name, {}).get(r["patient"], {})

            p = choose(group.get("image", {}).get(cfg("CLASSIFICATION_INPUT_MODALITY", "FLAIR"), []))

            if p:

                self.items.append((p, label, r["patient"], subset))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):

        p, label, pid, subset = self.items[idx]

        _, vol = load_nii(p)

        z = idx % max(vol.shape[2], 1)  # deterministic item index; overridden below by stored slice

        # Items are expanded into individual slices at init in normal use.

        return torch.from_numpy(resize2d(normalize_volume(vol)[:, :, z], self.size)[None]), torch.tensor(label, dtype=torch.long)





class SliceDatasetExpanded(Dataset):

    def __init__(self, rows, subset, records_by_class, size):

        self.items = []

        self.size = size

        mod = cfg("CLASSIFICATION_INPUT_MODALITY", "FLAIR")

        for r in rows:

            cname = "MS" if int(r["label"]) else "non_MS"

            group = records_by_class.get(cname, {}).get(r["patient"], {})

            p = choose(group.get("image", {}).get(mod, []))

            if not p: continue

            _, vol = load_nii(p)

            norm = normalize_volume(vol)

            for z in range(norm.shape[2]):

                sl = norm[:, :, z]

                if np.max(np.abs(sl)) < 1e-6: continue

                enhanced, _, _, _ = preprocess_slice(sl)

                self.items.append((resize2d(enhanced, size), int(r["label"]), r["patient"]))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):

        img, label, pid = self.items[i]

        return torch.from_numpy(img[None].astype(np.float32)), torch.tensor(label, dtype=torch.long)





class Classifier2D(nn.Module):

    def __init__(self, n_classes=2):

        super().__init__()

        layers, ch = [], 1

        for out_ch in (16, 32, 64, 128):

            layers += [nn.Conv2d(ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),

                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]

            ch = out_ch

        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))

        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.3), nn.Linear(ch, n_classes))

    def forward(self, x): return self.head(self.features(x))





def train_classifier(output: Path):

    ms_root = Path(cfg("MS_DATA_ROOT", cfg("DATA_ROOT", ".")))

    non_root = Path(cfg("NON_MS_DATA_ROOT", ""))

    records_by_class = {}

    for label, root in (("MS", ms_root), ("non_MS", non_root)):

        records_by_class[label] = inventory(root)

    split_file = output / "patient_split_classification.json"

    if split_file.exists():

        split = json.loads(split_file.read_text(encoding="utf-8"))

    else:

        split = patient_splits(records_by_class, output)

    size = int(cfg("CLASSIFICATION_IMAGE_SIZE", cfg("IMAGE_SIZE", 256)))

    train_ds = SliceDatasetExpanded(split["train"], "train", records_by_class, size)

    val_ds = SliceDatasetExpanded(split["val"], "val", records_by_class, size)

    test_ds = SliceDatasetExpanded(split["test"], "test", records_by_class, size)

    if not len(train_ds) or not len(val_ds):

        raise RuntimeError("Tập train/validation rỗng. Kiểm tra FLAIR và patient_split_classification.json.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier2D().to(device)

    counts = np.bincount([int(x[1]) for x in train_ds.items], minlength=2).astype(np.float32)

    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / 2, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights if cfg("USE_CLASS_WEIGHTS", True) else None)

    train_loader = DataLoader(train_ds, batch_size=int(cfg("CLASSIFICATION_BATCH_SIZE", 16)), shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=int(cfg("CLASSIFICATION_BATCH_SIZE", 16)), shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg("CLASSIFICATION_LEARNING_RATE", 1e-3)), weight_decay=1e-4)

    best_f1, history = -1, []

    epochs = int(cfg("CLASSIFICATION_EPOCHS", 50))

    for epoch in range(1, epochs + 1):

        model.train(); losses = []

        for x, y in train_loader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = criterion(model(x), y); loss.backward(); optimizer.step()

            losses.append(float(loss.item()))

        model.eval(); vl, ys, ps = [], [], []

        with torch.no_grad():

            for x, y in val_loader:

                x, y = x.to(device), y.to(device)

                logits = model(x); vl.append(float(criterion(logits, y).item()))

                ps.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist()); ys.extend(y.cpu().numpy().tolist())

        pred = (np.asarray(ps) >= .5).astype(int)

        f1 = f1_score(ys, pred, zero_division=0)

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(vl)),

               "val_accuracy": accuracy_score(ys, pred), "val_f1": f1}

        history.append(row)

        print(f"Epoch {epoch}/{epochs}: train_loss={row['train_loss']:.4f}, val_f1={f1:.4f}")

        if f1 > best_f1:

            best_f1 = f1

            torch.save({"model_state": model.state_dict(), "image_size": size,

                        "modality": cfg("CLASSIFICATION_INPUT_MODALITY", "FLAIR"),

                        "class_to_id": {"non_MS": 0, "MS": 1}}, output / "best_ms_classifier.pt")

    write_csv(output / "classification_history.csv", history)

    plot_training_history(history, output / "classification_training_curves.png")

    if len(test_ds):

        evaluate_classifier(output / "best_ms_classifier.pt", test_ds, output / "classification_test_metrics.json")

    print(f"Saved classifier: {output / 'best_ms_classifier.pt'}")





def plot_training_history(history, path):

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="train loss")

    ax.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], label="val loss")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=.25)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)





@torch.no_grad()

def evaluate_classifier(model_path, dataset, output_json):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)

    model = Classifier2D().to(device); model.load_state_dict(ckpt["model_state"]); model.eval()

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    ys, ps = [], []

    for x, y in loader:

        prob = torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()

        ys.extend(y.numpy().tolist()); ps.extend(prob.tolist())

    pred = (np.asarray(ps) >= .5).astype(int)

    result = {"accuracy": float(accuracy_score(ys, pred)),

              "balanced_accuracy": float(balanced_accuracy_score(ys, pred)),

              "precision": float(precision_score(ys, pred, zero_division=0)),

              "recall_sensitivity": float(recall_score(ys, pred, zero_division=0)),

              "specificity": float(confusion_matrix(ys, pred, labels=[0, 1])[0, 0] /

                                   max(confusion_matrix(ys, pred, labels=[0, 1])[0].sum(), 1)),

              "f1": float(f1_score(ys, pred, zero_division=0)),

              "confusion_matrix_labels_nonMS_MS": confusion_matrix(ys, pred, labels=[0, 1]).tolist()}

    if len(set(ys)) == 2: result["roc_auc"] = float(roc_auc_score(ys, ps))

    write_json(Path(output_json), result)

    print(json.dumps(result, indent=2))





def export_yolo_dataset(output: Path):

    """Export FLAIR slices + lesion polygons for Ultralytics YOLO segmentation."""

    root = Path(cfg("MS_DATA_ROOT", cfg("DATA_ROOT", ".")))

    records = inventory(root)

    # Use same patient split where possible; never split slices independently.

    split_path = output / "patient_split_yolo.json"

    pids = [pid for pid in records if find_image_mask(records, pid, "FLAIR")[0] and find_image_mask(records, pid, "FLAIR")[1]]

    if len(pids) < 3:

        raise RuntimeError("Cần ít nhất 3 bệnh nhân MS có cả FLAIR và LesionSeg-FLAIR để export YOLO.")

    trv, test = train_test_split(pids, test_size=float(cfg("TEST_RATIO", .15)), random_state=SEED)

    val_frac = float(cfg("VAL_RATIO", .15)) / (float(cfg("TRAIN_RATIO", .70)) + float(cfg("VAL_RATIO", .15)))

    train, val = train_test_split(trv, test_size=val_frac, random_state=SEED)

    split = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    write_json(split_path, split)

    base = output / "yolo_dataset"

    for s in ("train", "val", "test"):

        (base / "images" / s).mkdir(parents=True, exist_ok=True)

        (base / "labels" / s).mkdir(parents=True, exist_ok=True)

    size = int(cfg("YOLO_IMAGE_SIZE", 640))

    index, skipped = [], []

    for subset, patients in split.items():

        for pid in patients:

            img_path, mask_path = find_image_mask(records, pid, "FLAIR")

            try:

                nii, vol = load_nii(img_path)

                mask, resampled = align_mask(nii, mask_path)

                for z in range(vol.shape[2]):

                    im = vol[:, :, z]

                    ma = mask[:, :, z] > float(cfg("MASK_THRESHOLD", .5))

                    if np.max(np.abs(im)) < 1e-6 and not ma.any(): continue

                    enhanced, roi, _, _ = preprocess_slice(im)

                    im_u8 = np.uint8(np.clip(enhanced, 0, 1) * 255)

                    im_u8 = cv2.resize(im_u8, (size, size), interpolation=cv2.INTER_LINEAR)

                    ma_u8 = cv2.resize(ma.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)

                    # YOLO segmentation labels: class x1 y1 x2 y2 ... normalized polygon coordinates.

                    name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{pid}_z{z:04d}")

                    ip = base / "images" / subset / f"{name}.jpg"

                    lp = base / "labels" / subset / f"{name}.txt"

                    write_image_unicode(ip, im_u8)

                    contours, _ = cv2.findContours(ma_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    lines = []

                    for c in contours:

                        if cv2.contourArea(c) < 2 or len(c) < 3: continue

                        pts = c.reshape(-1, 2).astype(float)

                        coords = []

                        for x, y in pts:

                            coords.extend([min(max(x / size, 0), 1), min(max(y / size, 0), 1)])

                        if len(coords) >= 6:

                            lines.append("0 " + " ".join(f"{v:.6f}" for v in coords))

                    # Empty label file is intentional: a slice without lesions is a negative example.

                    lp.write_text("\n".join(lines), encoding="utf-8")

                    index.append({"subset": subset, "patient": pid, "slice": z, "image": str(ip),

                                  "label": str(lp), "lesion_pixels": int(ma_u8.sum()), "mask_resampled": resampled})

            except Exception as e:

                skipped.append({"patient": pid, "error": str(e)})

    yaml = base / "data.yaml"

    yaml.write_text(f"path: {base.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: lesion\n", encoding="utf-8")

    write_csv(base / "yolo_export_index.csv", index)
    write_json(base / "yolo_export_skipped.json", skipped)
    print(f"YOLO dataset: {base}")
    print(f"Images exported: {len(index)}; skipped: {len(skipped)}")
    if skipped:
        print(f"Chi tiết bệnh nhân bị bỏ qua: {base / 'yolo_export_skipped.json'}")
    print(f"Dataset YAML: {yaml}")
    if not index:
        raise RuntimeError(
            f"YOLO export không tạo được ảnh nào. Kiểm tra dữ liệu đầu vào và "
            f"{base / 'yolo_export_skipped.json'}."
        )





def train_yolo(output: Path):

    try:

        from ultralytics import YOLO

    except ImportError:

        raise ImportError("Thiếu ultralytics. Cài bằng: pip install ultralytics")

    data_yaml = output / "yolo_dataset" / "data.yaml"

    if not data_yaml.exists():

        raise FileNotFoundError("Chạy `python ms_pipeline.py yolo-export` trước.")

    model = YOLO(str(cfg("YOLO_MODEL", "yolo11n-seg.pt")))

    model.train(data=str(data_yaml), imgsz=int(cfg("YOLO_IMAGE_SIZE", 640)),

                epochs=int(cfg("YOLO_EPOCHS", 100)), batch=int(cfg("YOLO_BATCH_SIZE", 8)),

                workers=int(cfg("YOLO_WORKERS", 0)), device=cfg("YOLO_DEVICE", "cpu"),

                project=str(output / "yolo_runs"), name="ms_lesion_seg", seed=SEED)

    print("YOLO training finished. Inspect YOLO run directory for weights and metrics.")





def predict_unet_volume(image_path: Path, model_path: Path, outdir: Path, mask_path=None):

    """Compatibility route for existing best_unet2d.pt checkpoints."""

    from skimage.morphology import remove_small_objects, binary_closing, disk

    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)

    # Local U-Net class compatible with earlier checkpoint layout.

    class ConvBlock(nn.Module):

        def __init__(self, a, b):

            super().__init__()

            self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),

                                     nn.Conv2d(b,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

        def forward(self,x): return self.net(x)

    class UNet2D(nn.Module):

        def __init__(self, in_ch=1, out_ch=1, base=16):

            super().__init__()

            self.e1=ConvBlock(in_ch,base); self.e2=ConvBlock(base,base*2); self.e3=ConvBlock(base*2,base*4)

            self.e4=ConvBlock(base*4,base*8); self.pool=nn.MaxPool2d(2); self.bridge=ConvBlock(base*8,base*16)

            self.up4=nn.ConvTranspose2d(base*16,base*8,2,2); self.d4=ConvBlock(base*16,base*8)

            self.up3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=ConvBlock(base*8,base*4)

            self.up2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=ConvBlock(base*4,base*2)

            self.up1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=ConvBlock(base*2,base)

            self.out=nn.Conv2d(base,out_ch,1)

        def forward(self,x):

            e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))

            b=self.bridge(self.pool(e4)); d4=self.d4(torch.cat([self.up4(b),e4],1)); d3=self.d3(torch.cat([self.up3(d4),e3],1))

            d2=self.d2(torch.cat([self.up2(d3),e2],1)); d1=self.d1(torch.cat([self.up1(d2),e1],1))

            return self.out(d1)

    model=UNet2D().to(device); model.load_state_dict(ckpt["model_state"]); model.eval()

    nii, vol=load_nii(image_path); norm=normalize_volume(vol)

    pred=np.zeros(vol.shape,np.uint8); probs=np.zeros(vol.shape,np.float32)

    size=int(ckpt.get("image_size",cfg("IMAGE_SIZE",256))); threshold=float(ckpt.get("threshold",cfg("THRESHOLD",.5)))

    for z in tqdm(range(vol.shape[2]), desc="Predict"):

        _, _, _, _ = preprocess_slice(vol[:,:,z])

        small=resize2d(norm[:,:,z],size)

        with torch.no_grad(): prob=torch.sigmoid(model(torch.from_numpy(small[None,None]).to(device)))[0,0].cpu().numpy()

        prob=resize(prob,vol.shape[:2],order=1,preserve_range=True,anti_aliasing=False)

        m=prob>=threshold; m=binary_closing(m,footprint=disk(1)); m=remove_small_objects(m,min_size=int(cfg("MIN_COMPONENT_PIXELS",4)))

        pred[:,:,z]=m.astype(np.uint8); probs[:,:,z]=prob

        if m.any():

            save_roi_report(vol[:,:,z], np.ones_like(m,bool), outdir, f"slice_{z:03d}", m)

    n=nib.Nifti1Image(pred,nii.affine,nii.header); n.set_data_dtype(np.uint8); nib.save(n,str(outdir/"predicted_lesion_mask.nii.gz"))

    n=nib.Nifti1Image(probs,nii.affine,nii.header); n.set_data_dtype(np.float32); nib.save(n,str(outdir/"predicted_probability.nii.gz"))

    zoom=nii.header.get_zooms()[:3]; vox=int(pred.sum()); volume_mm3=vox*float(np.prod(zoom))

    result={"input":str(image_path),"predicted_lesion_voxels":vox,"predicted_lesion_volume_mm3":volume_mm3,

            "predicted_lesion_volume_cm3":volume_mm3/1000,"voxel_spacing_mm":[float(x) for x in zoom],

            "slices_with_lesion":int(np.count_nonzero(pred.sum(axis=(0,1)))),

            "warning":"Kết quả nghiên cứu, không thay thế chẩn đoán y khoa."}

    write_json(outdir/"prediction_summary.json",result); print(json.dumps(result,indent=2,ensure_ascii=False))

    if mask_path:

        truth,_=align_mask(nii,Path(mask_path)); truth=truth>0; p=pred>0

        tp=int((p&truth).sum()); fp=int((p&~truth).sum()); fn=int((~p&truth).sum())

        write_json(outdir/"reference_metrics.json",{"dice":2*tp/max(2*tp+fp+fn,1),"iou":tp/max(tp+fp+fn,1),"tp":tp,"fp":fp,"fn":fn})







def predict_yolo_volume(image_path: Path, model_path: Path, outdir: Path):

    """Run a trained Ultralytics YOLO-seg model slice-by-slice on a FLAIR NIfTI."""

    try:

        from ultralytics import YOLO

    except ImportError:

        raise ImportError("Thiếu ultralytics. Cài bằng: pip install ultralytics")

    outdir.mkdir(parents=True, exist_ok=True)

    nii, vol = load_nii(image_path)

    model = YOLO(str(model_path))

    h, w, depth = vol.shape

    pred_volume = np.zeros((h, w, depth), dtype=np.uint8)

    probability_volume = np.zeros((h, w, depth), dtype=np.float32)

    stats_rows = []

    for z in tqdm(range(depth), desc="YOLO segmentation"):

        enhanced, brain_roi, _, _ = preprocess_slice(vol[:, :, z])

        img = np.uint8(np.clip(enhanced, 0, 1) * 255)

        result = model.predict(source=img, imgsz=int(cfg("YOLO_IMAGE_SIZE", 640)),

                               conf=float(cfg("YOLO_CONFIDENCE", .25)),

                               iou=float(cfg("YOLO_IOU", .5)), verbose=False)[0]

        slice_mask = np.zeros((h, w), dtype=np.uint8)
        slice_confidence = np.zeros((h, w), dtype=np.float32)
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.detach().cpu().numpy()
            confidences = (
                result.boxes.conf.detach().cpu().numpy()
                if result.boxes is not None and result.boxes.conf is not None
                else np.ones(len(masks), dtype=np.float32)
            )
            for i, m in enumerate(masks):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                instance_mask = m >= 0.5
                slice_mask |= instance_mask.astype(np.uint8)
                conf = float(confidences[i]) if i < len(confidences) else 1.0
                slice_confidence[instance_mask] = np.maximum(slice_confidence[instance_mask], conf)
        # Constrain detections to the automatically estimated brain ROI.
        slice_mask = (slice_mask > 0) & brain_roi
        slice_mask = binary_closing(slice_mask, footprint=disk(1))
        slice_mask = remove_small_objects(slice_mask.astype(bool), min_size=int(cfg("MIN_COMPONENT_PIXELS", 4)))
        pred_volume[:, :, z] = slice_mask.astype(np.uint8)
        probability_volume[:, :, z] = slice_confidence * slice_mask.astype(np.float32)

        if slice_mask.any():

            stats = save_roi_report(vol[:, :, z], brain_roi, outdir, f"slice_{z:03d}", slice_mask)

            stats_rows.append({"slice": z, **stats, "lesion_pixels": int(slice_mask.sum())})

            display = scale_u8(vol[:, :, z])

            rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)

            boundary = cv2.morphologyEx(slice_mask.astype(np.uint8), cv2.MORPH_GRADIENT,

                                        np.ones((3, 3), np.uint8)) > 0

            rgb[boundary] = [255, 0, 0]

            cv2.imwrite(str(outdir / f"lesion_overlay_slice_{z:03d}.png"),

                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    pred_nii = nib.Nifti1Image(pred_volume, nii.affine, nii.header)

    pred_nii.set_data_dtype(np.uint8)

    nib.save(pred_nii, str(outdir / "yolo_predicted_lesion_mask.nii.gz"))

    prob_nii = nib.Nifti1Image(probability_volume, nii.affine, nii.header)

    prob_nii.set_data_dtype(np.float32)

    nib.save(prob_nii, str(outdir / "yolo_predicted_probability.nii.gz"))

    zooms = nii.header.get_zooms()[:3]

    voxels = int(pred_volume.sum())

    volume_mm3 = voxels * float(np.prod(zooms))

    write_csv(outdir / "lesion_roi_statistics.csv", stats_rows)

    summary = {"input_image": str(image_path), "model": str(model_path),

               "predicted_lesion_voxels": voxels, "predicted_lesion_volume_mm3": volume_mm3,

               "predicted_lesion_volume_cm3": volume_mm3 / 1000,

               "voxel_spacing_mm": [float(v) for v in zooms],

               "slices_with_lesion": int(np.count_nonzero(pred_volume.sum(axis=(0, 1)))),

               "warning": "Kết quả nghiên cứu, không thay thế chẩn đoán y khoa."}

    write_json(outdir / "yolo_prediction_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))





def main():

    parser=argparse.ArgumentParser(description="MRI MS classification + lesion segmentation research pipeline")

    parser.add_argument("command", choices=["audit","split","preprocess","train-classifier","evaluate-classifier",

                                             "yolo-export","train-yolo","predict-yolo","predict-unet"])

    parser.add_argument("--output", default=str(cfg("OUTPUT_ROOT","./brain_mri_ms_output")))

    parser.add_argument("--image", help="Đường dẫn FLAIR .nii/.nii.gz cần dự đoán")

    parser.add_argument("--model", help="Checkpoint .pt (mặc định best_unet2d.pt trong output)")

    parser.add_argument("--mask", help="Mask LesionSeg-FLAIR để đối chiếu tùy chọn")

    parser.add_argument("--patient", help="ID bệnh nhân cho lệnh preprocess, ví dụ Patient-01")

    parser.add_argument("--slice", type=int, default=0, help="Chỉ số lát cắt cho preprocess")

    args=parser.parse_args(); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)

    if args.command=="audit":

        audit(output)

    elif args.command=="split":

        records=audit(output); patient_splits(records,output)

    elif args.command=="preprocess":

        root=Path(cfg("MS_DATA_ROOT",cfg("DATA_ROOT",".")))

        records=inventory(root)

        if not args.patient: parser.error("preprocess cần --patient Patient-ID")

        p=choose(records.get(args.patient,{}).get("image",{}).get("FLAIR",[]))

        if not p: raise FileNotFoundError(f"Không tìm thấy FLAIR cho {args.patient}")

        _, vol=load_nii(p); z=max(0,min(args.slice,vol.shape[2]-1))

        enhanced,roi,labels,otsu=preprocess_slice(vol[:,:,z],output/"preprocessing",f"{args.patient}_z{z:03d}")

        stats=save_roi_report(enhanced,roi,output/"preprocessing",f"{args.patient}_z{z:03d}")

        print(json.dumps(stats,indent=2))

    elif args.command=="train-classifier":

        train_classifier(output)

    elif args.command=="evaluate-classifier":

        ms=Path(cfg("MS_DATA_ROOT",cfg("DATA_ROOT","."))); non=Path(cfg("NON_MS_DATA_ROOT",""))

        rec={"MS":inventory(ms),"non_MS":inventory(non)}

        split=json.loads((output/"patient_split_classification.json").read_text(encoding="utf-8"))

        ds=SliceDatasetExpanded(split["test"],"test",rec,int(cfg("CLASSIFICATION_IMAGE_SIZE",cfg("IMAGE_SIZE",256))))

        evaluate_classifier(Path(args.model) if args.model else output/"best_ms_classifier.pt",ds,output/"classification_test_metrics.json")

    elif args.command=="yolo-export":

        export_yolo_dataset(output)

    elif args.command=="train-yolo":

        train_yolo(output)

    elif args.command=="predict-yolo":

        if not args.image: parser.error("predict-yolo cần --image path/to/FLAIR.nii")

        model=Path(args.model) if args.model else output/"yolo_runs"/"ms_lesion_seg"/"weights"/"best.pt"

        predict_yolo_volume(Path(args.image),model,output/"yolo_prediction")

    elif args.command=="predict-unet":

        if not args.image: parser.error("predict-unet cần --image path/to/FLAIR.nii")

        model=Path(args.model) if args.model else output/"best_unet2d.pt"

        predict_unet_volume(Path(args.image),model,output/"prediction",args.mask)





if __name__=="__main__":

    main()
