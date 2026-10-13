"""
MS lesion segmentation starter pipeline:
  audit -> prepare patient split -> train 2D U-Net -> evaluate -> predict/overlay

Dataset assumption:
  one patient folder per subject, with NIfTI files containing names like:
  1-FLAIR.nii, 1-T1.nii, 1-T2.nii,
  1-LesionSeg-FLAIR.nii, 1-LesionSeg-T1.nii, 1-LesionSeg-T2.nii
The discovery code uses filename tokens and does not require exact numbering.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from nibabel.processing import resample_from_to
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config


# ------------------------- file discovery / geometry -------------------------

MODALITIES = ("FLAIR", "T1", "T2")


def nii_files(root: Path):
    return sorted([p for p in root.rglob("*") if p.is_file() and
                   (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))])


def modality_from_name(name: str):
    upper = name.upper()
    # Prefer explicit FLAIR token. LesionSeg is classified separately by is_mask().
    for modality in MODALITIES:
        if re.search(rf"(^|[^A-Z]){modality}([^A-Z]|$)", upper):
            return modality
    # Some names can contain e.g. LesionSegFLAIR without separators.
    for modality in MODALITIES:
        if modality in upper:
            return modality
    return None


def is_mask(path: Path):
    return "LESIONSEG" in path.name.upper() or "LESION-SEG" in path.name.upper() or "LESION_SEG" in path.name.upper()


def patient_id(path: Path, root: Path):
    # Dataset audit showed patient directories such as Patient-59.
    try:
        rel = path.relative_to(root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass
    m = re.search(r"patient[-_ ]?\d+", str(path), re.IGNORECASE)
    return m.group(0) if m else path.parent.name


def build_inventory(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    records = {}
    for p in nii_files(root):
        pid = patient_id(p, root)
        mod = modality_from_name(p.name)
        if not mod:
            continue
        kind = "mask" if is_mask(p) else "image"
        records.setdefault(pid, {}).setdefault(kind, {}).setdefault(mod, []).append(p)
    return records


def choose_one(paths):
    # If duplicates exist, prefer the shortest/least ambiguous filename, but report duplicates.
    return sorted(paths, key=lambda p: (len(p.name), p.name.lower()))[0] if paths else None


def load_nifti(path: Path):
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    # NIfTI arrays can be 2D or 3D; this pipeline expects 3D.
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {arr.shape}: {path}")
    return img, arr


def align_mask_to_image(image_nii, mask_nii):
    """Resample mask to image grid with nearest-neighbor interpolation if geometry differs."""
    same_shape = image_nii.shape[:3] == mask_nii.shape[:3]
    same_affine = np.allclose(image_nii.affine, mask_nii.affine, atol=1e-3)
    if same_shape and same_affine:
        return np.asarray(mask_nii.dataobj, dtype=np.float32), False
    aligned = resample_from_to(mask_nii, (image_nii.shape[:3], image_nii.affine),
                               order=0, mode="constant", cval=0)
    return np.asarray(aligned.dataobj, dtype=np.float32), True


def resolve_pair(records, pid, modality="FLAIR"):
    group = records.get(pid, {})
    img = choose_one(group.get("image", {}).get(modality, []))
    mask = choose_one(group.get("mask", {}).get(modality, []))
    return img, mask


def audit(root: Path, output: Path):
    records = build_inventory(root)
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for pid in sorted(records):
        row = {"patient": pid}
        for mod in MODALITIES:
            imgs = records[pid].get("image", {}).get(mod, [])
            masks = records[pid].get("mask", {}).get(mod, [])
            row[f"{mod}_image_count"] = len(imgs)
            row[f"{mod}_mask_count"] = len(masks)
            row[f"{mod}_image"] = str(choose_one(imgs) or "")
            row[f"{mod}_mask"] = str(choose_one(masks) or "")
            if len(imgs) > 1 or len(masks) > 1:
                row[f"{mod}_duplicate_warning"] = True
            else:
                row[f"{mod}_duplicate_warning"] = False

            image_path, mask_path = choose_one(imgs), choose_one(masks)
            if image_path and mask_path:
                try:
                    image_nii, image = load_nifti(image_path)
                    mask_nii, mask = load_nifti(mask_path)
                    aligned_mask, resampled = align_mask_to_image(image_nii, mask_nii)
                    unique = np.unique(aligned_mask)
                    spacing = image_nii.header.get_zooms()[:3]
                    row[f"{mod}_shape"] = str(image.shape)
                    row[f"{mod}_mask_shape_original"] = str(mask.shape)
                    row[f"{mod}_mask_resampled"] = resampled
                    row[f"{mod}_spacing_mm"] = str(tuple(float(x) for x in spacing))
                    row[f"{mod}_mask_unique_values"] = str(unique[:20].tolist())
                    row[f"{mod}_mask_empty"] = bool(np.count_nonzero(aligned_mask > 0) == 0)
                    row[f"{mod}_mask_voxels"] = int(np.count_nonzero(aligned_mask > 0))
                    row[f"{mod}_mask_volume_mm3"] = float(np.count_nonzero(aligned_mask > 0) * np.prod(spacing))
                except Exception as exc:
                    row[f"{mod}_error"] = str(exc)
        report.append(row)

    csv_path = output / "dataset_audit.csv"
    keys = sorted(set(k for row in report for k in row))
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report)

    # Summary for the selected FLAIR pair.
    summary = {
        "data_root": str(root),
        "patients_detected": len(records),
        "nifti_files": len(nii_files(root)),
        "selected_modality": "FLAIR",
        "patients_with_flair_image_and_mask": sum(
            bool(resolve_pair(records, pid, "FLAIR")[0] and resolve_pair(records, pid, "FLAIR")[1])
            for pid in records
        ),
        "patients_missing_flair_pair": [],
        "empty_flair_masks": [],
        "duplicate_flair_pairs": [],
    }
    for pid in sorted(records):
        im, ma = resolve_pair(records, pid, "FLAIR")
        if not im or not ma:
            summary["patients_missing_flair_pair"].append(pid)
        if len(records[pid].get("image", {}).get("FLAIR", [])) > 1 or len(records[pid].get("mask", {}).get("FLAIR", [])) > 1:
            summary["duplicate_flair_pairs"].append(pid)
        if im and ma:
            try:
                ii, _ = load_nifti(im)
                mi = nib.load(str(ma))
                mm, _ = align_mask_to_image(ii, mi)
                if np.count_nonzero(mm > 0) == 0:
                    summary["empty_flair_masks"].append(pid)
            except Exception:
                pass

    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed report: {csv_path}")
    return records


# ------------------------- patient-level split and slices -------------------------

def make_splits(records, output: Path):
    patients = []
    for pid in sorted(records):
        img, mask = resolve_pair(records, pid, config.IMAGE_MODALITY)
        if img and mask:
            patients.append(pid)
    if len(patients) < 5:
        raise RuntimeError(
            f"Only {len(patients)} patient(s) have both {config.IMAGE_MODALITY} image and mask. "
            "Check dataset filenames and dataset_audit.csv before training."
        )
    train_val, test = train_test_split(
        patients, test_size=config.TEST_RATIO, random_state=config.SEED
    )
    val_fraction_of_train_val = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_fraction_of_train_val, random_state=config.SEED
    )
    split = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "patient_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print("Patient split:", {k: len(v) for k, v in split.items()})
    print(f"Saved: {output / 'patient_split.json'}")
    return split


def normalize_volume(volume):
    """Robust per-volume z-score using nonzero voxels; return float32."""
    arr = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(arr)
    nonzero = finite & (arr != 0)
    values = arr[nonzero] if np.any(nonzero) else arr[finite]
    out = np.zeros_like(arr, dtype=np.float32)
    if values.size == 0:
        return out
    lo, hi = np.percentile(values, [1, 99])
    clipped = np.clip(arr, lo, hi)
    vals2 = clipped[nonzero] if np.any(nonzero) else clipped[finite]
    mean = float(np.mean(vals2))
    std = float(np.std(vals2))
    if std < 1e-6:
        std = 1.0
    out[finite] = (clipped[finite] - mean) / std
    # Clip standardized outliers for stable training.
    out = np.clip(out, -5, 5) / 5.0
    return out.astype(np.float32)


def resize_2d(arr, size, order):
    return resize(arr, (size, size), order=order, preserve_range=True,
                  anti_aliasing=(order > 0), mode="reflect").astype(np.float32)


def prepare_dataset(root: Path, output: Path):
    records = build_inventory(root)
    split_path = output / "patient_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else make_splits(records, output)
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    index = []
    skipped = []

    for subset, pids in split.items():
        for pid in pids:
            image_path, mask_path = resolve_pair(records, pid, config.IMAGE_MODALITY)
            if not image_path or not mask_path:
                skipped.append({"patient": pid, "reason": "missing image/mask pair"})
                continue
            try:
                image_nii, image = load_nifti(image_path)
                mask_nii = nib.load(str(mask_path))
                mask, resampled = align_mask_to_image(image_nii, mask_nii)
                image = normalize_volume(image)
                mask = (mask > 0).astype(np.uint8)

                # Image and mask are now on the same voxel grid. Keep all slices, including
                # negative slices, so the model learns background. Exclude only all-zero image slices.
                for z in range(image.shape[2]):
                    im_slice = image[:, :, z]
                    ma_slice = mask[:, :, z]
                    if not np.isfinite(im_slice).all():
                        im_slice = np.nan_to_num(im_slice)
                    if np.max(np.abs(im_slice)) < 1e-6 and np.count_nonzero(ma_slice) == 0:
                        continue
                    im_resized = resize_2d(im_slice, config.IMAGE_SIZE, order=1)
                    ma_resized = (resize_2d(ma_slice, config.IMAGE_SIZE, order=0) > 0.5).astype(np.uint8)
                    # Save per-slice .npz to keep RAM usage modest.
                    out_file = prepared / f"{subset}_{re.sub(r'[^A-Za-z0-9_-]', '_', pid)}_z{z:03d}.npz"
                    np.savez_compressed(out_file, image=im_resized[None, ...], mask=ma_resized[None, ...])
                    index.append({
                        "subset": subset, "patient": pid, "z": z, "file": str(out_file),
                        "image_path": str(image_path), "mask_path": str(mask_path),
                        "mask_resampled": bool(resampled),
                        "positive_pixels": int(np.count_nonzero(ma_resized)),
                    })
            except Exception as exc:
                skipped.append({"patient": pid, "reason": str(exc)})

    with (output / "slice_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(index[0].keys()) if index else ["subset", "patient", "z", "file"])
        writer.writeheader()
        writer.writerows(index)
    (output / "prepare_skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    counts = {s: sum(1 for r in index if r["subset"] == s) for s in ("train", "val", "test")}
    print("Prepared slice counts:", counts)
    print("Skipped patients/files:", len(skipped))
    print("Prepared data:", prepared)
    if not index:
        raise RuntimeError("No slices prepared. Inspect audit report and filename matching.")
    return index


# ------------------------- U-Net -------------------------

class SliceDataset(Dataset):
    def __init__(self, index_csv, subset):
        with open(index_csv, newline="", encoding="utf-8-sig") as f:
            self.rows = [r for r in csv.DictReader(f) if r["subset"] == subset]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        data = np.load(row["file"])
        x = data["image"].astype(np.float32)
        y = data["mask"].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bridge(self.pool(e4))
        d4 = self.d4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)  # logits


def dice_loss_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def combined_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss_logits(logits, target)


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = fn = tn = 0
    loss_sum = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(combined_loss(logits, y).item())
        n += 1
        pred = torch.sigmoid(logits) >= threshold
        truth = y >= 0.5
        tp += int((pred & truth).sum().item())
        fp += int((pred & ~truth).sum().item())
        fn += int((~pred & truth).sum().item())
        tn += int((~pred & ~truth).sum().item())
    eps = 1e-8
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return {
        "loss": loss_sum / max(n, 1), "dice": dice, "iou": iou,
        "precision": precision, "recall": recall, "specificity": specificity,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def train(output: Path):
    index_csv = output / "slice_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError("Run prepare first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    train_ds = SliceDataset(index_csv, "train")
    val_ds = SliceDataset(index_csv, "val")
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError("Train/validation slice set is empty. Check preparation output.")
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"))

    model = UNet2D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    best_dice = -1.0
    history = []
    weights_path = output / "best_unet2d.pt"

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running = 0.0
        batches = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = combined_loss(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1

        val_metrics = evaluate_loader(model, val_loader, device, config.THRESHOLD)
        scheduler.step(val_metrics["dice"])
        row = {"epoch": epoch, "train_loss": running / max(batches, 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
              f"val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f}")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save({
                "model_state": model.state_dict(),
                "image_size": config.IMAGE_SIZE,
                "threshold": config.THRESHOLD,
                "modality": config.IMAGE_MODALITY,
                "best_val_dice": best_dice,
            }, weights_path)

    with (output / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output / "training_curves.png")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Saved model: {weights_path}")


def plot_history(history, output_path):
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [r["train_loss"] for r in history], label="Train loss")
    ax.plot(epochs, [r["val_loss"] for r in history], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ------------------------- prediction / post-processing / physical area -------------------------

def postprocess_mask(probability, threshold, min_pixels=4, use_kmeans=False, kmeans_clusters=3):
    raw = probability >= threshold
    raw = binary_closing(raw, footprint=disk(1))
    raw = remove_small_objects(raw.astype(bool), min_size=min_pixels)
    if use_kmeans:
        # K-means is optional and experimental. Cluster (intensity, probability) only inside
        # candidate pixels; keep cluster with highest mean probability. It is not a diagnostic rule.
        from sklearn.cluster import KMeans
        coords = np.argwhere(raw)
        if len(coords) >= max(kmeans_clusters, 10):
            feats = np.column_stack([
                probability[raw],
                np.linalg.norm(coords / np.maximum(np.array(raw.shape) - 1, 1), axis=1)
            ])
            labels = KMeans(n_clusters=kmeans_clusters, random_state=config.SEED, n_init=10).fit_predict(feats)
            means = [float(probability[raw][labels == k].mean()) if np.any(labels == k) else -1
                     for k in range(kmeans_clusters)]
            chosen = int(np.argmax(means))
            keep_flat = labels == chosen
            new_mask = np.zeros_like(raw, dtype=bool)
            new_mask[tuple(coords[keep_flat].T)] = True
            # Use as a conservative experimental filter, not the default.
            raw = new_mask
    return raw.astype(bool)


def draw_overlay(image_slice, mask, output_path, title="MS lesion prediction"):
    img = np.asarray(image_slice, dtype=np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max() + 1e-6)
    display = np.clip((img - lo) / (hi - lo), 0, 1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(display.T, cmap="gray", origin="lower")
    # Contour coordinates are pixel indices; transpose to match display orientation.
    ax.contour(mask.T.astype(float), levels=[0.5], colors=["red"], linewidths=1.0)
    ax.imshow(np.ma.masked_where(~mask.T, mask.T), cmap="autumn", alpha=0.35, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def predict_volume(image_path: Path, model_path: Path, output_dir: Path,
                   mask_path: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = UNet2D().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    size = int(checkpoint.get("image_size", config.IMAGE_SIZE))
    threshold = float(checkpoint.get("threshold", config.THRESHOLD))

    image_nii, image = load_nifti(image_path)
    norm = normalize_volume(image)
    pred_volume = np.zeros(image.shape, dtype=np.uint8)
    prob_volume = np.zeros(image.shape, dtype=np.float32)

    with torch.no_grad():
        for z in tqdm(range(image.shape[2]), desc="Predicting slices"):
            resized = resize_2d(norm[:, :, z], size, order=1)
            x = torch.from_numpy(resized[None, None, ...]).to(device)
            probability_small = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            probability = resize(probability_small, image.shape[:2], order=1,
                                 preserve_range=True, anti_aliasing=False, mode="reflect")
            probability = np.clip(probability, 0, 1)
            mask = postprocess_mask(
                probability, threshold,
                min_pixels=config.MIN_COMPONENT_PIXELS,
                use_kmeans=config.USE_KMEANS_POSTPROCESS,
                kmeans_clusters=config.KMEANS_CLUSTERS
            )
            pred_volume[:, :, z] = mask.astype(np.uint8)
            prob_volume[:, :, z] = probability.astype(np.float32)
            if np.any(mask):
                overlay_path = output_dir / f"overlay_slice_{z:03d}.png"
                draw_overlay(image[:, :, z], mask, overlay_path,
                             title=f"Predicted MS lesions | slice {z}")

    pred_nii = nib.Nifti1Image(pred_volume, image_nii.affine, image_nii.header)
    pred_nii.set_data_dtype(np.uint8)
    nib.save(pred_nii, str(output_dir / "predicted_lesion_mask.nii.gz"))
    prob_nii = nib.Nifti1Image(prob_volume, image_nii.affine, image_nii.header)
    prob_nii.set_data_dtype(np.float32)
    nib.save(prob_nii, str(output_dir / "predicted_probability.nii.gz"))

    zooms = image_nii.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(zooms))
    voxel_count = int(pred_volume.sum())
    total_volume_mm3 = voxel_count * voxel_volume_mm3
    slice_rows = []
    for z in range(image.shape[2]):
        area_mm2 = int(pred_volume[:, :, z].sum()) * float(zooms[0]) * float(zooms[1])
        slice_rows.append({"slice": z, "lesion_pixels": int(pred_volume[:, :, z].sum()),
                           "area_mm2": area_mm2})
    with (output_dir / "slice_areas.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["slice", "lesion_pixels", "area_mm2"])
        writer.writeheader()
        writer.writerows(slice_rows)

    result = {
        "input_image": str(image_path),
        "model": str(model_path),
        "shape_voxels": list(image.shape),
        "voxel_spacing_mm": [float(v) for v in zooms],
        "predicted_lesion_voxels": voxel_count,
        "predicted_lesion_volume_mm3": total_volume_mm3,
        "predicted_lesion_volume_cm3": total_volume_mm3 / 1000.0,
        "slices_with_predicted_lesion": int(np.count_nonzero(pred_volume.sum(axis=(0, 1)))),
        "threshold": threshold,
        "note": "Research output only; prediction is not a clinical diagnosis."
    }
    (output_dir / "prediction_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Overlays (only slices with predicted lesions): {output_dir}")
    if mask_path:
        evaluate_single_volume(image_path, mask_path, output_dir / "reference_comparison.json", pred_volume)
    return result


def evaluate_single_volume(image_path, mask_path, output_json, pred_volume=None):
    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    mask, _ = align_mask_to_image(image_nii, mask_nii)
    truth = mask > 0
    if pred_volume is None:
        raise ValueError("Prediction volume is required for comparison.")
    pred = pred_volume > 0
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    result = {"dice": dice, "iou": iou, "tp_voxels": tp, "fp_voxels": fp, "fn_voxels": fn}
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Reference-mask comparison:", json.dumps(result, indent=2))
    return result


def evaluate_test_set(output: Path):
    index_csv = output / "slice_index.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(output / "best_unet2d.pt", map_location=device)
    model = UNet2D().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    ds = SliceDataset(index_csv, "test")
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    metrics = evaluate_loader(model, loader, device, float(checkpoint.get("threshold", 0.5)))
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Held-out test metrics:", json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="MS lesion segmentation pipeline")
    parser.add_argument("command", choices=["audit", "prepare", "train", "evaluate", "predict"])
    parser.add_argument("--data", default=str(config.DATA_ROOT), help="Dataset root directory")
    parser.add_argument("--output", default=str(config.OUTPUT_ROOT), help="Output directory")
    parser.add_argument("--image", help="Input FLAIR NIfTI path for prediction")
    parser.add_argument("--model", help="Trained model path; defaults to output/best_unet2d.pt")
    parser.add_argument("--mask", help="Optional reference lesion mask NIfTI for prediction evaluation")
    args = parser.parse_args()
    root, output = Path(args.data), Path(args.output)

    if args.command == "audit":
        audit(root, output)
    elif args.command == "prepare":
        audit(root, output)
        make_splits(build_inventory(root), output)
        prepare_dataset(root, output)
    elif args.command == "train":
        train(output)
    elif args.command == "evaluate":
        evaluate_test_set(output)
    elif args.command == "predict":
        if not args.image:
            parser.error("predict requires --image path/to/FLAIR.nii")
        model = Path(args.model) if args.model else output / "best_unet2d.pt"
        predict_volume(Path(args.image), model, output / "prediction", Path(args.mask) if args.mask else None)


if __name__ == "__main__":
    main()
