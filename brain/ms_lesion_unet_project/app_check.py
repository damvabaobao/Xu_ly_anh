# -*- coding: utf-8 -*-
"""Tkinter interface for FLAIR lesion segmentation and MS classification."""

from __future__ import annotations

import queue
import re
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import nibabel as nib
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import brain.ms_lesion_unet_project.config as config
import brain.ms_lesion_unet_project.ms_pipeline as pipeline


def _nifti_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        name = name[:-7]
    elif name.lower().endswith(".nii"):
        name = name[:-4]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def run_prediction(
    image_path: Path,
    yolo_path: Path,
    classifier_path: Path,
    output_root: Path,
) -> dict:
    """Run both trained models and prepare display-ready volume measurements."""
    for label, path in (
        ("Ảnh FLAIR", image_path),
        ("Model YOLO", yolo_path),
        ("Model CNN", classifier_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {label}: {path}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / f"{_nifti_stem(image_path)}_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    nii, volume = pipeline.load_nii(image_path)
    pipeline.predict_yolo_volume(image_path, yolo_path, output_dir)

    mask_path = output_dir / "yolo_predicted_lesion_mask.nii.gz"
    if not mask_path.is_file():
        raise FileNotFoundError(f"YOLO không tạo được mặt nạ dự đoán: {mask_path}")
    lesion_volume = np.asarray(nib.load(str(mask_path)).dataobj) > 0
    if lesion_volume.shape != volume.shape:
        raise ValueError(
            "Kích thước mask YOLO không khớp với ảnh FLAIR: "
            f"{lesion_volume.shape} != {volume.shape}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(classifier_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            "Checkpoint CNN không đúng định dạng. Cần best_ms_classifier.pt "
            "được tạo bởi ms_pipeline.py."
        )

    class_to_id = checkpoint.get("class_to_id", {"non_MS": 0, "MS": 1})
    if class_to_id.get("non_MS") != 0 or class_to_id.get("MS") != 1:
        raise ValueError("Checkpoint CNN không có thứ tự lớp non_MS=0, MS=1.")

    model = pipeline.Classifier2D().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_size = int(
        checkpoint.get(
            "image_size",
            pipeline.cfg("CLASSIFICATION_IMAGE_SIZE", pipeline.cfg("IMAGE_SIZE", 256)),
        )
    )
    normalized = pipeline.normalize_volume(volume)
    slice_probabilities = []
    with torch.no_grad():
        for z in range(normalized.shape[2]):
            normalized_slice = normalized[:, :, z]
            if np.max(np.abs(normalized_slice)) < 1e-6:
                continue
            enhanced, _, _, _ = pipeline.preprocess_slice(normalized_slice)
            resized = pipeline.resize2d(enhanced, image_size)
            tensor = torch.from_numpy(resized[None, None]).to(device)
            probability = torch.softmax(model(tensor), dim=1)[0, 1].item()
            slice_probabilities.append(float(probability))

    if not slice_probabilities:
        raise ValueError("Không có lát FLAIR hợp lệ để CNN phân loại.")

    mean_ms_probability = float(np.mean(slice_probabilities))
    has_ms_sign = mean_ms_probability >= 0.5
    lesion_counts = lesion_volume.sum(axis=(0, 1))
    slice_index = int(np.argmax(lesion_counts))
    if int(lesion_counts[slice_index]) == 0:
        slice_index = volume.shape[2] // 2

    raw_slice = volume[:, :, slice_index]
    lesion_slice = lesion_volume[:, :, slice_index]
    preprocessing_images = pipeline.preprocess_slice_stages(raw_slice)
    _, brain_roi, _, _ = pipeline.preprocess_slice(raw_slice)
    brain_values = raw_slice[brain_roi & np.isfinite(raw_slice)]
    lesion_values = volume[lesion_volume & np.isfinite(volume)]

    voxel_spacing = nii.header.get_zooms()[:3]
    lesion_voxels = int(lesion_volume.sum())
    lesion_volume_mm3 = lesion_voxels * float(np.prod(voxel_spacing))
    summary = {
        "input_image": str(image_path),
        "yolo_model": str(yolo_path),
        "classifier_model": str(classifier_path),
        "lesion_voxels": lesion_voxels,
        "lesion_volume_mm3": lesion_volume_mm3,
        "lesion_volume_cm3": lesion_volume_mm3 / 1000.0,
        "mean_lesion_intensity": (
            float(lesion_values.mean()) if lesion_values.size else None
        ),
        "mean_brain_roi_intensity_display_slice": (
            float(brain_values.mean()) if brain_values.size else None
        ),
        "display_slice": slice_index,
        "mean_ms_probability": mean_ms_probability,
        "classification": "MS" if has_ms_sign else "non_MS",
        "classification_slices": len(slice_probabilities),
        "voxel_spacing_mm": [float(value) for value in voxel_spacing],
        "warning": "Kết quả nghiên cứu, không thay thế chẩn đoán y khoa.",
    }
    pipeline.write_json(output_dir / "app_prediction_summary.json", summary)

    return {
        "volume": volume,
        "lesion_slice": lesion_slice,
        "slice_index": slice_index,
        "brain_roi": brain_roi,
        "preprocessing_images": preprocessing_images,
        "summary": summary,
        "output_dir": output_dir,
    }


class PredictionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dự đoán tổn thương MRI FLAIR")
        self.root.geometry("1360x960")
        self.root.minsize(1060, 780)

        output_default = Path(config.OUTPUT_ROOT) / "app_predictions"
        yolo_default = (
            Path(config.OUTPUT_ROOT)
            / "yolo_runs"
            / "ms_lesion_seg"
            / "weights"
            / "best.pt"
        )
        classifier_default = Path(config.OUTPUT_ROOT) / "best_ms_classifier.pt"

        self.image_path = tk.StringVar()
        self.yolo_path = tk.StringVar(value=str(yolo_default))
        self.classifier_path = tk.StringVar(value=str(classifier_default))
        self.output_path = tk.StringVar(value=str(output_default))
        self.status = tk.StringVar(value="Chọn ảnh FLAIR và kiểm tra đường dẫn hai model.")
        self.result_queue: queue.Queue = queue.Queue()
        self.is_running = False

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        self._path_row(
            controls, 0, "Ảnh FLAIR (.nii/.nii.gz)", self.image_path,
            self._browse_image, "Chọn ảnh..."
        )
        self._path_row(
            controls, 1, "Model YOLO (.pt)", self.yolo_path,
            self._browse_yolo, "Chọn model..."
        )
        self._path_row(
            controls, 2, "Model CNN (.pt)", self.classifier_path,
            self._browse_classifier, "Chọn model..."
        )
        self._path_row(
            controls, 3, "Thư mục lưu kết quả", self.output_path,
            self._browse_output, "Chọn thư mục..."
        )

        actions = ttk.Frame(controls)
        actions.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.predict_button = ttk.Button(
            actions, text="Dự đoán", command=self._start_prediction
        )
        self.predict_button.pack(side="left")
        ttk.Label(actions, textvariable=self.status).pack(
            side="left", padx=12, fill="x", expand=True
        )

        display = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        display.grid(row=1, column=0, sticky="nsew")
        display.columnconfigure(0, weight=1)
        display.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(13, 7.5), dpi=100, constrained_layout=True)
        self.axes = self.figure.subplots(
            2, 4, gridspec_kw={"height_ratios": [1.05, 1]}
        )
        self.canvas = FigureCanvasTkAgg(self.figure, master=display)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._draw_empty()

    @staticmethod
    def _path_row(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command,
        button_text: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=3
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        ttk.Button(parent, text=button_text, command=command).grid(
            row=row, column=2, padx=(8, 0), pady=3
        )

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn ảnh MRI FLAIR",
            filetypes=[
                ("NIfTI", "*.nii *.nii.gz"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if path:
            self.image_path.set(path)

    def _browse_yolo(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn trọng số YOLO",
            filetypes=[("PyTorch model", "*.pt"), ("Tất cả tệp", "*.*")],
        )
        if path:
            self.yolo_path.set(path)

    def _browse_classifier(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn trọng số CNN",
            filetypes=[("PyTorch model", "*.pt"), ("Tất cả tệp", "*.*")],
        )
        if path:
            self.classifier_path.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Chọn thư mục lưu kết quả")
        if path:
            self.output_path.set(path)

    def _draw_empty(self) -> None:
        titles = (
            "FLAIR và vùng tổn thương",
            "Sau Gaussian",
            "Sau Median",
            "Sau CLAHE",
            "Histogram cường độ",
            "Thông số đo",
            "Kết quả CNN",
            "",
        )
        for axis, title in zip(self.axes.flat, titles):
            axis.clear()
            axis.set_title(title)
            axis.axis("off")
        self.canvas.draw_idle()

    def _start_prediction(self) -> None:
        if self.is_running:
            return
        paths = (
            Path(self.image_path.get().strip()),
            Path(self.yolo_path.get().strip()),
            Path(self.classifier_path.get().strip()),
        )
        for label, path in zip(("Ảnh FLAIR", "Model YOLO", "Model CNN"), paths):
            if not path.is_file():
                messagebox.showerror(
                    "Thiếu tệp", f"Không tìm thấy {label}:\n{path}", parent=self.root
                )
                return
        output_value = self.output_path.get().strip()
        if not output_value:
            messagebox.showerror(
                "Thiếu thư mục", "Hãy chọn thư mục lưu kết quả.", parent=self.root
            )
            return
        output_root = Path(output_value)

        self.is_running = True
        self.predict_button.configure(state="disabled")
        self.status.set("Đang phân đoạn YOLO và phân loại CNN...")
        self.root.after(150, self._poll_result)
        worker = threading.Thread(
            target=self._prediction_worker,
            args=(*paths, output_root),
            daemon=True,
        )
        worker.start()

    def _prediction_worker(
        self,
        image_path: Path,
        yolo_path: Path,
        classifier_path: Path,
        output_root: Path,
    ) -> None:
        try:
            result = run_prediction(
                image_path, yolo_path, classifier_path, output_root
            )
            self.result_queue.put(("success", result))
        except Exception as exc:
            self.result_queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _poll_result(self) -> None:
        try:
            result_type, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll_result)
            return

        self.is_running = False
        self.predict_button.configure(state="normal")
        if result_type == "error":
            self.status.set("Dự đoán thất bại.")
            messagebox.showerror("Lỗi dự đoán", payload, parent=self.root)
            return

        self._show_result(payload)

    def _show_result(self, result: dict) -> None:
        volume = result["volume"]
        lesion = result["lesion_slice"]
        slice_index = result["slice_index"]
        roi = result["brain_roi"]
        preprocessing_images = result["preprocessing_images"]
        summary = result["summary"]
        image = volume[:, :, slice_index]

        (
            image_axis,
            gaussian_axis,
            median_axis,
            clahe_axis,
            histogram_axis,
            measurements_axis,
            classification_axis,
            unused_axis,
        ) = self.axes.flat
        image_axis.clear()
        image_axis.imshow(pipeline.scale_u8(image).T, cmap="gray", origin="lower")
        rgba = np.zeros((*lesion.T.shape, 4), dtype=np.float32)
        rgba[lesion.T, :] = (1.0, 0.0, 0.0, 0.32)
        image_axis.imshow(rgba, origin="lower")
        if lesion.any():
            image_axis.contour(
                lesion.T.astype(np.uint8),
                levels=[0.5],
                colors=["red"],
                linewidths=0.8,
                origin="lower",
            )
        image_axis.set_title(f"FLAIR + tổn thương (lát {slice_index})")
        image_axis.axis("off")

        for axis, stage, title in (
            (gaussian_axis, "gaussian", "Sau Gaussian"),
            (median_axis, "median", "Sau Median"),
            (clahe_axis, "clahe", "Sau CLAHE"),
        ):
            axis.clear()
            axis.imshow(
                preprocessing_images[stage].T, cmap="gray", origin="lower"
            )
            axis.set_title(title)
            axis.axis("off")

        histogram_axis.clear()
        histogram_axis.set_title("Histogram cường độ trên lát đang xem")
        histogram_axis.set_xlabel("Cường độ FLAIR")
        histogram_axis.set_ylabel("Số voxel")
        brain_values = image[roi & np.isfinite(image)]
        lesion_values = image[lesion & np.isfinite(image)]
        pooled_values = (
            np.concatenate(
                [values for values in (brain_values, lesion_values) if values.size]
            )
            if brain_values.size or lesion_values.size
            else np.array([], dtype=np.float32)
        )
        bins = (
            np.histogram_bin_edges(
                pooled_values, bins=int(pipeline.cfg("HISTOGRAM_BINS", 64))
            )
            if pooled_values.size
            else int(pipeline.cfg("HISTOGRAM_BINS", 64))
        )
        if brain_values.size:
            histogram_axis.hist(
                brain_values, bins=bins,
                alpha=0.75, label="ROI não"
            )
        if lesion_values.size:
            histogram_axis.hist(
                lesion_values, bins=bins,
                alpha=0.75, label="Tổn thương"
            )
        if brain_values.size or lesion_values.size:
            histogram_axis.legend()

        measurements_axis.clear()
        measurements_axis.axis("off")
        measurements_axis.set_title("Thông số đo")
        lesion_mean = summary["mean_lesion_intensity"]
        roi_mean = summary["mean_brain_roi_intensity_display_slice"]
        voxel_spacing = " × ".join(
            f"{value:.3g}" for value in summary["voxel_spacing_mm"]
        )
        roi_mean_text = f"{roi_mean:.4g}" if roi_mean is not None else "không có dữ liệu"
        lesion_mean_text = (
            f"{lesion_mean:.4g}" if lesion_mean is not None else "không có tổn thương"
        )
        measurements = (
            f"Lát hiển thị: {slice_index}\n"
            f"Mean ROI não (lát này): {roi_mean_text}\n"
            f"Mean tổn thương (toàn thể tích): {lesion_mean_text}"
        )
        measurements += (
            f"\nSố voxel tổn thương: {summary['lesion_voxels']:,}"
            f"\nThể tích tổn thương: {summary['lesion_volume_mm3']:.2f} mm³"
            f" ({summary['lesion_volume_cm3']:.4f} cm³)"
            f"\nVoxel spacing: {voxel_spacing} mm"
        )
        measurements_axis.text(
            0.04, 0.95, measurements, va="top", ha="left",
            fontsize=11, linespacing=1.8, transform=measurements_axis.transAxes
        )

        classification_axis.clear()
        classification_axis.axis("off")
        classification_axis.set_title("Kết quả CNN")
        is_ms = summary["classification"] == "MS"
        classification_text = (
            "CÓ DẤU HIỆU MS" if is_ms else "KHÔNG PHÁT HIỆN DẤU HIỆU MS"
        )
        classification_axis.text(
            0.5, 0.66, classification_text, ha="center", va="center",
            fontsize=16, fontweight="bold",
            color="#b42318" if is_ms else "#26734d",
            wrap=True, transform=classification_axis.transAxes
        )
        classification_axis.text(
            0.5, 0.42,
            f"Xác suất MS trung bình: {summary['mean_ms_probability']:.1%}\n"
            f"Tính từ {summary['classification_slices']} lát FLAIR hợp lệ",
            ha="center", va="center", fontsize=11,
            transform=classification_axis.transAxes
        )
        classification_axis.text(
            0.5, 0.12,
            "Chỉ dùng cho nghiên cứu, không thay thế chẩn đoán y khoa.",
            ha="center", va="center", fontsize=9, wrap=True,
            transform=classification_axis.transAxes
        )
        unused_axis.clear()
        unused_axis.axis("off")

        self.canvas.draw_idle()
        output_dir = result["output_dir"]
        self.status.set(f"Hoàn tất. Đã lưu kết quả tại: {output_dir}")
        messagebox.showinfo(
            "Dự đoán hoàn tất",
            f"Đã lưu mask, xác suất và báo cáo tại:\n{output_dir}",
            parent=self.root,
        )


def main() -> None:
    root = tk.Tk()
    PredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()