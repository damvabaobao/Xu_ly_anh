# MS Brain MRI Lesion Segmentation — U-Net 2D

Starter implementation for a research workflow using FLAIR NIfTI volumes and their `LesionSeg-FLAIR` masks. It audits NIfTI files, checks image-mask geometry, resamples masks to the image grid using nearest-neighbor interpolation when required, splits at patient level, trains a 2D U-Net, evaluates held-out test slices, and predicts a 3D mask with overlays and physical measurements.

## Important dataset assumptions

- Dataset root defaults to `D:\Xử lý ảnh\Brai MIR`.
- One folder per patient, e.g. `Patient-59`.
- File names contain modality tokens `FLAIR`, `T1`, `T2`; mask names contain `LesionSeg` and the modality.
- This first model uses **FLAIR image -> LesionSeg-FLAIR mask**. It does not combine T1/T2 yet.
- The audit script selects one file if duplicates exist and reports duplicates. Review `dataset_audit.csv` before training.
- If the dataset has a different filename convention, edit `modality_from_name`, `is_mask`, `patient_id`, or `resolve_pair` in `ms_pipeline.py`.

## 1. Environment (Windows)

Open PowerShell or the VS Code terminal:

```powershell
cd D:\Cell_A549\brain\ms_lesion_unet_project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build suitable for your machine. The requirements file installs a default PyTorch package; if CUDA is needed, use the installation command recommended by the official PyTorch selector for your OS and CUDA version. CPU mode is supported, but training may be slow.

Check GPU:
```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

## 2. Configure paths

Edit `config.py`:
```python
DATA_ROOT = Path(r"D:\Xử lý ảnh\Brai MIR")
OUTPUT_ROOT = Path(r"D:\Xử lý ảnh\brain_mri_ms_output")
```

## 3. Audit first (do not skip)

```powershell
python ms_pipeline.py audit
```

Review:
- `D:\Xử lý ảnh\brain_mri_ms_output\dataset_audit.csv`
- `D:\Xử lý ảnh\brain_mri_ms_output\audit_summary.json`

Confirm that most/all 60 patients have one FLAIR image and one FLAIR lesion mask. Previously, one empty mask was observed for `Patient-59/59-LesionSeg-T1.nii`; this is a T1 mask and is not the selected FLAIR target. Still, the new audit should verify the FLAIR mask as well. Empty masks may be valid only if the corresponding image truly has no annotated lesion; investigate before excluding them.

## 4. Prepare data and patient-level split

```powershell
python ms_pipeline.py prepare
```

Creates:
- `patient_split.json` (70% train / 15% validation / 15% test by patient)
- `prepared/` resized slices in compressed NPZ format
- `slice_index.csv`
- `prepare_skipped.json`

All slices belonging to a patient remain in the same split to avoid data leakage. Review the CSV and JSON outputs. If patient counts are low because filenames were not recognized, stop and fix file matching before training.

## 5. Train

```powershell
python ms_pipeline.py train
```

Outputs:
- `best_unet2d.pt`
- `training_history.csv`
- `training_curves.png`

Training selects the checkpoint with best validation Dice. This starter code uses a 2D U-Net and a combined BCE + soft Dice loss. It is a baseline, not a clinically validated model.

## 6. Evaluate on held-out test set

```powershell
python ms_pipeline.py evaluate
```

Writes `test_metrics.json` with Dice, IoU, precision, recall, specificity and confusion counts. These metrics aggregate pixels/slices; additionally inspect performance per patient and lesion-level sensitivity before making claims about model quality.

## 7. Predict a new FLAIR NIfTI volume

```powershell
python ms_pipeline.py predict --image "D:\path\to\new-FLAIR.nii.gz"
```

Optional comparison with an available reference mask:
```powershell
python ms_pipeline.py predict --image "D:\path\to\FLAIR.nii.gz" --mask "D:\path\to\LesionSeg-FLAIR.nii.gz"
```

Outputs in `OUTPUT_ROOT\prediction`:
- `predicted_lesion_mask.nii.gz`
- `predicted_probability.nii.gz`
- `overlay_slice_XXX.png` for slices with predicted lesions
- `slice_areas.csv`
- `prediction_summary.json`
- Optional `reference_comparison.json`

The 3D predicted volume is calculated from the NIfTI voxel spacing in mm. Per-slice area uses the first two voxel spacing values. NIfTI array axis order can differ from anatomical display conventions; verify overlays visually against the source image.

## 8. Prediction GUI

After training the YOLO segmentation model and the MS/non-MS classifier, launch the GUI from this folder:

```powershell
python app_check.py
```

Choose a `.nii` or `.nii.gz` FLAIR file and the trained YOLO and CNN checkpoints. The default model paths use `OUTPUT_ROOT\yolo_runs\ms_lesion_seg\weights\best.pt` and `OUTPUT_ROOT\best_ms_classifier.pt`; use the browse buttons if your checkpoints are elsewhere. The window shows a representative axial slice with the YOLO lesion outline, previews after Gaussian, median, and CLAHE preprocessing, an ROI intensity histogram, lesion/ROI mean measurements, lesion volume, and the CNN's mean MS probability across valid slices. The preprocessing previews use the same cumulative order and settings as the prediction pipeline. Predictions and a JSON summary are saved below `OUTPUT_ROOT\app_predictions`.

The CNN result is a research estimate and is not a medical diagnosis. Its image preprocessing and slice-level prediction follow the training pipeline.

## K-means and morphology

Morphological closing and tiny-component filtering are enabled. K-means is disabled by default (`USE_KMEANS_POSTPROCESS = False`) because clustering may delete true small lesions or retain false positives. Enable it only for a controlled validation experiment and compare test metrics before/after.

## Limitations / responsible use

- Research prototype only, not a diagnostic medical device.
- The pipeline uses FLAIR only initially; T1/T2 multi-modal input can be a later experiment after registration and label correspondence are confirmed.
- Small dataset size (60 patients) means patient-level testing is essential and uncertainty is high.
- The prediction command expects the same preprocessing assumptions and compatible MRI acquisition conventions as the training data.
- Review empty masks, duplicates, geometry, and orientation before training.
