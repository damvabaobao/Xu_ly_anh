"""Read one PET/CT DICOM study and preview geometry-based fusion.
Usage: python 01_check_pet_ct.py [path-to-study-folder]
The study folder must directly contain CT_... and PT_... subfolders.
"""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk


def choose_folder():
        return str(Path(__file__).resolve().parent/"data"/"patient_01")


def load_series(root, modality):
    folders = [p for p in root.iterdir()
               if p.is_dir() and p.name.upper().startswith(modality + "_")]
    if len(folders) != 1:
        raise ValueError(f"Expected one {modality}_ folder, found {len(folders)}. Check selected folder.")
    folder = folders[0]
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
    if not ids or len(ids) != 1:
        raise ValueError(f"Expected one DICOM series in {folder.name}, found {len(ids or [])}.")
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), ids[0])
    header = pydicom.dcmread(files[0], stop_before_pixels=True)
    if str(header.get("Modality", "")) != modality:
        raise ValueError(f"Unexpected Modality in {folder.name}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D {modality} image.")
    print(f"\n===== {modality} =====")
    print("DICOM files:", len(files))
    print("Size (x,y,z):", image.GetSize())
    print("Spacing (mm):", image.GetSpacing())
    print("Origin (mm):", image.GetOrigin())
    print("Direction:", image.GetDirection())
    print("Pixel type:", image.GetPixelIDTypeAsString())
    if modality == "PT":
        print("PET Units (DICOM):", header.get("Units", "Not specified"))
    return image, header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", help="Folder directly containing CT_ and PT_ series")
    args = parser.parse_args()
    selected = args.folder or choose_folder()
    if not selected:
        print("No folder selected.")
        return
    root = Path(selected)
    ct, ct_header = load_series(root, "CT")
    pet, pet_header = load_series(root, "PT")
    for tag in ("PatientID", "StudyInstanceUID"):
        a, b = str(ct_header.get(tag, "")), str(pet_header.get(tag, ""))
        if not a or not b or a != b:
            raise ValueError(f"Cannot confirm matching {tag}. Check that CT and PET belong to the same study.")
    print("\nCT and PET: matching PatientID and StudyInstanceUID.")
    a = str(ct_header.get("FrameOfReferenceUID", ""))
    b = str(pet_header.get("FrameOfReferenceUID", ""))
    same_frame = bool(a and b and a == b)
    print("Same FrameOfReferenceUID:", same_frame)
    if not same_frame:
        print("No overlay generated: verify coordinate frames or establish a transform first.")
        return

    # Find a CT slice near the PET volume center in physical coordinates.
    pet_center = pet.TransformContinuousIndexToPhysicalPoint(
        tuple((n - 1) / 2 for n in pet.GetSize()))
    ct_index = ct.TransformPhysicalPointToContinuousIndex(pet_center)
    nx, ny, nz = ct.GetSize()
    z = int(round(ct_index[2]))
    if not 0 <= z < nz:
        raise ValueError("PET center is outside the CT slice range; inspect field of view.")
    # Resample onto a single CT slice (3D grid with depth 1), saving memory.
    ct_slice = sitk.RegionOfInterest(ct, [nx, ny, 1], [0, 0, z])
    pet_slice = sitk.Resample(pet, ct_slice, sitk.Transform(3, sitk.sitkIdentity),
                              sitk.sitkLinear, float('nan'), sitk.sitkFloat32)
    ct2 = sitk.GetArrayFromImage(ct_slice)[0]
    pet2 = sitk.GetArrayFromImage(pet_slice)[0]
    finite = pet2[np.isfinite(pet2)]
    if finite.size == 0:
        raise ValueError("No overlapping PET samples on this CT slice.")
    vmax = max(float(np.percentile(finite, 99)), 1e-6)
    pet_display = np.ma.masked_invalid(pet2)
    sx, sy, _ = ct.GetSpacing()
    aspect = sy / sx
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(ct2, cmap="gray", vmin=-200, vmax=250, aspect=aspect)
    ax[0].set_title(f"CT (slice {z})")
    ax[1].imshow(pet_display, cmap="hot", vmin=0, vmax=vmax, aspect=aspect)
    ax[1].set_title("PET on CT grid")
    ax[2].imshow(ct2, cmap="gray", vmin=-200, vmax=250, aspect=aspect)
    ax[2].imshow(np.ma.masked_where(~np.isfinite(pet2) | (pet2 <= 0.05*vmax), pet2),
                 cmap="hot", vmin=0, vmax=vmax, alpha=0.5, aspect=aspect)
    ax[2].set_title("Baseline overlay (alpha=0.5)")
    for item in ax:
        item.axis("off")
    fig.suptitle("Metadata-based alignment only; no optimized registration")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "baseline_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("\nSaved preview:", out)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
