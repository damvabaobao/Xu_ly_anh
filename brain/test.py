"""
MRI dataset inventory and NIfTI compatibility check.
Read-only: does not rename, move, or modify source files.

Install (Windows CMD):
    py -3.9 -m pip install nibabel numpy pandas

Run from the folder where this script is saved:
    py -3.9 inspect_mri_datasets.py

Outputs are written to ./outputs_mri_inventory/
"""
from pathlib import Path
import sys

try:
    import nibabel as nib
    import numpy as np
    import pandas as pd
except ImportError as exc:
    print("Missing dependency:", exc)
    print("Install with: py -3.9 -m pip install nibabel numpy pandas")
    sys.exit(1)

MS_ROOT = Path(r"D:\Xử lý ảnh\Brai MIR")
NON_MS_ROOT = Path(r"D:\Xử lý ảnh\non_brain")
OUT_ROOT = Path.cwd() / "outputs_mri_inventory"
MASK_WORDS = ("lesionseg", "lesion_seg", "lesion-mask", "lesion_mask", "mask")


def is_nifti(p):
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def modality(p):
    n = p.name.lower()
    if any(k in n for k in MASK_WORDS):
        if "flair" in n: return "MASK_FLAIR"
        if "t1" in n: return "MASK_T1"
        if "t2" in n: return "MASK_T2"
        return "MASK_OTHER"
    if "flair" in n: return "FLAIR"
    if "t1_acq" in n or "t1-acq" in n or "t1acq" in n: return "T1_ACQ"
    if "t1" in n: return "T1"
    if "t2" in n: return "T2"
    return "OTHER_NIFTI"


def subject_folder(path, root):
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else root.name


def inspect_file(path, group, root):
    rel = str(path.relative_to(root))
    row = {
        "group": group, "subject_folder": subject_folder(path, root),
        "relative_path": rel, "filename": path.name, "modality": modality(path),
        "load_ok": False, "shape": "", "ndim": "", "dtype": "",
        "voxel_spacing_mm": "", "orientation": "", "min": "", "max": "",
        "mean": "", "std": "", "finite_voxels_pct": "", "nonzero_voxels": "",
        "mask_unique_values_if_le20": "", "error": ""
    }
    try:
        img = nib.load(str(path))
        arr = img.get_fdata(dtype=np.float32)
        finite = np.isfinite(arr)
        vals = arr[finite]
        row["load_ok"] = True
        row["shape"] = " x ".join(map(str, img.shape))
        row["ndim"] = len(img.shape)
        row["dtype"] = str(img.get_data_dtype())
        row["voxel_spacing_mm"] = " x ".join(f"{x:.5g}" for x in img.header.get_zooms()[:len(img.shape)])
        try: row["orientation"] = "".join(nib.aff2axcodes(img.affine))
        except Exception: row["orientation"] = "unknown"
        row["finite_voxels_pct"] = round(100 * float(finite.sum()) / arr.size, 6) if arr.size else ""
        row["nonzero_voxels"] = int(np.count_nonzero(vals)) if vals.size else 0
        if vals.size:
            row["min"], row["max"] = float(vals.min()), float(vals.max())
            row["mean"], row["std"] = float(vals.mean()), float(vals.std())
        if "MASK" in row["modality"] and vals.size:
            unique = np.unique(vals)
            row["mask_unique_values_if_le20"] = ", ".join(map(str, unique.tolist())) if len(unique) <= 20 else f">{20} unique values"
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def find_candidate_image(mask_path, root):
    n = mask_path.name.lower()
    mod = "flair" if "flair" in n else ("t1" if "t1" in n else ("t2" if "t2" in n else None))
    if not mod: return None
    rel_parts = mask_path.relative_to(root).parts
    folder = root / rel_parts[0] if len(rel_parts) > 1 else root
    candidates = []
    for p in folder.rglob("*"):
        if not p.is_file() or not is_nifti(p) or p == mask_path: continue
        name = p.name.lower()
        if any(k in name for k in MASK_WORDS): continue
        if mod in name and not ("t1_acq" in name or "t1-acq" in name or "t1acq" in name):
            candidates.append(p)
    candidates.sort(key=lambda p: (len(p.name), p.name.lower()))
    return candidates[0] if candidates else None


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    roots = {"MS": MS_ROOT, "NON_MS": NON_MS_ROOT}
    rows, report = [], [
        "MRI DATASET INVENTORY REPORT",
        f"MS_ROOT: {MS_ROOT}", f"NON_MS_ROOT: {NON_MS_ROOT}",
        f"Output: {OUT_ROOT.resolve()}", ""
    ]
    for group, root in roots.items():
        report.append(f"[{group}]")
        if not root.exists():
            report.append(f"ERROR: Folder does not exist: {root}")
            report.append("")
            continue
        paths = sorted(p for p in root.rglob("*") if p.is_file() and is_nifti(p))
        report.append(f"NIfTI files found: {len(paths)}")
        for i, path in enumerate(paths, 1):
            print(f"[{group}] {i}/{len(paths)}: {path.name}")
            rows.append(inspect_file(path, group, root))
        report.append("")

    if not rows:
        report.append("No NIfTI files found. Check the folder paths and extensions.")
        (OUT_ROOT / "run_report.txt").write_text("\n".join(report), encoding="utf-8")
        print("No NIfTI files found. Check paths in script.")
        return

    df = pd.DataFrame(rows)
    summaries = []
    for (group, subject), f in df.groupby(["group", "subject_folder"], dropna=False):
        mods = set(f["modality"])
        summaries.append({
            "group": group, "subject_folder": subject,
            "n_nifti_files": len(f), "n_loaded_ok": int(f["load_ok"].sum()),
            "has_FLAIR": "FLAIR" in mods, "has_T1": "T1" in mods,
            "has_T1_ACQ": "T1_ACQ" in mods, "has_T2": "T2" in mods,
            "has_MASK_FLAIR": "MASK_FLAIR" in mods,
            "has_MASK_T1": "MASK_T1" in mods, "has_MASK_T2": "MASK_T2" in mods,
            "modalities_found": ", ".join(sorted(mods)),
            "load_errors": int((~f["load_ok"]).sum())
        })
    sdf = pd.DataFrame(summaries)

    checks = []
    for _, r in df[df["modality"].str.startswith("MASK")].iterrows():
        root = roots[r["group"]]
        mp = root / r["relative_path"]
        ip = find_candidate_image(mp, root)
        c = {
            "group": r["group"], "subject_folder": r["subject_folder"],
            "mask_file": r["relative_path"],
            "candidate_image": str(ip.relative_to(root)) if ip else "",
            "candidate_found": ip is not None, "same_shape": "",
            "same_affine_approx": "", "mask_unique_values_if_le20": r["mask_unique_values_if_le20"],
            "note": ""
        }
        if ip is None:
            c["note"] = "No likely image found by filename; manually verify."
        else:
            try:
                m, im = nib.load(str(mp)), nib.load(str(ip))
                c["same_shape"] = tuple(m.shape) == tuple(im.shape)
                c["same_affine_approx"] = bool(np.allclose(m.affine, im.affine, atol=1e-3))
                c["note"] = "Shape/affine match is preliminary; visually inspect overlay." if c["same_shape"] and c["same_affine_approx"] else "Shape and/or affine differ; investigate before resampling."
            except Exception as e:
                c["note"] = f"Comparison error: {type(e).__name__}: {e}"
        checks.append(c)

    df.to_csv(OUT_ROOT / "file_inventory.csv", index=False, encoding="utf-8-sig")
    sdf.to_csv(OUT_ROOT / "patient_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checks).to_csv(OUT_ROOT / "mask_image_checks.csv", index=False, encoding="utf-8-sig")

    report += [
        "SUMMARY", "-------",
        f"Total NIfTI files: {len(df)}",
        f"Loaded successfully: {int(df['load_ok'].sum())}",
        f"Load errors: {int((~df['load_ok']).sum())}",
        f"Inferred subject folders: {len(sdf)}", "", "By group:"
    ]
    for group, f in sdf.groupby("group"):
        report.append(f"  {group}: {len(f)} inferred subject folders; {int(f['n_nifti_files'].sum())} NIfTI files")
    report += [
        "", "IMPORTANT:",
        "- Subject folder is inferred from first directory under each root; verify it matches patient IDs.",
        "- T1_acq is kept as a separate label for inspection; do not assume it is an independent modality.",
        "- Mask-image pairing is filename-based and preliminary only.",
        "- Matching shape/affine does not prove correct mask alignment; inspect overlays.",
        "- Source data are not modified. This script does not normalize or resample images.",
        "", "CSV outputs: file_inventory.csv, patient_summary.csv, mask_image_checks.csv"
    ]
    (OUT_ROOT / "run_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nDONE. Results saved to:", OUT_ROOT.resolve())
    print("Open patient_summary.csv first, then file_inventory.csv and mask_image_checks.csv.")
    print("Read run_report.txt for warnings.")


if __name__ == "__main__":
    main()
