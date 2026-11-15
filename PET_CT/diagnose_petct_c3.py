"""PET/CT diagnostic of saved C3 transforms; no registration optimization.
Run from D:\\PET: python diagnose_petct_c3.py --patient patient_04
Requires the same packages as register_petct.py, plus no new dependencies.
Transforms follow SimpleITK fixed CT -> moving PET convention.
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import register_petct as base


def compose(outer, inner):
    # SimpleITK CompositeTransform applies the back transform first.
    tx = sitk.CompositeTransform(3)
    tx.AddTransform(outer)
    tx.AddTransform(inner)
    return tx


def load(folder, name):
    path = folder / (name + '.tfm')
    if not path.is_file():
        raise FileNotFoundError(path)
    t = sitk.ReadTransform(str(path))
    if t.GetDimension() != 3:
        raise ValueError(f'Not a 3D transform: {path}')
    return t


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patient', default='patient_04')
    p.add_argument('--test', default='C3', choices=['C1','C2','C3','T1','T2','T3'])
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--samples', type=int, default=100000)
    args = p.parse_args()
    if args.samples < 100: p.error('--samples must be >= 100')
    root = args.project.resolve()
    results = root/'results'/args.patient
    out = results/'diagnostics'/args.test
    out.mkdir(parents=True, exist_ok=True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    known = load(results/args.test, 'known_fixed_to_moved')
    rigid = load(results/args.test, 'rigid_fixed_to_moved')
    multi = load(results/args.test, 'multires_fixed_to_moved')
    r0 = load(results/'T0', 'rigid_fixed_to_moved')
    m0 = load(results/'T0', 'multires_fixed_to_moved')
    # The synthetic transform saved by register_petct.py is Euler3D.
    known = sitk.Euler3DTransform(known)
    ident = sitk.Transform(3, sitk.sitkIdentity)
    ct, ch = base.read_series(root/'data'/args.patient, 'CT')
    pet, ph = base.read_series(root/'data'/args.patient, 'PT')
    for key in ('PatientID','StudyInstanceUID','FrameOfReferenceUID'):
        if not ch.get(key) or str(ch.get(key)) != str(ph.get(key)):
            raise ValueError('CT/PET metadata mismatch: '+key)
    pet = sitk.Cast(pet, sitk.sitkFloat32)
    moved = base.make_moved(pet, known)
    fixed = sitk.Cast(sitk.Shrink(ct, [2,2,2]), sitk.sitkFloat32)
    mask = sitk.Cast(fixed > -350, sitk.sitkUInt8)
    mask = sitk.Cast(sitk.RelabelComponent(sitk.ConnectedComponent(mask)) == 1, sitk.sitkUInt8)
    # All candidate scores are recomputed on exactly the same sampled fixed voxels.
    cta = sitk.GetArrayViewFromImage(fixed)
    body = np.flatnonzero(sitk.GetArrayViewFromImage(mask).ravel())
    rng = np.random.default_rng(43)
    idx = rng.choice(body, size=min(args.samples, len(body)), replace=False)
    candidates = [
        ('Original metadata', pet, ident),
        ('Misaligned baseline', moved, ident),
        ('Rigid MI saved', moved, rigid),
        ('Multi-resolution MI saved', moved, multi),
        ('Known S (oracle)', moved, known),
        ('S o T0 rigid (oracle)', moved, compose(known, r0)),
        ('S o T0 multi (oracle)', moved, compose(known, m0)),
    ]
    values = []
    for name, image, tx in candidates:
        print('Resampling:', name, flush=True)
        res = base.resample(image, fixed, tx)
        values.append(sitk.GetArrayViewFromImage(res).ravel()[idx].copy())
        del res
    valid = np.logical_and.reduce([np.isfinite(v) for v in values])
    if valid.sum() < 100: raise ValueError('Too few common valid samples')
    idx = idx[valid]
    values = [v[valid] for v in values]
    a = cta.ravel()[idx]
    arange = (-350., 500.)
    brange = (0., max(float(np.percentile(values[0], 99.5)), 1e-6))
    # Shared fixed mask for an independent Mattes MI diagnostic (no pyramid,
    # no stochastic sampling); this is NOT the saved optimizer objective.
    common_mask_arr = np.zeros(cta.shape, dtype=np.uint8)
    common_mask_arr.ravel()[idx] = 1
    common_mask = sitk.GetImageFromArray(common_mask_arr)
    common_mask.CopyInformation(fixed)
    print('Common samples:', len(idx), flush=True)
    rows = []
    for (name, image, tx), b in zip(candidates, values):
        print('Scoring:', name, flush=True)
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        reg.SetMetricFixedMask(common_mask)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetInitialTransform(tx, inPlace=False)
        # MetricEvaluate uses the full-resolution supplied fixed grid and
        # the common mask; no optimizer is run.
        try:
            mattes = -float(reg.MetricEvaluate(fixed, image))
        except RuntimeError as exc:
            print('  Mattes evaluation unavailable:', str(exc).splitlines()[-1])
            mattes = float('nan')
        row = dict(candidate=name, histogram_mi_nats=base.mi(a,b,arange,brange),
                   mattes_mi_common_mask=mattes, common_samples=len(idx),
                   sample_fraction_retained=float(valid.mean()))
        rows.append(row)
        print(f'  histogram MI={row["histogram_mi_nats"]:.6f}; Mattes MI={mattes:.6f}')
    write_csv(out/'mi_diagnostic.csv', rows)
    labels = ['Original','Baseline','Rigid','Multi-res','Known S','S o T0 R','S o T0 M']
    for key, title, fn in [('histogram_mi_nats','Histogram MI on common samples','01_histogram_mi.png'),
                            ('mattes_mi_common_mask','Mattes MI on common fixed mask','02_mattes_mi.png')]:
        fig, ax = plt.subplots(figsize=(10,5))
        ys = [r[key] for r in rows]
        ax.bar(labels, ys)
        ax.set_ylabel('MI (higher is better)'); ax.set_title(f'{args.patient} {args.test}: {title}')
        ax.tick_params(axis='x', rotation=20); ax.grid(axis='y', alpha=.25)
        for i,y in enumerate(ys):
            if np.isfinite(y): ax.annotate(f'{y:.4f}', (i,y), xytext=(0,4), textcoords='offset points', ha='center', fontsize=9)
        fig.tight_layout(); fig.savefig(out/fn, dpi=160); plt.close(fig)
    (out/'notes.txt').write_text(
        'Oracle transforms use the known synthetic motion for diagnostics only; never use as an initialization in the comparative registration experiment.\n'
        'S o T0 applies T0 first, then S. All scores use common fixed voxels and identical histogram settings.\n'
        'Mattes scores are independent full-grid diagnostics, not the original sampled multiresolution optimizer objectives.\n'
        'MI differences alone do not prove a local optimum or anatomical accuracy. No registration was rerun.\n', encoding='utf-8')
    print('Saved:', out)

if __name__ == '__main__':
    try: main()
    except (ValueError, RuntimeError, OSError) as exc:
        print('ERROR:', exc, file=sys.stderr); sys.exit(1)
