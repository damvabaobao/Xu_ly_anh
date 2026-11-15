"""PET/CT controlled rigid registration experiment, one DICOM study.
Usage: python 02_register_pet_ct.py [study-folder]
Default: data/patient_01 relative to this script.
Requires: SimpleITK, pydicom, numpy, matplotlib.
Transforms stored in SimpleITK resampling convention: fixed CT -> moving PET.
"""
import argparse
import csv
from pathlib import Path
import sys
import time
import numpy as np
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt


def read_series(root, modality):
    folders = [p for p in root.iterdir() if p.is_dir() and p.name.upper().startswith(modality + '_')]
    if len(folders) != 1:
        raise ValueError(f'Expected one {modality}_ folder in {root}, found {len(folders)}')
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folders[0]))
    if not ids or len(ids) != 1:
        raise ValueError(f'Expected one DICOM series in {folders[0]}')
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folders[0]), ids[0])
    header = pydicom.dcmread(files[0], stop_before_pixels=True)
    if str(header.get('Modality', '')) != modality:
        raise ValueError(f'Unexpected modality in {folders[0]}')
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute(), header


def center(image):
    return image.TransformContinuousIndexToPhysicalPoint(tuple((n-1)/2 for n in image.GetSize()))


def make_moved(pet, transform):
    # Move the entire PET coordinate grid rigidly, without interpolating intensities.
    # A point p in the original PET now lies at transform(p).
    moved = sitk.Image(pet)
    moved.SetOrigin(transform.TransformPoint(pet.GetOrigin()))
    rotation = np.asarray(transform.GetMatrix()).reshape(3, 3)
    direction = np.asarray(pet.GetDirection()).reshape(3, 3)
    moved.SetDirection(tuple((rotation @ direction).ravel()))
    return moved


def register(fixed, moving, mask, multi, seed):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricFixedMask(mask)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=0.03,
                                               numberOfIterations=150,
                                               relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1] if multi else [1])
    reg.SetSmoothingSigmasPerLevel([2, 0] if multi else [0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    initial = sitk.Euler3DTransform()
    initial.SetCenter(center(fixed))
    reg.SetInitialTransform(initial, inPlace=True)
    t0 = time.perf_counter()
    reg.Execute(fixed, moving)
    elapsed = time.perf_counter() - t0
    print('  stop:', reg.GetOptimizerStopConditionDescription())
    print('  objective (negative Mattes MI):', reg.GetMetricValue())
    print('  seconds:', round(elapsed, 2))
    return sitk.Euler3DTransform(initial), elapsed


def resample(moving, fixed, transform):
    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear,
                         float('nan'), sitk.sitkFloat32)


def mi(a, b, a_range, b_range):
    a = np.clip(a, *a_range)
    b = np.clip(b, *b_range)
    counts, _, _ = np.histogram2d(a, b, bins=32, range=[a_range, b_range])
    p = counts / counts.sum()
    independent = p.sum(axis=1)[:, None] * p.sum(axis=0)[None, :]
    valid = p > 0
    return float(np.sum(p[valid] * np.log(p[valid] / independent[valid])))


def recovery_errors(points, estimated, known):
    distances = np.array([np.linalg.norm(np.asarray(estimated.TransformPoint(p)) -
                                        np.asarray(known.TransformPoint(p))) for p in points])
    return float(distances.mean()), float(np.sqrt(np.mean(distances**2)))


def plot(ct, pet, moved, transforms, out):
    z = ct.GetSize()[2] // 2
    ref = sitk.RegionOfInterest(ct, [ct.GetSize()[0], ct.GetSize()[1], 1], [0, 0, z])
    ct2 = sitk.GetArrayFromImage(ref)[0]
    orig = sitk.GetArrayFromImage(resample(pet, ref, sitk.Transform(3, sitk.sitkIdentity)))[0]
    arrs = [orig] + [sitk.GetArrayFromImage(resample(moved, ref, t))[0] for _, t, _ in transforms]
    vals = orig[np.isfinite(orig) & (orig > 0)]
    vmax = max(float(np.percentile(vals, 99)) if len(vals) else 1.0, 1e-6)
    titles = ['Original metadata'] + [name for name, _, _ in transforms]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, title, a in zip(axes, titles, arrs):
        ax.imshow(ct2, cmap='gray', vmin=-200, vmax=250)
        ax.imshow(np.ma.masked_where(~np.isfinite(a) | (a <= .05*vmax), a),
                  cmap='hot', vmin=0, vmax=vmax, alpha=.5)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f'Controlled rigid recovery | CT axial index {z} | common PET display scale')
    fig.tight_layout()
    fig.savefig(out/'registration_comparison.png', dpi=160, bbox_inches='tight')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', type=Path, default=Path(__file__).resolve().parent/'data'/'patient_01')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--shift', nargs=3, type=float, default=[5., -3., 2.], metavar=('X','Y','Z'))
    parser.add_argument('--angle', type=float, default=3., help='Z rotation in degrees')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    out = args.out or Path(__file__).resolve().parent/'results'/(args.folder.name + '_registration')
    out.mkdir(parents=True, exist_ok=True)
    # Fixed seeds; one thread to reduce run-to-run variation.
    seed = 42
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    print('Reading:', args.folder)
    ct, ch = read_series(args.folder, 'CT')
    pet, ph = read_series(args.folder, 'PT')
    for key in ('PatientID','StudyInstanceUID','FrameOfReferenceUID'):
        if not ch.get(key) or str(ch.get(key)) != str(ph.get(key)):
            raise ValueError(f'CT/PET {key} missing or mismatched')
    pet = sitk.Cast(pet, sitk.sitkFloat32)
    print('PET units:', ph.get('Units','unknown'))
    # Coarse CT grid for both methods to limit memory; final display uses native CT.
    fixed = sitk.Cast(sitk.Shrink(ct, [2,2,2]), sitk.sitkFloat32)
    mask = sitk.Cast(fixed > -350, sitk.sitkUInt8)
    # Largest connected foreground component (remove separate objects/table where possible).
    mask = sitk.Cast(sitk.RelabelComponent(sitk.ConnectedComponent(mask)) == 1, sitk.sitkUInt8)
    known = sitk.Euler3DTransform()
    known.SetCenter(center(ct))
    known.SetRotation(0., 0., np.deg2rad(args.angle))
    known.SetTranslation(tuple(args.shift))
    moved = make_moved(pet, known)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    print('Synthetic motion:', args.shift, 'mm; Z rotation:', args.angle, 'deg')
    print('Registering single resolution...')
    rigid, dt1 = register(fixed, moved, mask, False, seed)
    print('Registering multi-resolution...')
    multi, dt2 = register(fixed, moved, mask, True, seed)
    transforms = [('Misaligned baseline', identity, 0.), ('Rigid MI', rigid, dt1), ('Multi-resolution MI', multi, dt2)]
    for name, t in [('known_fixed_to_moved', known), ('rigid_fixed_to_moved', rigid), ('multires_fixed_to_moved', multi)]:
        sitk.WriteTransform(t, str(out/(name+'.tfm')))
    # Evaluate MI on the same independent common set of valid body voxels for every method.
    # It is an evaluation metric, not the sampled optimizer objective.
    ct_a = sitk.GetArrayViewFromImage(fixed)
    body = sitk.GetArrayViewFromImage(mask).astype(bool)
    candidates = np.flatnonzero(body.ravel())
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(candidates, size=min(100000, len(candidates)), replace=False)
    if len(idx) < 20:
        raise ValueError('Too few body voxels for evaluation')
    arrays = []
    for image, tx in [(pet, identity)] + [(moved,t) for _,t,_ in transforms]:
        sampled = resample(image, fixed, tx)
        arrays.append(sitk.GetArrayViewFromImage(sampled).ravel()[idx].copy())
    common = np.logical_and.reduce([np.isfinite(a) for a in arrays])
    if common.sum() < 100:
        raise ValueError('Insufficient common PET coverage for evaluation')
    idx = idx[common]
    arrays = [a[common] for a in arrays]
    a = ct_a.ravel()[idx]
    arange = (-350., 500.)
    brange = (0., max(float(np.percentile(arrays[0], 99.5)), 1e-6))
    # Synthetic control-point error: not anatomical landmark TRE.
    point_idx = rng.choice(idx, size=min(200, len(idx)), replace=False)
    zz, yy, xx = np.unravel_index(point_idx, ct_a.shape)
    points = [fixed.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for x,y,z in zip(xx,yy,zz)]
    rows = []
    print('\nOriginal metadata MI (nats):', mi(a, arrays[0], arange, brange))
    for (name, tx, seconds), values in zip(transforms, arrays[1:]):
        mean, rmse = recovery_errors(points, tx, known)
        row = dict(method=name, synthetic_shift_x_mm=args.shift[0], synthetic_shift_y_mm=args.shift[1],
                   synthetic_shift_z_mm=args.shift[2], synthetic_rotation_z_deg=args.angle,
                   mean_control_error_mm=mean, rmse_control_error_mm=rmse,
                   mutual_information_nats=mi(a, values, arange, brange), common_samples=len(idx), seconds=seconds, seed=seed)
        rows.append(row)
        print(f'{name}: mean error={mean:.3f} mm; RMSE={rmse:.3f} mm; MI={row["mutual_information_nats"]:.5f}')
    with (out/'registration_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plot(ct, pet, moved, transforms, out)
    print('Results:', out)
    print('Note: control-point error measures recovery of known synthetic motion, not clinical anatomical TRE.')
    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(1)
