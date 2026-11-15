"""PET/CT relative transformation analysis for saved T0-T3 experiments.

Run: python analyze_relative_transform.py
     python analyze_relative_transform.py --root results/patient_01 --no-show
     python analyze_relative_transform.py --self-test

Convention: saved transforms map fixed CT coordinates to moving PET coordinates.
Relative map = Ts @ inv(T0), applied to points in original PET physical space.
No DICOM reading or registration. The probe grid is synthetic, not anatomical TRE.
Requires numpy, matplotlib, SimpleITK (only needed to read actual .tfm files).
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

TESTS = ('T0', 'T1', 'T2', 'T3')
SHIFTS = (0., 2., 5., 10.)
FILES = {'Known': 'known_fixed_to_moved.tfm',
         'Rigid MI': 'rigid_fixed_to_moved.tfm',
         'Multi-resolution MI': 'multires_fixed_to_moved.tfm'}
METHODS = ('Rigid MI', 'Multi-resolution MI')
COLORS = {'Rigid MI': 'tab:orange', 'Multi-resolution MI': 'tab:green'}


def apply(h, points):
    return np.asarray(points) @ h[:3, :3].T + h[:3, 3]


def read_tfm(path):
    import SimpleITK as sitk
    if not path.is_file():
        raise FileNotFoundError(path)
    t = sitk.ReadTransform(str(path))
    if t.GetDimension() != 3:
        raise ValueError(f'{path}: expected 3D transform')
    # Derive affine map from TransformPoint, preserving center, rotation order,
    # and even a composite rigid transform. Never subtract raw translations.
    o = np.asarray(t.TransformPoint((0., 0., 0.)))
    r = np.column_stack([np.asarray(t.TransformPoint(tuple(v))) - o
                         for v in np.eye(3)])
    h = np.eye(4); h[:3, :3] = r; h[:3, 3] = o
    if not np.isfinite(h).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1., atol=1e-7):
        raise ValueError(f'{path}: expected a finite rigid transform')
    # Original files are Euler3D. Center is used only for the reference grid.
    center = None
    if t.GetName() == 'Euler3DTransform':
        center = np.array(sitk.Euler3DTransform(t).GetCenter())
    return h, center


def load(root):
    data = {}; center = None
    for test, shift in zip(TESTS, SHIFTS):
        data[test] = {}
        for method, filename in FILES.items():
            h, c = read_tfm(root / test / filename)
            data[test][method] = h
            if test == 'T0' and method == 'Known':
                center = c
        expected = np.eye(4); expected[0, 3] = shift
        if not np.allclose(data[test]['Known'], expected, atol=1e-6, rtol=0):
            raise ValueError(f'{test}: known transform is not the expected pure +X {shift:g} mm motion. Check files.')
    if center is None:
        raise ValueError('T0 known transform must be Euler3D to recover its physical center.')
    return data, center


def probe_grid(center, half_extent, count):
    # Same physical coordinates in every test/method. Not sampled from DICOM.
    axes = [np.linspace(-d, d, count) for d in half_extent]
    xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return xyz + center


def angle_deg(r):
    # Stable SO(3) principal angle, including near identity.
    s = np.linalg.norm([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]]) / 2
    c = np.clip((np.trace(r)-1)/2, -1., 1.)
    return float(np.degrees(np.arctan2(s, c)))


def stats(errors):
    d = np.linalg.norm(errors, axis=1)
    return float(d.mean()), float(np.sqrt(np.mean(d*d))), float(np.percentile(d,95)), float(d.max())


def analyze(data, center, points):
    rows = []; point_rows = []; matrix_rows = []
    for test, shift in zip(TESTS, SHIFTS):
        known = data[test]['Known']
        expected = apply(known, points)
        for method in METHODS:
            h = data[test][method]
            h0 = data['T0'][method]
            relative = h @ np.linalg.inv(h0)
            absolute_err = apply(h, points) - expected
            relative_err = apply(relative, points) - expected
            a = stats(absolute_err); r = stats(relative_err)
            disp = apply(relative, center) - center
            residual = apply(relative, center) - apply(known, center)
            row = {'test': test, 'method': method, 'input_x_mm': shift,
                   'n_probe_points': len(points),
                   'absolute_mean_mm': a[0], 'absolute_rmse_mm': a[1],
                   'absolute_p95_mm': a[2], 'absolute_max_mm': a[3],
                   'relative_mean_mm': r[0], 'relative_rmse_mm': r[1],
                   'relative_p95_mm': r[2], 'relative_max_mm': r[3],
                   'relative_rotation_error_deg': angle_deg(relative[:3,:3] @ known[:3,:3].T)}
            for i, axis in enumerate('xyz'):
                row[f'relative_displacement_{axis}_mm'] = float(disp[i])
                row[f'relative_residual_{axis}_mm'] = float(residual[i])
            row['relative_center_error_mm'] = float(np.linalg.norm(residual))
            rows.append(row)
            matrix_rows.append({'test':test,'method':method, **{f'm{i}{j}':float(relative[i,j]) for i in range(4) for j in range(4)}})
            for k, (p, ea, er) in enumerate(zip(points, absolute_err, relative_err)):
                point_rows.append({'test':test,'method':method,'point_id':k,
                    **{f'p_{axis}_mm':float(p[i]) for i,axis in enumerate('xyz')},
                    'absolute_error_mm':float(np.linalg.norm(ea)), 'relative_error_mm':float(np.linalg.norm(er)),
                    **{f'relative_residual_{axis}_mm':float(er[i]) for i,axis in enumerate('xyz')}})
    return rows, point_rows, matrix_rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def plots(rows, out, show):
    import matplotlib.pyplot as plt
    def val(method, field):
        return [r[field] for r in rows if r['method']==method]
    def save(fig, name):
        fig.tight_layout(); fig.savefig(out/name, dpi=180, bbox_inches='tight')
        if not show: plt.close(fig)
    for field, title, name in [('mean','Mean error','01_absolute_vs_relative_mean.png'),('rmse','RMSE','02_absolute_vs_relative_rmse.png')]:
        fig, ax = plt.subplots(figsize=(8,4.8))
        for method in METHODS:
            for mode, style in [('absolute','o--'),('relative','s-')]:
                ax.plot(SHIFTS,val(method,f'{mode}_{field}_mm'),style,color=COLORS[method],label=f'{method} / {mode}')
        ax.set(title=f'{title}: absolute vs T0-relative (common probe grid)',xlabel='Synthetic PET shift +X (mm)',ylabel=f'{title} (mm)',xticks=SHIFTS); ax.set_ylim(bottom=0); ax.grid(alpha=.25); ax.legend(fontsize=8)
        save(fig,name)
    fig, axs = plt.subplots(1,3,figsize=(12,4.3),sharex=True)
    for axis, ax in zip('xyz', axs):
        for m in METHODS:
            ax.plot(SHIFTS,val(m,f'relative_residual_{axis}_mm'),'o-',color=COLORS[m],label=m)
        ax.axhline(0,color='black',ls='--',lw=1,label='Expected'); ax.set(title=axis.upper(),xlabel='Input +X (mm)',xticks=SHIFTS); ax.grid(alpha=.25)
    axs[0].set_ylabel('Relative residual (mm)'); axs[-1].legend(fontsize=8); fig.suptitle('Residual displacement at common physical center after T0 correction')
    save(fig,'03_relative_residual_xyz.png')
    fig, ax = plt.subplots(figsize=(8,4.8))
    for m in METHODS:
        ax.plot(SHIFTS,val(m,'relative_rotation_error_deg'),'o-',color=COLORS[m],label=m)
    ax.axhline(0,color='black',ls='--',lw=1); ax.set(title='Relative rotation error (SO(3) principal angle)',xlabel='Input +X (mm)',ylabel='Rotation error (degrees)',xticks=SHIFTS); ax.set_ylim(bottom=0); ax.grid(alpha=.25); ax.legend(); save(fig,'04_relative_rotation.png')
    fig, ax = plt.subplots(figsize=(8,4.8))
    for m in METHODS:
        for field, style in [('relative_mean_mm','o-'),('relative_p95_mm','s--'),('relative_max_mm','^:')]:
            ax.plot(SHIFTS,val(m,field),style,color=COLORS[m],label=f'{m} / {field.split("_")[1]}')
    ax.set(title='Relative error distribution on common probe grid',xlabel='Input +X (mm)',ylabel='Error (mm)',xticks=SHIFTS); ax.set_ylim(bottom=0); ax.grid(alpha=.25); ax.legend(fontsize=8); save(fig,'05_relative_error_distribution.png')
    if show: plt.show()


def self_test():
    # Nonzero center and rotation: exact Ts = S @ T0 must recover S.
    c=np.array([100.,-200.,-750.]); theta=.03
    rot=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    h0=np.eye(4); h0[:3,:3]=rot; h0[:3,3]=c-rot@c+[.8,-.5,-1.5]
    data={}
    for test,shift in zip(TESTS,SHIFTS):
        s=np.eye(4); s[0,3]=shift
        data[test]={'Known':s, **{m:s@h0 for m in METHODS}}
    p=probe_grid(c,[100,100,200],5)
    rows,_,_=analyze(data,c,p)
    assert max(r['relative_max_mm'] for r in rows)<1e-9
    assert min(r['absolute_mean_mm'] for r in rows)>0
    # Add known +0.25 mm Y residual to one T3 transform.
    data['T3'][METHODS[0]][1,3]+=.25
    rows,_,_=analyze(data,c,p)
    assert np.isclose(rows[-2]['relative_mean_mm'],.25,atol=1e-9)
    print('SELF-TEST PASSED: rotation, center, composition order, known residual.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'results'/'patient_01')
    p.add_argument('--out',type=Path); p.add_argument('--no-show',action='store_true')
    p.add_argument('--half-extent',type=float,nargs=3,default=[100,100,200],metavar=('X','Y','Z'),help='Probe grid half-widths in mm (default 100 100 200)')
    p.add_argument('--grid-count',type=int,default=5,help='Odd number of probes per axis (default 5)')
    p.add_argument('--self-test',action='store_true'); args=p.parse_args()
    if args.self_test: self_test(); return
    if args.grid_count<3 or args.grid_count>31 or args.grid_count%2==0 or not np.isfinite(args.half_extent).all() or min(args.half_extent)<=0:
        p.error('grid-count must be odd, 3..31; half-extents must be positive finite mm')
    if args.no_show:
        import matplotlib; matplotlib.use('Agg')
    data,center=load(args.root)
    points=probe_grid(center,args.half_extent,args.grid_count)
    rows,point_rows,matrix_rows=analyze(data,center,points)
    out=args.out or args.root/'relative_evaluation'; out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'relative_summary.csv',rows); write_csv(out/'relative_point_errors.csv',point_rows); write_csv(out/'relative_matrices.csv',matrix_rows)
    (out/'analysis_notes.txt').write_text(f'Convention: fixed CT -> moving PET; relative = Ts @ inverse(T0).\nCenter LPS mm: {center.tolist()}\nGrid half-extents mm: {args.half_extent}\nPoints per axis: {args.grid_count}; N={len(points)}\nCommon synthetic physical grid, not anatomical landmarks; points need not be inside body/PET overlap.\nAbsolute errors recomputed on this grid; not identical to original registration_metrics.csv.\nT0 relative error is zero by construction; not independent validation.\nRelative errors compare to known motion, not anatomical accuracy.\n',encoding='utf-8')
    print('Reference center (LPS mm):',center); print('Common probe grid:',len(points),'points')
    print(f'{"Test":<5} {"Method":<21} {"Abs mean":>10} {"Rel mean":>10} {"Rel RMSE":>10} {"Rel angle":>10}')
    for r in rows:
        print(f'{r["test"]:<5} {r["method"]:<21} {r["absolute_mean_mm"]:10.4f} {r["relative_mean_mm"]:10.4f} {r["relative_rmse_mm"]:10.4f} {r["relative_rotation_error_deg"]:10.5f}')
    print('Units: errors mm, angle degrees. T0 relative = 0 by construction.')
    print('Saved to:',out.resolve())
    plots(rows,out,not args.no_show)


if __name__=='__main__':
    try: main()
    except (OSError,ValueError,RuntimeError,ImportError) as e:
        print('ERROR:',e,file=sys.stderr); sys.exit(1)
