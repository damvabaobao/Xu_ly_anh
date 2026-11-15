"""Plot existing PET/CT T0-T3 results; does not run registration or read DICOM.
Usage: python plot_registration_results.py --root results/patient_01
Optional: --no-show, --out PATH. SimpleITK is needed only for .tfm analysis.
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

METHODS = ('Misaligned baseline', 'Rigid MI', 'Multi-resolution MI')
LABELS = ('Baseline', 'Rigid MI', 'Multi-resolution MI')
TESTS = ('T0', 'T1', 'T2', 'T3')
FIELDS = ('mean_control_error_mm', 'rmse_control_error_mm', 'mutual_information_nats', 'seconds')
SHIFTS = (0, 2, 5, 10)


def load(root):
    rows = []
    for test, shift in zip(TESTS, SHIFTS):
        path = root / test / 'registration_metrics.csv'
        with path.open(encoding='utf-8-sig', newline='') as f:
            records = list(csv.DictReader(f))
        if len(records) != 3 or {r['method'] for r in records} != set(METHODS):
            raise ValueError(f'{path}: expected exactly one row for each of {METHODS}')
        for method in METHODS:
            r = next(r for r in records if r['method'] == method)
            motion = [float(r[k]) for k in ('synthetic_shift_x_mm','synthetic_shift_y_mm','synthetic_shift_z_mm','synthetic_rotation_z_deg')]
            if not np.allclose(motion, [shift,0,0,0]):
                raise ValueError(f'{path}: unexpected motion {motion}; expected {[shift,0,0,0]}')
            d = dict(r, test=test)
            for key in FIELDS:
                d[key] = float(r[key])
                if not np.isfinite(d[key]):
                    raise ValueError(f'{path}: nonfinite {key}')
            rows.append(d)
        base = rows[-3]['mean_control_error_mm']
        for r in rows[-3:]:
            r['reduction_percent'] = 100*(base-r['mean_control_error_mm'])/base if base > 1e-12 else ''
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def save(fig, out, name, show):
    fig.tight_layout(); fig.savefig(out/name, dpi=180, bbox_inches='tight')
    if not show: plt.close(fig)


def metric_plots(rows, out, show):
    def values(method, field):
        return np.array([r[field] if r[field] != '' else np.nan for r in rows if r['method']==method], dtype=float)
    x = np.array(SHIFTS)
    for field, title, ylabel, name in [
        (FIELDS[0], 'Mean synthetic control-point error', 'Mean error (mm)', '01_mean_error.png'),
        (FIELDS[1], 'Root mean square control-point error', 'RMSE (mm)', '02_rmse.png'),
        (FIELDS[2], 'Mutual information after registration', 'MI (nats)', '03_mutual_information.png')]:
        fig, ax = plt.subplots(figsize=(8,4.7))
        for method, label in zip(METHODS, LABELS):
            ax.plot(x, values(method,field), 'o-', label=label)
        if field==FIELDS[2]:
            ax.axhline(values(METHODS[0],field)[0], linestyle='--', color='gray', label='Original metadata (T0)')
        ax.set(title=title, xlabel='Synthetic PET shift +X (mm)', ylabel=ylabel, xticks=x)
        ax.grid(alpha=.25); ax.legend(); save(fig,out,name,show)
    fig, ax = plt.subplots(figsize=(8,4.7)); loc=np.arange(3); width=.34
    for i, method in enumerate(METHODS[1:]):
        y=values(method,'reduction_percent')[1:]
        bars=ax.bar(loc+(i-.5)*width,y,width,label=LABELS[i+1]); ax.bar_label(bars,fmt='%.1f%%',padding=3)
    ax.axhline(0,color='gray',linewidth=1); ax.set(xticks=loc,xticklabels=['T1: 2 mm','T2: 5 mm','T3: 10 mm'],ylabel='Error reduction (%)',title='Reduction relative to misaligned baseline (T0 undefined)'); ax.margins(y=.15); ax.legend(); save(fig,out,'04_error_reduction.png',show)
    fig, ax=plt.subplots(figsize=(8,4.7)); loc=np.arange(4)
    for i,method in enumerate(METHODS[1:]):
        bars=ax.bar(loc+(i-.5)*width,values(method,'seconds'),width,label=LABELS[i+1]); ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set(xticks=loc,xticklabels=TESTS,ylabel='Registration time (s)',title='Optimization runtime'); ax.margins(y=.15); ax.legend(); save(fig,out,'05_runtime.png',show)
    fig, ax=plt.subplots(figsize=(8,4.7))
    for method,label in zip(METHODS[1:],LABELS[1:]):
        times=values(method,'seconds'); errs=values(method,FIELDS[0]); line=ax.scatter(times,errs,label=label)
        for t,a,b in zip(TESTS,times,errs): ax.annotate(t,(a,b),xytext=(5,4),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Registration time (s)',ylabel='Mean error (mm)',title='Accuracy vs runtime'); ax.grid(alpha=.25); ax.legend(); save(fig,out,'06_accuracy_runtime.png',show)


def transforms(root,out,show):
    files=['known_fixed_to_moved.tfm','rigid_fixed_to_moved.tfm','multires_fixed_to_moved.tfm']
    missing=[str(root/t/f) for t in TESTS for f in files if not (root/t/f).is_file()]
    if missing:
        print('Skipping transform analysis; missing files:', *missing, sep='\n  '); return
    try: import SimpleITK as sitk
    except ImportError:
        print('Skipping transform analysis: install SimpleITK.'); return
    center=tuple(sitk.Euler3DTransform(sitk.ReadTransform(str(root/'T0'/files[0]))).GetCenter())
    p=np.asarray(center)
    data=[]
    for test in TESTS:
        known=sitk.Euler3DTransform(sitk.ReadTransform(str(root/test/files[0])))
        expected=np.asarray(known.TransformPoint(center))-p
        if not np.allclose(expected,[SHIFTS[TESTS.index(test)],0,0],atol=1e-5) or not np.allclose(known.GetMatrix(),np.eye(3).ravel(),atol=1e-6):
            raise ValueError(f'{test}: .tfm motion does not match T0-T3')
        for label,file in zip(LABELS[1:],files[1:]):
            tx=sitk.Euler3DTransform(sitk.ReadTransform(str(root/test/file)))
            disp=np.asarray(tx.TransformPoint(center))-p
            delta=disp-expected
            rot=np.array([tx.GetAngleX(),tx.GetAngleY(),tx.GetAngleZ()])*180/np.pi
            d={'test':test,'method':label}
            for axis,i in zip('xyz',range(3)):
                d[f'center_{axis}_mm']=p[i]; d[f'known_{axis}_mm']=expected[i]; d[f'estimated_{axis}_mm']=disp[i]; d[f'residual_{axis}_mm']=delta[i]; d[f'estimated_rotation_{axis}_deg']=rot[i]
            d['center_error_mm']=np.linalg.norm(delta); data.append(d)
    write_csv(out/'transform_analysis.csv',data)
    for field,title,ylabel,name in [('estimated','Estimated fixed CT -> moving PET displacement at common center','Displacement (mm)','07_transform_translation.png'),('residual','Residual displacement at common center','Estimated minus known (mm)','08_transform_residual.png'),('estimated_rotation','Estimated Euler angles (known = 0 deg)','Angle (deg)','09_transform_rotation.png')]:
        fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True)
        for axis,ax in zip('xyz',axes):
            suffix='deg' if field=='estimated_rotation' else 'mm'
            for label in LABELS[1:]:
                ax.plot(SHIFTS,[d[f'{field}_{axis}_{suffix}'] for d in data if d['method']==label],'o-',label=label)
            expected=SHIFTS if field=='estimated' and axis=='x' else [0]*4
            ax.plot(SHIFTS,expected,'k--',label='Expected'); ax.set(title=axis.upper(),xlabel='Input +X (mm)',xticks=SHIFTS); ax.grid(alpha=.25)
        axes[0].set_ylabel(ylabel); axes[-1].legend(fontsize=8); fig.suptitle(title); save(fig,out,name,show)
    print('Transform analysis: displacement evaluated at a common physical center, not raw translation parameters.')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'results'/'patient_01'); p.add_argument('--out',type=Path); p.add_argument('--no-show',action='store_true'); args=p.parse_args()
    out=args.out or args.root/'evaluation'; rows=load(args.root); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'summary_T0_T3.csv',rows); metric_plots(rows,out,not args.no_show); transforms(args.root,out,not args.no_show)
    print('Saved evaluation:',out.resolve())
    if not args.no_show: plt.show()

if __name__=='__main__':
    try: main()
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        print('ERROR:',exc,file=sys.stderr); sys.exit(1)
