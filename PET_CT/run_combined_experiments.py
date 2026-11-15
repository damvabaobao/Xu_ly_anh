"""Run PET/CT combined translation + Z rotation tests using existing register_petct.py.
C1/C2/C3: translation (5,-3,2) mm, rotation 3/5/10 deg about CT physical center.
Run: python run_combined_experiments.py --patients patient_01 patient_02
Existing complete outputs are reused only if their known transform matches the protocol.
Requires the existing registration script and its dependencies.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import traceback
import tempfile
import shutil
import os
from datetime import datetime
import numpy as np

CASES={'C1':3.,'C2':5.,'C3':10.}
SHIFT=(5.,-3.,2.)
METHODS=('Misaligned baseline','Rigid MI','Multi-resolution MI')
COLORS=('tab:blue','tab:orange','tab:green')
FILES=('known_fixed_to_moved.tfm','rigid_fixed_to_moved.tfm','multires_fixed_to_moved.tfm')


def matrix(tx):
    o=np.asarray(tx.TransformPoint((0.,0.,0.))); r=np.column_stack([np.asarray(tx.TransformPoint(tuple(v)))-o for v in np.eye(3)])
    if not np.isfinite(o).all() or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(r),1.,atol=1e-7): raise ValueError('Expected rigid transform')
    h=np.eye(4);h[:3,:3]=r;h[:3,3]=o;return h


def apply(h,p): return np.asarray(p)@h[:3,:3].T+h[:3,3]


def error(h,known,pts):
    d=np.linalg.norm(apply(h,pts)-apply(known,pts),axis=1)
    return float(d.mean()),float(np.sqrt(np.mean(d*d)))


def angle_error(h,known):
    r=h[:3,:3]@known[:3,:3].T
    sine=np.linalg.norm([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
    return float(np.degrees(np.arctan2(sine,np.clip((np.trace(r)-1)/2,-1,1))))


def read_case(folder,angle):
    import SimpleITK as sitk
    
    missing=[f for f in (*FILES,'registration_metrics.csv') if not (folder/f).is_file()]
    if missing: raise FileNotFoundError(f'{folder}: missing {", ".join(missing)}')
    tx=sitk.ReadTransform(str(folder/FILES[0])); known=matrix(tx)
    if tx.GetName()!='Euler3DTransform': raise ValueError(f'{folder}: expected Euler3D known transform')
    tx=sitk.Euler3DTransform(tx); c=np.asarray(tx.GetCenter())
    if not np.allclose(tx.GetTranslation(),SHIFT if angle else (0,0,0),atol=1e-6,rtol=0) or not np.allclose(np.degrees([tx.GetAngleX(),tx.GetAngleY(),tx.GetAngleZ()]),[0,0,angle],atol=1e-6,rtol=0): raise ValueError(f'{folder}: saved motion does not match requested protocol')
    rigid,multi=[matrix(sitk.ReadTransform(str(folder/f))) for f in FILES[1:]]
    with (folder/'registration_metrics.csv').open(encoding='utf-8-sig',newline='') as f: rows={r['method']:r for r in csv.DictReader(f)}
    for m in METHODS:
        if m not in rows: raise ValueError(f'{folder}: missing method {m}')
        r=rows[m]; expected=SHIFT if angle else (0,0,0)
        if not np.allclose([float(r[f'synthetic_shift_{a}_mm']) for a in 'xyz'],expected,atol=1e-6,rtol=0) or not np.isclose(float(r['synthetic_rotation_z_deg']),angle,atol=1e-6,rtol=0): raise ValueError(f'{folder}: CSV and requested motion disagree')
    return c,known,[np.eye(4),rigid,multi],rows


def summarize(root,patient):
    # T0 provides each method's reference for relative motion analysis.
    c,_,t0,_=read_case(root/'T0',0)
    pts=c+np.stack(np.meshgrid(*[np.linspace(-v,v,5) for v in (100,100,200)],indexing='ij'),axis=-1).reshape(-1,3)
    rows=[]
    for case,angle in CASES.items():
        center,known,est,old=read_case(root/case,angle)
        if not np.allclose(center,c,atol=.01,rtol=0): raise ValueError(f'{root/case}: CT center differs from T0')
        for m,h,h0 in zip(METHODS,est,t0):
            a,ar=error(h,known,pts); rel=h@np.linalg.inv(h0); b,br=error(rel,known,pts)
            residual=apply(rel,c)-apply(known,c)
            r=old[m]
            rows.append(dict(patient=patient,test=case,rotation_z_deg=angle,shift_x_mm=SHIFT[0],shift_y_mm=SHIFT[1],shift_z_mm=SHIFT[2],method=m,absolute_mean_mm=a,absolute_rmse_mm=ar,relative_mean_mm=b,relative_rmse_mm=br,absolute_rotation_error_deg=angle_error(h,known),relative_rotation_error_deg=angle_error(rel,known),relative_center_x_residual_mm=residual[0],relative_center_y_residual_mm=residual[1],relative_center_z_residual_mm=residual[2],mutual_information_nats=float(r['mutual_information_nats']),seconds=float(r.get('seconds') or 0),n_probe_points=len(pts),mi_common_samples=int(r['common_samples'])))
    out=root/'combined_evaluation';out.mkdir(exist_ok=True)
    write_csv(out/'combined_summary.csv',rows)
    (out/'notes.txt').write_text('Fixed translation (5,-3,2) mm and Z rotations 3,5,10 deg about CT center. Known motion is changed PET geometry, not resampled motion. 125 fixed physical probes centered on CT with half-extents (100,100,200) mm. Absolute errors vs known transform; relative = Ts @ inv(T0), not clinical TRE. MI from each original run uses its own common valid sample set; compare methods within each test, not absolute MI across angles. T0 is not anatomical ground truth.\n',encoding='utf-8')
    plots(rows,out,patient)
    print('\nCombined summary:',patient)
    for r in rows: print(f"{r['test']} {r['method']:22} abs={r['absolute_mean_mm']:.4f} mm rel={r['relative_mean_mm']:.4f} mm angle err={r['relative_rotation_error_deg']:.5f} deg")
    print('Saved:',out.resolve())
    return rows


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plots(rows,out,patient):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def v(m,key): return [r[key] for r in rows if r['method']==m]
    for i,(key,title,y,methods) in enumerate([('absolute_mean_mm','Absolute control-point error','Mean error (mm)',METHODS),('relative_mean_mm','T0-relative control-point error','Mean error (mm)',METHODS[1:]),('relative_rotation_error_deg','T0-relative rotation error','Error (degrees)',METHODS[1:]),('seconds','Registration runtime','Time (s)',METHODS[1:])],1):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for m in methods: ax.plot(list(CASES.values()),v(m,key),'o-',color=COLORS[METHODS.index(m)],label=m)
        ax.set(title=f'{patient}: {title}',xlabel='Input Z rotation (degrees), shift fixed (5,-3,2) mm',ylabel=y,xticks=list(CASES.values()));ax.set_ylim(bottom=0);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/f'{i:02d}_{key}.png',dpi=180);plt.close(fig)


def locate_study(path):
    """Accept a study folder or a wrapper containing one unambiguous study."""
    if not path.is_dir():
        raise FileNotFoundError(f'Patient folder not found: {path}')
    def valid(p):
        names=[x.name.upper() for x in p.iterdir() if x.is_dir()]
        return sum(n.startswith('CT_') for n in names)==1 and sum(n.startswith('PT_') for n in names)==1
    if valid(path): return path
    candidates=[p for p in path.rglob('*') if p.is_dir() and valid(p)]
    if len(candidates)!=1:
        raise ValueError(f'{path}: found {len(candidates)} study folders containing CT_ and PT_. Specify one study, not multiple studies.')
    return candidates[0]


def run_case(script, data, root, case, angle, force):
    """Stage runs in a temporary folder; preserve earlier completed outputs."""
    out=root/case
    try:
        read_case(out,angle)
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        if out.exists(): print(f'{case}: previous output incomplete/incompatible: {exc}',flush=True)
    else:
        if not force and (out/'registration_comparison.png').is_file():
            print('Reusing:',out,flush=True); return
    root.mkdir(parents=True,exist_ok=True)
    logs=root/'combined_logs';logs.mkdir(exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logfile=logs/f'{case}_{stamp}.log'
    with tempfile.TemporaryDirectory(prefix=f'.{case}_',dir=root) as td:
        stage=Path(td)/'output'
        cmd=[sys.executable,'-u',str(script),str(data),'--shift',*[str(x) for x in SHIFT],
             '--angle',str(angle),'--out',str(stage),'--no-show']
        print('Running:',data,case,flush=True)
        print('Command:',subprocess.list2cmdline(cmd),flush=True)
        env=dict(os.environ,MPLBACKEND='Agg',PYTHONIOENCODING='utf-8')
        with logfile.open('w',encoding='utf-8') as log:
            log.write(subprocess.list2cmdline(cmd)+'\n')
            with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  cwd=str(script.parent),env=env,text=True,encoding='utf-8',errors='replace') as proc:
                for line in proc.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=proc.wait()
        if code: raise RuntimeError(f'{case}: registration exited with code {code}. See log: {logfile}')
        read_case(stage,angle)
        if not (stage/'registration_comparison.png').is_file():
            raise RuntimeError(f'{case}: registration did not export comparison image. See {logfile}')
        backup=None
        if out.exists():
            backup=root/'combined_backups'/f'{case}_{stamp}'
            backup.parent.mkdir(exist_ok=True)
            out.rename(backup)
        try: stage.rename(out)
        except OSError:
            if backup is not None and not out.exists(): backup.rename(out)
            raise
    print('Saved:',out,flush=True)


def main():
    base=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patients',nargs='+',default=['patient_01'])
    p.add_argument('--data-root',type=Path,default=base/'data')
    p.add_argument('--results-root',type=Path,default=base/'results')
    p.add_argument('--register-script',type=Path,default=None)
    p.add_argument('--force',action='store_true',help='Rerun even valid C1-C3; keep backup')
    p.add_argument('--summary-only',action='store_true',help='Read saved results; no registration')
    p.add_argument('--check-only',action='store_true',help='Check prerequisites and T0 without running')
    p.add_argument('--no-show',action='store_true',help='Accepted for compatibility; plots always saved only')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--debug',action='store_true')
    a=p.parse_args()
    if a.self_test:
        s=np.eye(4);theta=np.deg2rad(10);s[:3,:3]=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
        c=np.array([20.,-30.,100.]);s[:3,3]=c+SHIFT-s[:3,:3]@c
        h0=np.eye(4);h0[1,3]=1.5;h=s@h0;pts=np.array([[0,0,0],[50,100,200]])
        assert np.allclose(error(h@np.linalg.inv(h0),s,pts),0);assert np.allclose(error(h,s,pts),1.5)
        assert angle_error(s,np.eye(4))>9.99
        print('SELF-TEST PASSED: rotation, nonzero center, composition, errors.');return
    import SimpleITK
    a.data_root=a.data_root.resolve();a.results_root=a.results_root.resolve()
    script=a.register_script
    if not a.summary_only:
        if script is None:
            script=next((base/n for n in ('register_petct.py','02_register_pet_ct.py') if (base/n).is_file()),None)
        if script is None or not script.is_file(): raise FileNotFoundError('Missing register_petct.py. Put it next to this script or use --register-script PATH.')
        script=script.resolve()
    # Check all prerequisites BEFORE spending time on registration.
    plan=[]
    for patient in a.patients:
        root=a.results_root/patient
        if not root.is_dir(): raise FileNotFoundError(f'Missing {root}. Run T0 first using register_petct.py.')
        try: read_case(root/'T0',0)
        except (OSError,RuntimeError,ValueError,KeyError) as exc:
            raise ValueError(f'{patient}: T0 invalid or incomplete. Run T0 first. Details: {exc}') from exc
        data=None if a.summary_only else locate_study(a.data_root/patient)
        print('Preflight OK:',patient,'T0:',root/'T0','data:',data,flush=True)
        plan.append((patient,data,root))
    if a.check_only: print('CHECK PASSED: no registration run.');return
    allrows=[]
    for patient,data,root in plan:
        for case,angle in CASES.items():
            if a.summary_only: read_case(root/case,angle)
            else: run_case(script,data,root,case,angle,a.force)
        allrows.extend(summarize(root,patient))
    if len(a.patients)>1:
        out=a.results_root/'combined_summary_all_patients.csv';write_csv(out,allrows);print('All-patient combined results:',out)

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'ERROR [{type(e).__name__}]: {e}',file=sys.stderr)
        if '--debug' in sys.argv: traceback.print_exc()
        if isinstance(e,ImportError): print('Install: python -m pip install SimpleITK pydicom numpy matplotlib',file=sys.stderr)
        sys.exit(1)
