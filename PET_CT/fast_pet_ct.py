"""Fast rigid PET/CT registration: screen 7 angles, optimize top K, refine 2.
Requires the existing register_petct.py, register_petct_improved.py and fixed
 evaluate_seg_roi.py in the same project. No affine, no overwrite of old results.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import register_petct_improved as core

ANGLES = [-15,-10,-5,0,5,10,15]
TESTS = ['T0','T1','T2','T3','C1','C2','C3']


def rank(pts,a,sample,items,brange,base):
    scores,n=core.score_group(pts,a,sample,[core.matrix(t) for _,t in items],brange,base)
    order=sorted(range(len(items)),key=lambda j:(-scores[j],j))
    return order,scores,n


def select_distinct(items,order,c):
    chosen=[]
    for j in order:
        name,t=items[j]; h=core.matrix(t)
        if all(np.linalg.norm(core.apply(h,c)-core.apply(core.matrix(v),c))>.25 or core.angle_error(h,core.matrix(v))>.25 for _,v in chosen):
            chosen.append((name,t))
        if len(chosen)==2: break
    return chosen


def fast(fixed,moved,mask,c,k,sitk,base,out):
    start=time.perf_counter(); logs=[]; trace=[]
    pts,a=core.sample_points(fixed,mask,sitk,60000,91); sample=core.sampler(moved,sitk)
    native=sitk.GetArrayViewFromImage(moved).ravel()[::31]; positive=native[np.isfinite(native)&(native>0)]
    if not len(positive): raise ValueError('PET has no positive finite intensities')
    brange=(0.,max(float(np.percentile(positive,99.5)),1e-6))
    initial=[]
    for deg in ANGLES:
        t=sitk.Euler3DTransform();t.SetCenter(tuple(c));t.SetRotation(0.,0.,np.deg2rad(deg))
        initial.append((f'initial {deg:+d}',t))
    order,scores,n=rank(pts,a,sample,initial,brange,base)
    selected=set(order[:k]); logs.extend(dict(stage='screen',candidate=initial[j][0],histogram_mi=scores[j],common_samples=n,selected=int(j in selected)) for j in order)
    print(f'K={k}: selected angles {[ANGLES[j] for j in order[:k]]}',flush=True)
    # Keep all raw starts as fallback; only K get expensive coarse optimization.
    items=list(initial)
    for j in sorted(selected):
        name,t=initial[j]; print('  coarse',name,flush=True)
        try:
            t,stop=core.optimize(fixed,moved,mask,sitk.Euler3DTransform(t),[4,2],[4,2],2.,120,sitk)
            items.append((f'coarse {ANGLES[j]:+d}',t));trace.append(dict(stage='coarse',candidate=name,status=stop))
        except RuntimeError as e: trace.append(dict(stage='coarse',candidate=name,status=str(e)))
    order,scores,n=rank(pts,a,sample,items,brange,base); chosen=select_distinct(items,order,c); names={name for name,_ in chosen}
    logs.extend(dict(stage='coarse ranking',candidate=items[j][0],histogram_mi=scores[j],common_samples=n,selected=int(items[j][0] in names)) for j in order)
    finalists=[(f'pre-fine {name}',sitk.Euler3DTransform(t)) for name,t in chosen]
    for name,t in chosen:
        print('  fine',name,flush=True)
        try:
            r,stop=core.optimize(fixed,moved,mask,sitk.Euler3DTransform(t),[2,1],[2,0],1.,200,sitk)
            finalists.append((f'fine {name}',r));trace.append(dict(stage='fine',candidate=name,status=stop))
        except RuntimeError as e: trace.append(dict(stage='fine',candidate=name,status=str(e)))
    order,scores,n=rank(pts,a,sample,finalists,brange,base); best=order[0]
    logs.extend(dict(stage='final ranking',candidate=finalists[j][0],histogram_mi=scores[j],common_samples=n,selected=int(j==best)) for j in order)
    result=finalists[best][1];elapsed=time.perf_counter()-start
    out.mkdir(parents=True,exist_ok=True);sitk.WriteTransform(result,str(out/'fast_fixed_to_moved.tfm'))
    core.write_csv(out/'candidate_scores.csv',logs);core.write_csv(out/'optimizer_log.csv',trace)
    print(f'  chosen {finalists[best][0]} | MI={scores[best]:.6f} | {elapsed:.2f} s',flush=True)
    return result,elapsed


def method_name(k): return f'Fast-{k}' if k<7 else 'Full-7'


def self_test():
    h=np.eye(4);h[:3,3]=[5,-3,2];p=np.array([[1.,2.,3.]])
    assert np.allclose(core.apply(h,p),[[6,-1,5]])
    assert core.errors(h,h,p)==(0.,0.)
    assert method_name(7)=='Full-7'
    # Ranking and pruning logic can be checked without DICOM/ITK.
    class Tx:
        def __init__(self,x): self.x=x
        def TransformPoint(self,p): return tuple(np.asarray(p)+[self.x,0,0])
    items=[('a',Tx(0)),('b',Tx(.1)),('c',Tx(2))]
    assert [n for n,_ in select_distinct(items,[0,1,2],np.zeros(3))]==['a','c']
    print('SELF-TEST PASSED: transform math and distinct candidate selection')

def roi_eval(ct,pet,segments,rois,cases,out,roi,sitk,plt,units,methods):
    rows=[]
    for j,r in enumerate(sorted(rois,key=lambda r:r['volume_ml'],reverse=True)):
        c=r['center'];ref=roi.make_ref(ct,r['corners'],17);ctarr=sitk.GetArrayFromImage(ref)
        target,bg=roi.regions(r,segments,ref,5,15,ctarr>-350)
        # Per test same physical CT-grid target/background and common support for every method.
        for test,(known,hs) in cases.items():
            maps=[('Original metadata',np.eye(4),np.eye(4))]+[(m,known,h) for m,h in zip(methods,hs)]
            transforms=[np.linalg.inv(k)@h for m,k,h in maps]
            vals=[roi.resample(pet,ref,h) for h in transforms];common=np.logical_and.reduce([np.isfinite(v) for v in vals]);tar=target&common;back=bg&common
            for (m,k,h),v in zip(maps,vals):
                q=core.apply(np.linalg.inv(h)@k,c);e=q-c;cnr,mean,peak,bmean,bsd=roi.cnr_values(v,tar,back)
                # Exact continuous volume ratio for affine output (not a voxel-count approximation).
                ratio=abs(np.linalg.det((np.linalg.inv(h)@k)[:3,:3]));
                rows.append(dict(test=test,method=m,roi_id=r['id'],centroid_error_mm=float(np.linalg.norm(e)),dx_mm=e[0],dy_mm=e[1],dz_mm=e[2],cnr=cnr,pet_mean=mean,pet_max=peak,pet_units=units,volume_ml=r['volume_ml'],output_volume_ml=r['volume_ml']*ratio,volume_ratio=ratio,target_n=int(tar.sum()),background_n=int(back.sum())))
            if j==0:
                z=int(np.clip(round(ref.TransformPhysicalPointToContinuousIndex(tuple(c))[2]),0,ctarr.shape[0]-1));cols=3; nrows=(len(maps)+cols-1)//cols; fig,axs=plt.subplots(nrows,cols,figsize=(14,4.5*nrows),squeeze=False);positive=vals[0][np.isfinite(vals[0])&(vals[0]>0)];vmax=max(float(np.percentile(positive,99)) if len(positive) else 1.,1e-6)
                for ax,(m,k,h),v,tr in zip(axs.ravel(),maps,vals,transforms):
                    ax.imshow(ctarr[z],cmap='gray',vmin=-200,vmax=250);ax.imshow(np.ma.masked_where(~np.isfinite(v[z])|(v[z]<=.05*vmax),v[z]),cmap='hot',vmin=0,vmax=vmax,alpha=.5)
                    mask=roi.resample(r['mask'],ref,tr,mask=True)>0
                    if target[z].any() and not target[z].all():ax.contour(target[z],levels=[.5],colors='lime')
                    if mask[z].any() and not mask[z].all():ax.contour(mask[z],levels=[.5],colors='cyan',linestyles='dashed')
                    ax.set_title(m);ax.axis('off')
                for ax in axs.ravel()[len(maps):]: ax.axis("off")
                fig.suptitle(f'{test} | {r["id"]} | fixed SEG green, transformed SEG cyan');fig.tight_layout();fig.savefig(out/f'{test}_roi_overlay.png',dpi=160);plt.close(fig)
        print('ROI:',r['id'],flush=True)
    core.write_csv(out/'roi_metrics.csv',rows);summary=[]
    for test in cases:
        for m in ['Original metadata']+methods:
            rr=[r for r in rows if r['test']==test and r['method']==m];v=np.asarray([r['cnr'] for r in rr]);v=v[np.isfinite(v)]
            summary.append(dict(test=test,method=m,n_roi=len(rr),mean_centroid_error_mm=np.mean([r['centroid_error_mm'] for r in rr]),max_centroid_error_mm=np.max([r['centroid_error_mm'] for r in rr]),median_cnr=np.median(v) if len(v) else np.nan,n_valid_cnr=len(v),mean_volume_ratio=np.mean([r['volume_ratio'] for r in rr])))
    core.write_csv(out/'roi_summary.csv',summary)
    for key,title in [('mean_centroid_error_mm','Mean SEG centroid error (mm)'),('median_cnr','Median CNR on fixed ROI'),('mean_volume_ratio','Continuous ROI output / original volume')]:
        fig,ax=plt.subplots(figsize=(9,4.5))
        for m in ['Original metadata']+methods:
            ax.plot(list(cases),[r[key] for r in summary if r['method']==m],'o-',label=m)
        ax.set(title=title,xlabel='Experiment');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/('roi_'+key+'.png'),dpi=160);plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--patients',nargs='+',default=['patient_04']);p.add_argument('--tests',nargs='+',choices=TESTS,default=['T0','C1','C2','C3'])
    p.add_argument('--top-k',nargs='+',type=int,choices=range(1,8),default=[2,3,7]);p.add_argument('--no-roi',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test: self_test();return
    import SimpleITK as sitk
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    import register_petct as base
    import evaluate_seg_roi as roi
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    root=args.project.resolve();ks=sorted(set(args.top_k));tests=list(dict.fromkeys(['T0']+args.tests));methods=['Baseline','Rigid MI','Multi-resolution MI']+[method_name(k) for k in ks]
    for patient in args.patients:
        study=roi.locate_study(root/'data'/patient);results=root/'results'/patient;out=results/'fast_evaluation'/('_'.join('k'+str(k) for k in ks))
        if (out/'fast_summary.csv').exists() and not args.overwrite: raise FileExistsError(f'{out}: existing summary; use --overwrite or a different K list')
        files=['known_fixed_to_moved','rigid_fixed_to_moved','multires_fixed_to_moved'];saved={t:[sitk.ReadTransform(str(results/t/(f+'.tfm'))) for f in files] for t in tests}
        out.mkdir(parents=True,exist_ok=True)
        ct,ch=base.read_series(study,'CT');pet,ph=base.read_series(study,'PT');pet=sitk.Cast(pet,sitk.sitkFloat32)
        for key in ('PatientID','StudyInstanceUID','FrameOfReferenceUID'):
            if not ch.get(key) or str(ch.get(key))!=str(ph.get(key)): raise ValueError('CT/PET metadata mismatch: '+key)
        c=np.asarray(base.center(ct));fixed=sitk.Cast(sitk.Shrink(ct,[2,2,2]),sitk.sitkFloat32);mask=sitk.Cast(fixed>-350,sitk.sitkUInt8);mask=sitk.Cast(sitk.RelabelComponent(sitk.ConnectedComponent(mask))==1,sitk.sitkUInt8)
        probes=c+np.stack(np.meshgrid(*[np.linspace(-v,v,5) for v in (100,100,200)],indexing='ij'),axis=-1).reshape(-1,3)
        pts,a=core.sample_points(fixed,mask,sitk,100000,43);sampler=core.sampler(pet,sitk);orig=sampler(pts,np.eye(4));brange=(0.,max(float(np.percentile(orig[np.isfinite(orig)],99.5)),1e-6))
        rows=[];cases={};t0=None
        for test in tests:
            print(f'\n=== {patient} {test} ===',flush=True)
            known=sitk.Euler3DTransform(saved[test][0]);h=core.matrix(known)
            if not np.allclose(known.GetCenter(),c,atol=.01,rtol=0): raise ValueError('Saved motion center differs from CT center')
            moved=base.make_moved(pet,known);hs=[np.eye(4),core.matrix(saved[test][1]),core.matrix(saved[test][2])];times=[0.]
            with (results/test/'registration_metrics.csv').open(encoding='utf-8-sig',newline='') as f: old={r['method']:float(r['seconds']) for r in csv.DictReader(f)}
            times.extend([old['Rigid MI'],old['Multi-resolution MI']])
            for k in ks:
                tx,elapsed=fast(fixed,moved,mask,c,k,sitk,base,out/test/f'k{k}');hs.append(core.matrix(tx));times.append(elapsed)
            if test=='T0':t0=hs
            cases[test]=(h,hs)
            vals,n=core.score_group(pts,a,sampler,[np.eye(4)]+[np.linalg.inv(h)@v for v in hs],brange,base)
            for m,v,v0,mi,sec in zip(methods,hs,t0,vals[1:],times):
                ab,ar=core.errors(v,h,probes);rel=v@np.linalg.inv(v0);re,rr=core.errors(rel,h,probes)
                rows.append(dict(patient=patient,test=test,method=m,absolute_mean_mm=ab,absolute_rmse_mm=ar,relative_mean_mm=re,relative_rmse_mm=rr,absolute_rotation_error_deg=core.angle_error(v,h),relative_rotation_error_deg=core.angle_error(rel,h),histogram_mi_nats=mi,original_mi_nats=vals[0],common_samples=n,seconds_total=sec,determinant=float(np.linalg.det(v[:3,:3]))))
                print(f'{m:22} abs={ab:.4f} rel={re:.4f} mm MI={mi:.6f} sec={sec:.2f}',flush=True)
        status={'patient':patient,'roi_status':'SKIPPED' if args.no_roi else 'PENDING'}
        if not args.no_roi:
            segments,rois,info=roi.load_seg(study,ct,pet,ch,ph)
            if rois: roi_eval(ct,pet,segments,rois,cases,out,roi,sitk,plt,str(ph.get('Units','UNKNOWN')),methods);status['roi_status']='OK'
            else: status['roi_status']='EMPTY_SEG';print('EMPTY SEG: no ROI metrics',flush=True)
        (out/'roi_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        for key in ['absolute_mean_mm','relative_mean_mm','relative_rotation_error_deg','histogram_mi_nats','seconds_total']:
            fig,ax=plt.subplots(figsize=(9,4.5))
            for m in methods: ax.plot(tests,[r[key] for r in rows if r['method']==m],'o-',label=m)
            ax.set(xlabel='Experiment',ylabel=key);ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/(key+'.png'),dpi=160);plt.close(fig)
        import scipy
        (out/'protocol.json').write_text(json.dumps(dict(patient=patient,tests=tests,angles=ANGLES,top_k=ks,selection='Initial histogram MI screening; retain all raw starts; coarse top K; fine up to 2 distinct poses; common support at each ranking',coarse=[4,2],fine=[2,1],seed_optimizer=42,seed_screen=91,seed_eval=43,threads=1,versions={'SimpleITK':sitk.Version_VersionString(),'numpy':np.__version__,'scipy':scipy.__version__},notes='Known S used only to construct motion and evaluate. Relative uses own T0. Timings of old methods are saved historical timings; new timings include selection and optimization. ROI CNR common support within test.'),indent=2),encoding='utf-8')
        core.write_csv(out/'fast_summary.csv',rows);print('Saved:',out,flush=True)

if __name__=='__main__': main()
