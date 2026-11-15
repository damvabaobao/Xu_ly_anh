"""Compare saved rigid registrations with multi-start rigid and affine refinement.
Windows: python register_petct_improved.py --patient patient_04 --tests T0 C3
Requires existing register_petct.py and fixed evaluate_seg_roi.py in project.
All transforms map fixed CT physical coordinates -> moving PET physical coordinates.
Known motion is used only to reconstruct the simulated input and evaluate results.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np

METHODS = ['Baseline','Rigid MI','Multi-resolution MI','Multi-start Rigid','Rigid + Affine']
ANGLES = [-15,-10,-5,0,5,10,15]


def apply(h,p):
    return np.asarray(p) @ h[:3,:3].T + h[:3,3]


def matrix(tx):
    o=np.asarray(tx.TransformPoint((0.,0.,0.)))
    h=np.eye(4); h[:3,3]=o
    h[:3,:3]=np.column_stack([np.asarray(tx.TransformPoint(tuple(v)))-o for v in np.eye(3)])
    if not np.all(np.isfinite(h)): raise ValueError('Non-finite transform')
    return h


def rotation(a):
    u,s,v=np.linalg.svd(a); r=u@v
    if np.linalg.det(r)<0:
        u[:,-1]*=-1; r=u@v
    return r


def angle_error(h,k):
    r=rotation(h[:3,:3])@rotation(k[:3,:3]).T
    return float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))


def errors(h,k,pts):
    e=np.linalg.norm(apply(h,pts)-apply(k,pts),axis=1)
    return float(e.mean()),float(np.sqrt(np.mean(e**2)))


def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def make_affine(h,c,sitk):
    t=sitk.AffineTransform(3);t.SetCenter(tuple(c));t.SetMatrix(tuple(h[:3,:3].ravel()))
    t.SetTranslation(tuple(apply(h,c)-c));return t


def optimize(fixed,moving,mask,initial,levels,sigmas,rate,iters,sitk,seed=42):
    r=sitk.ImageRegistrationMethod();r.SetMetricAsMattesMutualInformation(32)
    r.SetMetricFixedMask(mask);r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(.20,seed);r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=rate,minStep=.01,numberOfIterations=iters,relaxationFactor=.5,gradientMagnitudeTolerance=1e-6)
    r.SetOptimizerScalesFromPhysicalShift();r.SetShrinkFactorsPerLevel(levels)
    r.SetSmoothingSigmasPerLevel(sigmas);r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(initial,inPlace=True);r.Execute(fixed,moving)
    return initial,str(r.GetOptimizerStopConditionDescription())


def sample_points(fixed,mask,sitk,n=60000,seed=43):
    ids=np.flatnonzero(sitk.GetArrayViewFromImage(mask).ravel())
    rng=np.random.default_rng(seed);idx=rng.choice(ids,min(n,len(ids)),replace=False)
    z,y,x=np.unravel_index(idx,tuple(fixed.GetSize()[::-1]))
    ijk=np.column_stack((x,y,z));d=np.asarray(fixed.GetDirection()).reshape(3,3)
    pts=(ijk*np.asarray(fixed.GetSpacing()))@d.T+np.asarray(fixed.GetOrigin())
    a=sitk.GetArrayViewFromImage(fixed).ravel()[idx].copy()
    return pts,a


def sampler(image,sitk):
    from scipy.ndimage import map_coordinates
    arr=sitk.GetArrayViewFromImage(image);origin=np.asarray(image.GetOrigin())
    d=np.asarray(image.GetDirection()).reshape(3,3);sp=np.asarray(image.GetSpacing())
    def sample(pts,h):
        ijk=((apply(h,pts)-origin)@np.linalg.inv(d).T)/sp
        return map_coordinates(arr,ijk[:,::-1].T,order=1,mode='constant',cval=np.nan,prefilter=False)
    return sample


def score_group(pts,a,sample,hs,brange,base):
    vals=[sample(pts,h) for h in hs]
    valid=np.isfinite(a)&np.logical_and.reduce([np.isfinite(v) for v in vals])
    if valid.sum()<100 or valid.mean()<.5: raise ValueError('Insufficient common PET overlap for candidate comparison')
    return [base.mi(a[valid],v[valid],(-350.,500.),brange) for v in vals],int(valid.sum())


def run_new(fixed,moved,mask,c,sitk,base,out):
    # Selection uses only fixed CT + moved PET, never the known transform or original PET alignment.
    pts,a=sample_points(fixed,mask,sitk,60000,91);sample=sampler(moved,sitk)
    native=sitk.GetArrayViewFromImage(moved).ravel()[::31]
    positive=native[np.isfinite(native)&(native>0)]
    if not len(positive): raise ValueError('PET has no finite positive intensity')
    brange=(0.,max(float(np.percentile(positive,99.5)),1e-6))
    starts=[]; trace=[]; start_time=time.perf_counter()
    for deg in ANGLES:
        t=sitk.Euler3DTransform();t.SetCenter(tuple(c));t.SetRotation(0.,0.,np.deg2rad(deg))
        starts.append((f'initial {deg:+d}',sitk.Euler3DTransform(t)))
        print(f'Coarse start Z={deg:+d} deg',flush=True)
        try:
            t,stop=optimize(fixed,moved,mask,t,[4,2],[4,2],2.,120,sitk)
            starts.append((f'coarse {deg:+d}',t));trace.append(dict(stage='coarse',start_deg=deg,status=stop))
        except RuntimeError as e:
            trace.append(dict(stage='coarse',start_deg=deg,status=str(e)));print('  failed:',str(e).splitlines()[-1])
    scores,n=score_group(pts,a,sample,[matrix(t) for _,t in starts],brange,base)
    order=np.argsort(scores)[::-1];rankrows=[]
    for j in order:
        rankrows.append(dict(stage='coarse ranking',candidate=starts[j][0],histogram_mi=scores[j],common_samples=n))
    # Retain best starting candidate and refine the best two distinct poses.
    chosen=[]
    for j in order:
        h=matrix(starts[j][1])
        if all(np.linalg.norm(apply(h,c)-apply(matrix(t),c))>.25 or angle_error(h,matrix(t))>.25 for _,t in chosen):
            chosen.append(starts[j])
        if len(chosen)==2: break
    finalists=[(f'pre-fine {name}',sitk.Euler3DTransform(t)) for name,t in chosen]
    for name,t in chosen:
        print('Fine refinement:',name,flush=True)
        try:
            r,stop=optimize(fixed,moved,mask,sitk.Euler3DTransform(t),[2,1],[2,0],1.,200,sitk)
            finalists.append((f'fine {name}',r));trace.append(dict(stage='fine',start_deg=name,status=stop))
        except RuntimeError as e: trace.append(dict(stage='fine',start_deg=name,status=str(e)))
    fs,n=score_group(pts,a,sample,[matrix(t) for _,t in finalists],brange,base)
    for (name,t),v in zip(finalists,fs): rankrows.append(dict(stage='fine ranking',candidate=name,histogram_mi=v,common_samples=n))
    best=int(np.argmax(fs));rigid=sitk.Euler3DTransform(finalists[best][1]);dt=time.perf_counter()-start_time
    print('Selected:',finalists[best][0], 'MI:',fs[best],flush=True)
    sitk.WriteTransform(rigid,str(out/'multistart_fixed_to_moved.tfm'))
    write_csv(out/'candidate_scores.csv',rankrows);write_csv(out/'optimizer_log.csv',trace)
    t0=time.perf_counter();aff=make_affine(matrix(rigid),c,sitk)
    print('Affine refinement from multi-start rigid',flush=True)
    aff,stop=optimize(fixed,moved,mask,aff,[2,1],[2,0],.5,200,sitk)
    dt_aff=time.perf_counter()-t0;h=matrix(aff);sv=np.linalg.svd(h[:3,:3],compute_uv=False)
    if np.linalg.det(h[:3,:3])<=0 or sv.min()<1e-6:
        raise ValueError('Affine became non-invertible or orientation-reversing; inspect optimization')
    trace.append(dict(stage='affine',start_deg='multi-start',status=stop))
    write_csv(out/'candidate_scores.csv',rankrows);write_csv(out/'optimizer_log.csv',trace)
    return rigid,aff,dt,dt_aff


def roi_eval(ct,pet,segments,rois,cases,out,roi,sitk,plt,units):
    rows=[]
    for j,r in enumerate(sorted(rois,key=lambda r:r['volume_ml'],reverse=True)):
        c=r['center'];ref=roi.make_ref(ct,r['corners'],17);ctarr=sitk.GetArrayFromImage(ref)
        target,bg=roi.regions(r,segments,ref,5,15,ctarr>-350)
        # Per test same physical CT-grid target/background and common support for every method.
        for test,(known,hs) in cases.items():
            maps=[('Original metadata',np.eye(4),np.eye(4))]+[(m,known,h) for m,h in zip(METHODS,hs)]
            transforms=[np.linalg.inv(k)@h for m,k,h in maps]
            vals=[roi.resample(pet,ref,h) for h in transforms];common=np.logical_and.reduce([np.isfinite(v) for v in vals]);tar=target&common;back=bg&common
            for (m,k,h),v in zip(maps,vals):
                q=apply(np.linalg.inv(h)@k,c);e=q-c;cnr,mean,peak,bmean,bsd=roi.cnr_values(v,tar,back)
                # Exact continuous volume ratio for affine output (not a voxel-count approximation).
                ratio=abs(np.linalg.det((np.linalg.inv(h)@k)[:3,:3]));
                rows.append(dict(test=test,method=m,roi_id=r['id'],centroid_error_mm=float(np.linalg.norm(e)),dx_mm=e[0],dy_mm=e[1],dz_mm=e[2],cnr=cnr,pet_mean=mean,pet_max=peak,pet_units=units,volume_ml=r['volume_ml'],output_volume_ml=r['volume_ml']*ratio,volume_ratio=ratio,target_n=int(tar.sum()),background_n=int(back.sum())))
            if j==0:
                z=int(np.clip(round(ref.TransformPhysicalPointToContinuousIndex(tuple(c))[2]),0,ctarr.shape[0]-1));fig,axs=plt.subplots(2,3,figsize=(14,9));positive=vals[0][np.isfinite(vals[0])&(vals[0]>0)];vmax=max(float(np.percentile(positive,99)) if len(positive) else 1.,1e-6)
                for ax,(m,k,h),v,tr in zip(axs.ravel(),maps,vals,transforms):
                    ax.imshow(ctarr[z],cmap='gray',vmin=-200,vmax=250);ax.imshow(np.ma.masked_where(~np.isfinite(v[z])|(v[z]<=.05*vmax),v[z]),cmap='hot',vmin=0,vmax=vmax,alpha=.5)
                    mask=roi.resample(r['mask'],ref,tr,mask=True)>0
                    if target[z].any() and not target[z].all():ax.contour(target[z],levels=[.5],colors='lime')
                    if mask[z].any() and not mask[z].all():ax.contour(mask[z],levels=[.5],colors='cyan',linestyles='dashed')
                    ax.set_title(m);ax.axis('off')
                fig.suptitle(f'{test} | {r["id"]} | fixed SEG green, transformed SEG cyan');fig.tight_layout();fig.savefig(out/f'{test}_roi_overlay.png',dpi=160);plt.close(fig)
        print('ROI:',r['id'],flush=True)
    write_csv(out/'roi_metrics.csv',rows);summary=[]
    for test in cases:
        for m in ['Original metadata']+METHODS:
            rr=[r for r in rows if r['test']==test and r['method']==m];v=np.asarray([r['cnr'] for r in rr]);v=v[np.isfinite(v)]
            summary.append(dict(test=test,method=m,n_roi=len(rr),mean_centroid_error_mm=np.mean([r['centroid_error_mm'] for r in rr]),max_centroid_error_mm=np.max([r['centroid_error_mm'] for r in rr]),median_cnr=np.median(v) if len(v) else np.nan,n_valid_cnr=len(v),mean_volume_ratio=np.mean([r['volume_ratio'] for r in rr])))
    write_csv(out/'roi_summary.csv',summary)
    for key,title in [('mean_centroid_error_mm','Mean SEG centroid error (mm)'),('median_cnr','Median CNR on fixed ROI'),('mean_volume_ratio','Continuous ROI output / original volume')]:
        fig,ax=plt.subplots(figsize=(9,4.5))
        for m in ['Original metadata']+METHODS:
            ax.plot(list(cases),[r[key] for r in summary if r['method']==m],'o-',label=m)
        ax.set(title=title,xlabel='Experiment');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/('roi_'+key+'.png'),dpi=160);plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--patient',default='patient_04');p.add_argument('--tests',nargs='+',default=['T0','C3'],choices=['T0','T1','T2','T3','C1','C2','C3']);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--no-roi',action='store_true');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:
        h=np.eye(4);h[:3,3]=[5,-3,2];k=np.eye(4);k[:3,3]=[1,2,3];pts=np.array([[1,2,3],[8,9,10]])
        th=np.deg2rad(10);h[:3,:3]=[[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]]
        assert np.allclose(apply((h@k)@np.linalg.inv(k),pts),apply(h,pts));assert errors(h,h,pts)==(0.,0.);assert abs(angle_error(h,k)-10)<1e-8
        a=np.eye(4);a[:3,:3]=h[:3,:3]@np.diag([1.1,.95,1.]);assert abs(angle_error(a,h))<1e-5
        assert np.allclose(apply(np.linalg.inv(h)@h,pts),pts)
        assert np.isclose(1/np.linalg.det(a[:3,:3]),1/(1.1*.95))
        print('SELF-TEST PASSED (matrix composition, rotation, affine volume)');return
    import SimpleITK as sitk
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    import register_petct as base
    import evaluate_seg_roi as roi
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    root=args.project.resolve();study=roi.locate_study(root/'data'/args.patient);results=root/'results'/args.patient
    out=results/'improved_evaluation';out.mkdir(parents=True,exist_ok=True)
    tests=list(dict.fromkeys(['T0']+args.tests));files=['known_fixed_to_moved','rigid_fixed_to_moved','multires_fixed_to_moved']
    saved={t:[sitk.ReadTransform(str(results/t/(f+'.tfm'))) for f in files] for t in tests}
    ct,ch=base.read_series(study,'CT');pet,ph=base.read_series(study,'PT');pet=sitk.Cast(pet,sitk.sitkFloat32)
    for key in ('PatientID','StudyInstanceUID','FrameOfReferenceUID'):
        if not ch.get(key) or str(ch.get(key))!=str(ph.get(key)):raise ValueError('CT/PET metadata mismatch: '+key)
    c=np.asarray(base.center(ct));fixed=sitk.Cast(sitk.Shrink(ct,[2,2,2]),sitk.sitkFloat32);mask=sitk.Cast(fixed>-350,sitk.sitkUInt8);mask=sitk.Cast(sitk.RelabelComponent(sitk.ConnectedComponent(mask))==1,sitk.sitkUInt8)
    probes=c+np.stack(np.meshgrid(*[np.linspace(-v,v,5) for v in (100,100,200)],indexing='ij'),axis=-1).reshape(-1,3)
    pts,a=sample_points(fixed,mask,sitk,100000,43);original_sample=sampler(pet,sitk);orig=original_sample(pts,np.eye(4));brange=(0.,max(float(np.percentile(orig[np.isfinite(orig)],99.5)),1e-6))
    rows=[];cases={};t0=None
    for test in tests:
        print('\n===',args.patient,test,'===',flush=True);case=out/test;case.mkdir(exist_ok=True)
        known=sitk.Euler3DTransform(saved[test][0]);k=matrix(known)
        if not np.allclose(known.GetCenter(),c,atol=.01,rtol=0):raise ValueError('Saved transform center does not match CT')
        moved=base.make_moved(pet,known);r,aff,dt,da=run_new(fixed,moved,mask,c,sitk,base,case)
        sitk.WriteTransform(r,str(case/'multistart_fixed_to_moved.tfm'));sitk.WriteTransform(aff,str(case/'affine_fixed_to_moved.tfm'))
        hs=[np.eye(4),matrix(saved[test][1]),matrix(saved[test][2]),matrix(r),matrix(aff)];cases[test]=(k,hs)
        if test=='T0':t0=hs
        # Evaluate all methods on common original-PET samples; map CT -> original PET via S^-1 T.
        values,n=score_group(pts,a,original_sample,[np.eye(4)]+[np.linalg.inv(k)@h for h in hs],brange,base)
        old={}
        with (results/test/'registration_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
            old={x['method']:float(x['seconds']) for x in csv.DictReader(f)}
        times=[0,old['Rigid MI'],old['Multi-resolution MI'],dt,dt+da]
        for m,h,h0,mi,seconds in zip(METHODS,hs,t0,values[1:],times):
            ab,ar=errors(h,k,probes);rel=h@np.linalg.inv(h0);re,rr=errors(rel,k,probes);sv=np.linalg.svd(h[:3,:3],compute_uv=False);det=float(np.linalg.det(h[:3,:3]))
            rows.append(dict(test=test,method=m,absolute_mean_mm=ab,absolute_rmse_mm=ar,relative_mean_mm=re,relative_rmse_mm=rr,absolute_rotation_error_deg=angle_error(h,k),relative_rotation_error_deg=angle_error(rel,k),histogram_mi_nats=mi,original_mi_nats=values[0],common_samples=n,seconds_total=seconds,determinant=det,singular_max=sv[0],singular_mid=sv[1],singular_min=sv[2],output_volume_ratio=1/abs(det)))
            print(f'{m:22} abs={ab:.3f} rel={re:.3f} mm MI={mi:.6f} det={det:.5f}',flush=True)
        del moved
    write_csv(out/'improved_summary.csv',rows)
    if not args.no_roi:
        segments,rois,info=roi.load_seg(study,ct,pet,ch,ph)
        if rois: roi_eval(ct,pet,segments,rois,cases,out,roi,sitk,plt,str(ph.get('Units','UNKNOWN')))
        else: print('EMPTY SEG: no ROI metrics')
    for key,title in [('absolute_mean_mm','Absolute probe error (mm)'),('relative_mean_mm','T0-relative probe error (mm)'),('histogram_mi_nats','Histogram MI'),('relative_rotation_error_deg','T0-relative rotation error (deg)')]:
        fig,ax=plt.subplots(figsize=(9,4.5))
        for m in METHODS:ax.plot(tests,[r[key] for r in rows if r['method']==m],'o-',label=m)
        ax.set(title=title,xlabel='Experiment');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/(key+'.png'),dpi=160);plt.close(fig)
    (out/'protocol.json').write_text(json.dumps(dict(angles=ANGLES,selection='Common-sample histogram MI of moved PET and CT; no ground truth used for candidate selection',coarse=[4,2],fine=[2,1],affine='unconstrained local refinement; no guarantee of anatomical plausibility',rotation='proper polar rotation for affine',relative='Ts @ inverse(T0), each method own T0',runtime='Affine total includes multi-start; existing times from saved runs',roi='Analytic centroid and volume ratio; CNR common voxels per test, not across all tests; no clinical TRE or SUV',probe_half_extents_mm=[100,100,200]),indent=2),encoding='utf-8')
    print('Saved:',out)

if __name__=='__main__':main()
