"""PET/CT ROI evaluation from binary DICOM SEG and saved fixed->moving transforms.
Run from PET project: python evaluate_seg_roi.py --patient patient_01 --no-show
Requires SimpleITK, pydicom, numpy, scipy, matplotlib. No registration is repeated.
SEG components are research annotations, not independently verified anatomical landmarks.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import traceback
import numpy as np
try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

METHODS = ('Original metadata', 'Misaligned baseline', 'Rigid MI', 'Multi-resolution MI')
COLORS = ('gray', 'tab:blue', 'tab:orange', 'tab:green')
FILES = ('known_fixed_to_moved.tfm', 'rigid_fixed_to_moved.tfm', 'multires_fixed_to_moved.tfm')


def apply(h, p):
    return np.asarray(p) @ h[:3,:3].T + h[:3,3]


def matrix(tx):
    o = np.asarray(tx.TransformPoint((0.,0.,0.)))
    r = np.column_stack([np.asarray(tx.TransformPoint(tuple(v)))-o for v in np.eye(3)])
    if not np.allclose(r.T@r,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(r),1.,atol=1e-7):
        raise ValueError('Expected rigid 3D transform')
    h=np.eye(4); h[:3,:3]=r; h[:3,3]=o
    return h


def roi_position(known, estimated, center):
    # original PET -> synthetically moved PET -> output CT coordinates.
    return apply(np.linalg.inv(estimated) @ known, center)


def cnr_values(a, target, background):
    t=a[target]; b=a[background]
    if len(t)<5 or len(b)<20:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    tm=float(t.mean()); bm=float(b.mean()); sd=float(b.std(ddof=1))
    return (abs(tm-bm)/sd if sd>0 else np.nan, tm, float(t.max()), bm, sd)


def read_series(root, modality):
    import SimpleITK as sitk
    import pydicom
    folders=[p for p in root.iterdir() if p.is_dir() and p.name.upper().startswith(modality+'_')]
    if len(folders)!=1: raise ValueError(f'{root}: expected exactly one {modality}_ folder')
    ids=sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folders[0]))
    if not ids or len(ids)!=1: raise ValueError(f'{folders[0]}: expected one DICOM series')
    names=sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folders[0]),ids[0])
    hdr=pydicom.dcmread(names[0],stop_before_pixels=True)
    if hdr.Modality!=modality: raise ValueError(f'Wrong modality: {hdr.Modality}')
    reader=sitk.ImageSeriesReader(); reader.SetFileNames(names)
    return reader.Execute(),hdr


def fg_value(frame, shared, seq, key):
    for group in (frame,shared):
        if group is not None and seq in group:
            item=group[seq].value[0]
            if key in item: return item.get(key)
    return None


def load_seg(root, ct, pet, ch, ph):
    """Reconstruct cropped native SEG masks from per-frame patient geometry.
    Sparse/out-of-order frames supported; missing planes are zero. Reject unsupported geometry.
    """
    import pydicom
    import SimpleITK as sitk
    folders=[p for p in root.iterdir() if p.is_dir() and p.name.upper().startswith('SEG_')]
    if len(folders)!=1: raise ValueError(f'{root}: expected one SEG_ folder')
    candidates=[]
    for path in folders[0].rglob('*'):
        if not path.is_file(): continue
        try: hdr=pydicom.dcmread(path,stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError,OSError): continue
        if hdr.get('Modality')=='SEG': candidates.append(path)
    if len(candidates)!=1: raise ValueError(f'Expected one DICOM SEG object, found {len(candidates)}')
    ds=pydicom.dcmread(candidates[0])
    print('SEG file:',candidates[0].name,flush=True)
    print('SEG type:',ds.get('SegmentationType','missing'),'frames:',ds.get('NumberOfFrames',1),'matrix:',ds.get('Rows','?'),ds.get('Columns','?'),flush=True)
    for k in ('PatientID','StudyInstanceUID','FrameOfReferenceUID'):
        if not ds.get(k) or str(ds.get(k))!=str(ph.get(k)):
            raise ValueError(f'SEG/PET {k} missing or mismatched; cannot assume alignment')
    if ds.get('SegmentationType')!='BINARY': raise ValueError('Only binary DICOM SEG supported; fractional/labelmap requires separate handling')
    refs={str(i.SeriesInstanceUID) for i in ds.get('ReferencedSeriesSequence',[]) if i.get('SeriesInstanceUID')}
    if refs and not refs.intersection({str(ch.SeriesInstanceUID),str(ph.SeriesInstanceUID)}):
        raise ValueError('SEG does not reference the loaded CT/PT series')
    shared=ds.SharedFunctionalGroupsSequence[0] if ds.get('SharedFunctionalGroupsSequence') else None
    frames=ds.get('PerFrameFunctionalGroupsSequence')
    if not frames: raise ValueError('SEG missing PerFrameFunctionalGroupsSequence; cannot establish frame geometry')
    if len(frames)!=int(ds.get('NumberOfFrames',1)):
        raise ValueError('SEG functional group count does not match NumberOfFrames')
    if not frames: raise ValueError('SEG has no frames')
    # pydicom 3 can decode one frame at a time, avoiding a full dense SEG copy.
    # Use sparse nonzero coordinates only, preserving frame geometry.
    nonzero={}
    try:
        try:
            from pydicom.pixels import iter_pixels
        except ImportError:
            iter_pixels=None
        if iter_pixels is None:
            pix=ds.pixel_array
            if pix.ndim==2: pix=pix[None,...]
            decoded=iter(pix)
        else:
            decoded=iter_pixels(str(candidates[0]))
        ndecoded=0
        for i,frame_pixels in enumerate(decoded):
            ndecoded+=1
            if frame_pixels.shape!=(int(ds.Rows),int(ds.Columns)):
                raise ValueError(f'Unexpected SEG frame shape: {frame_pixels.shape}')
            yy,xx=np.nonzero(frame_pixels)
            if len(xx): nonzero[i]=(yy.astype(np.int32),xx.astype(np.int32))
        if ndecoded!=len(frames): raise ValueError('Unexpected SEG pixel/frame count')
    except (RuntimeError,NotImplementedError,AttributeError,ImportError) as exc:
        syntax=str(ds.file_meta.get('TransferSyntaxUID','unknown'))
        raise RuntimeError(f'Cannot decode SEG pixels (TransferSyntaxUID={syntax}). Try python -m pip install -U pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg; original error: {exc}') from exc
    if not nonzero:
        print('SEG contains no nonzero voxels. No annotated ROI to evaluate.')
        return [],[],{'seg_file':candidates[0].name,'seg_type':'BINARY','nonempty_rois':0,'status':'EMPTY_SEG','segments':len(ds.get('SegmentSequence',[]))}
    pos=[]; nums=[]; orient=[]; spacings=[]; zmeta=[]
    for f in frames:
        pp=fg_value(f,shared,'PlanePositionSequence','ImagePositionPatient')
        oo=fg_value(f,shared,'PlaneOrientationSequence','ImageOrientationPatient')
        sp=fg_value(f,shared,'PixelMeasuresSequence','PixelSpacing')
        zz=fg_value(f,shared,'PixelMeasuresSequence','SpacingBetweenSlices')
        sn=fg_value(f,shared,'SegmentIdentificationSequence','ReferencedSegmentNumber')
        if any(v is None for v in (pp,oo,sp,sn)): raise ValueError('SEG frame missing position/orientation/spacing/segment ID')
        pos.append(np.asarray(pp,float)); orient.append(np.asarray(oo,float)); spacings.append(np.asarray(sp,float)); nums.append(int(sn))
        if zz is not None: zmeta.append(abs(float(zz)))
    pos=np.asarray(pos); orient=np.asarray(orient); spacings=np.asarray(spacings); nums=np.asarray(nums)
    if not np.allclose(orient,orient[0],atol=1e-5,rtol=0) or not np.allclose(spacings,spacings[0],atol=1e-5,rtol=0):
        raise ValueError('SEG has varying orientation or in-plane spacing; unsupported')
    x,y=orient[0,:3],orient[0,3:]; z=np.cross(x,y); direction=np.column_stack((x,y,z))
    if not np.allclose(direction.T@direction,np.eye(3),atol=1e-5): raise ValueError('Invalid SEG orientation')
    if not np.isfinite(pos).all() or not np.isfinite(spacings).all() or np.min(spacings)<=0: raise ValueError('Invalid SEG geometry')
    plane=pos@z; origin=pos[np.argmin(plane)]; rel=pos-origin
    # A regular SEG volume needs consistent in-plane origins, no gantry tilt/drift.
    if np.max(np.abs(rel@x))>.05 or np.max(np.abs(rel@y))>.05: raise ValueError('SEG frames have varying in-plane origins')
    def valid_dz(d):
        if not np.isfinite(d) or d<=0: return False
        q=(plane-plane.min())/d
        return np.max(np.abs(q-np.rint(q))*d)<.05
    if zmeta:
        dz=zmeta[0]
        if not np.allclose(zmeta,dz,atol=.01,rtol=0) or not valid_dz(dz): raise ValueError('SEG slice spacing inconsistent with frame positions')
        zsource='SEG SpacingBetweenSlices'
    else:
        # Prefer the referenced series' physical slice interval, not SliceThickness.
        opts=[]
        for name,img,hdr in [('CT',ct,ch),('PT',pet,ph)]:
            d=np.asarray(img.GetDirection()).reshape(3,3)
            if abs(float(d[:,2]@z))<.99999: continue
            step=img.GetSpacing()[2]
            q=(pos-np.asarray(img.GetOrigin()))@d[:,2]/step
            if valid_dz(step) and np.max(np.abs(q-np.rint(q))*step)<.05:
                xy_match=np.allclose(img.GetSpacing()[:2],spacings[0,::-1],atol=.01,rtol=0)
                priority=(str(hdr.SeriesInstanceUID) in refs, bool(xy_match))
                opts.append((priority,step,name))
        if not opts: raise ValueError('Cannot establish SEG slice spacing from metadata/source geometry; do not guess')
        opts.sort(reverse=True); best=opts[0]
        if any(a[0]==best[0] and abs(a[1]-best[1])>.01 for a in opts): raise ValueError('Ambiguous SEG slice spacing')
        _,dz,name=best; zsource=f'{name} source grid'
    zi=np.rint((plane-plane.min())/dz).astype(int)
    spacing=np.array([spacings[0,1],spacings[0,0],dz])
    labels={int(s.SegmentNumber):str(s.SegmentLabel) for s in ds.SegmentSequence}
    if not set(nums).issubset(labels): raise ValueError('SEG frame refers to undefined segment')
    segments=[]; rois=[]
    for sn,label in labels.items():
        idx=np.flatnonzero(nums==sn)
        if len(set(zi[idx]))!=len(idx): raise ValueError(f'Duplicate frames at same plane for segment {sn}')
        nonempty=[i for i in idx if i in nonzero]
        if not nonempty: continue
        # Crop to nonzero extent (retain physical origin); avoid whole-body label arrays.
        bounds=[]
        for i in nonempty:
            yy,xx=nonzero[i]; bounds.append((xx.min(),xx.max(),yy.min(),yy.max(),zi[i]))
        b=np.asarray(bounds); lo=np.array([b[:,0].min(),b[:,2].min(),b[:,4].min()]); hi=np.array([b[:,1].max(),b[:,3].max(),b[:,4].max()])
        shape=hi-lo+1
        if np.prod(shape)>250_000_000: raise ValueError('SEG crop too large; check geometry')
        arr=np.zeros(tuple(shape[::-1]),np.uint8)
        for i in nonempty:
            yy,xx=nonzero[i]; arr[zi[i]-lo[2],yy-lo[1],xx-lo[0]]=1
        mask=sitk.GetImageFromArray(arr); mask.SetSpacing(tuple(spacing)); mask.SetDirection(tuple(direction.ravel())); mask.SetOrigin(tuple(origin+direction@(lo*spacing)))
        segments.append(mask)
        cc,n=ndi.label(arr,structure=np.ones((3,3,3),np.uint8))
        for k,sl in enumerate(ndi.find_objects(cc),1):
            if sl is None: continue
            component=(cc[sl]==k); count=int(component.sum()); start=np.array([s.start for s in sl][::-1]); stop=np.array([s.stop-1 for s in sl][::-1])
            local=np.array(ndi.center_of_mass(component))[::-1]+start
            c=np.asarray(mask.TransformContinuousIndexToPhysicalPoint(tuple(local)))
            cm=sitk.GetImageFromArray(component.astype(np.uint8)); cm.SetSpacing(tuple(spacing)); cm.SetDirection(tuple(direction.ravel())); cm.SetOrigin(mask.TransformIndexToPhysicalPoint(tuple(int(v) for v in start)))
            corners=np.array([mask.TransformIndexToPhysicalPoint(tuple(int(v) for v in ijk)) for ijk in itertools.product(*zip(start,stop))])
            rois.append(dict(id=f'S{sn:02d}_C{k:03d}',segment=sn,label=label,mask=cm,center=c,corners=corners,volume_ml=count*np.prod(spacing)/1000.,voxels=count))
    print(f'SEG: {len(frames)} frames, {len(labels)} segment(s), {len(rois)} connected ROI(s); spacing {spacing} mm ({zsource})')
    info={'seg_file':candidates[0].name,'seg_spacing_mm':spacing.tolist(),'z_spacing_source':zsource,'segments':len(labels),'nonempty_rois':len(rois),'seg_referenced_loaded_series':bool(refs),'seg_type':'BINARY'}
    return segments,rois,info


def make_ref(img,corners,pad):
    import SimpleITK as sitk
    pts=np.array([img.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in corners]); spacing=np.array(img.GetSpacing())
    # Use voxel half extents + shell padding; clip to image FOV.
    lo=np.maximum(0,np.floor(pts.min(0)-pad/spacing-2).astype(int)); hi=np.minimum(np.array(img.GetSize())-1,np.ceil(pts.max(0)+pad/spacing+2).astype(int))
    if np.any(hi<lo): raise ValueError('ROI outside reference grid')
    return sitk.RegionOfInterest(img,[int(v) for v in hi-lo+1],[int(v) for v in lo])


def resample(image,ref,h=None,mask=False):
    import SimpleITK as sitk
    tx=sitk.Transform(3,sitk.sitkIdentity)
    if h is not None:
        tx=sitk.AffineTransform(3); tx.SetMatrix(tuple(h[:3,:3].ravel())); tx.SetTranslation(tuple(h[:3,3]))
    return sitk.GetArrayFromImage(sitk.Resample(image,ref,tx,sitk.sitkNearestNeighbor if mask else sitk.sitkLinear,0 if mask else float('nan'),sitk.sitkUInt8 if mask else sitk.sitkFloat32))


def regions(roi,segments,ref,inner,outer,body):
    target=resample(roi['mask'],ref,mask=True)>0
    if not target.any(): return target,np.zeros_like(target)
    dist=ndi.distance_transform_edt(~target,sampling=ref.GetSpacing()[::-1])
    all_seg=np.zeros_like(target)
    for seg in segments: all_seg|=resample(seg,ref,mask=True)>0
    bg=(dist>=inner)&(dist<=outer)&~all_seg&body
    return target,bg


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_tests(root,tests):
    import SimpleITK as sitk
    data={}
    for t in tests:
        folder=root/t
        if not all((folder/f).is_file() for f in FILES): raise ValueError(f'{folder}: missing .tfm; run registration first')
        known,rigid,multi=[matrix(sitk.ReadTransform(str(folder/f))) for f in FILES]
        data[t]=(known,[np.eye(4),np.eye(4),rigid,multi])
    return data


def plot_results(rows,out,tests,show):
    import matplotlib.pyplot as plt
    # Medians over ROIs, not independent patients. Plot only valid CNRs.
    def summarize(test,m,key):
        v=np.array([r[key] for r in rows if r['test']==test and r['method']==m],float); v=v[np.isfinite(v)]
        return np.median(v) if len(v) else np.nan
    for key,title,y,name in [('centroid_error_mm','SEG centroid recovery (median across ROIs)','Error (mm)','01_roi_centroid_error.png'),('cnr','Fixed ROI CNR on CT grid (median across ROIs)','CNR','02_roi_cnr.png'),('pet_mean','Fixed ROI PET mean (median across ROIs)','PET intensity (native units)','03_roi_pet_mean.png')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for m,c in zip(METHODS,COLORS): ax.plot(tests,[summarize(t,m,key) for t in tests],'o-',color=c,label=m)
        ax.set(title=title,ylabel=y,xlabel='Experiment');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/name,dpi=180)
        if not show: plt.close(fig)
    if show: plt.show()


def run(args):
    import SimpleITK as sitk
    import matplotlib
    if args.no_show: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    base=Path(__file__).resolve().parent; root=args.data_root/args.patient; result=args.results_root/args.patient
    out=args.out or result/('roi_check' if args.check_only else 'roi_evaluation');out.mkdir(parents=True,exist_ok=True)
    root=locate_study(root)
    print('Reading study:',root,flush=True)
    ct,ch=read_series(root,'CT');pet,ph=read_series(root,'PT')
    print('CT size/spacing:',ct.GetSize(),ct.GetSpacing(),flush=True)
    print('PT size/spacing:',pet.GetSize(),pet.GetSpacing(),flush=True)
    for key in ('PatientID','StudyInstanceUID','FrameOfReferenceUID'):
        if not ch.get(key) or str(ch.get(key))!=str(ph.get(key)): raise ValueError(f'CT/PT {key} missing or mismatched')
    pet=sitk.Cast(pet,sitk.sitkFloat32); units=str(ph.get('Units','UNKNOWN'))
    print('PET units:',units)
    segments,rois,info=load_seg(root,ct,pet,ch,ph)
    rois.sort(key=lambda r: r['volume_ml'],reverse=True)
    if not rois:
        info.update({'patient':args.patient,'pet_units':units,'status':'EMPTY_SEG'})
        (out/'roi_status.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
        print('EMPTY SEG: no ROI metrics generated. Try another patient. Status:',out/'roi_status.json')
        if (out/'roi_metrics.csv').exists(): print('WARNING: existing ROI metrics are from an older run, not this empty SEG check.')
        return
    if args.check_only:
        info.update({'patient':args.patient,'pet_units':units,'status':'CHECK_OK','rois':[{'roi_id':r['id'],'volume_ml':float(r['volume_ml']),'center_lps_mm':r['center'].tolist()} for r in rois]})
        (out/'roi_status.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
        print('CHECK PASSED: SEG decoded; ROI count:',len(rois),'No transforms evaluated.')
        return
    tests=load_tests(result,args.tests)
    # Verify these saved tests belong to this CT geometry (center of known transform).
    expected_center=np.asarray(ct.TransformContinuousIndexToPhysicalPoint(tuple((n-1)/2 for n in ct.GetSize())))
    for t in args.tests:
        tx=sitk.ReadTransform(str(result/t/FILES[0]))
        if tx.GetName()!='Euler3DTransform' or not np.allclose(sitk.Euler3DTransform(tx).GetCenter(),expected_center,atol=.01,rtol=0):
            raise ValueError(f'{t}: known transform center does not match loaded CT; check patient/results')
        protocol={'T0':((0,0,0),0),'T1':((2,0,0),0),'T2':((5,0,0),0),'T3':((10,0,0),0),'C1':((5,-3,2),3),'C2':((5,-3,2),5),'C3':((5,-3,2),10)}
        if t in protocol:
            shift,angle=protocol[t]; e=sitk.Euler3DTransform(tx)
            if not np.allclose(e.GetTranslation(),shift,atol=1e-6,rtol=0) or not np.allclose(np.degrees([e.GetAngleX(),e.GetAngleY(),e.GetAngleZ()]),[0,0,angle],atol=1e-6,rtol=0):
                raise ValueError(f'{t}: saved known motion does not match the experiment protocol')
    rows=[]; native_rows=[]
    for j,roi in enumerate(rois):
        c=roi['center']; ref=make_ref(ct,roi['corners'],args.outer+2)
        ctarr=sitk.GetArrayFromImage(ref)
        target,bg=regions(roi,segments,ref,args.inner,args.outer,ctarr>-350)
        # All methods and experiments use same physical fixed ROI/background and common valid samples.
        maps=[]
        original=resample(pet,ref)
        common=np.isfinite(original)
        for t,(known,estimated) in tests.items():
            for m,est in zip(METHODS,estimated):
                h=np.eye(4) if m==METHODS[0] else np.linalg.inv(known)@est
                a=original if m==METHODS[0] else resample(pet,ref,h)
                common &= np.isfinite(a)
                maps.append((t,m,known,est,h))
                if m!=METHODS[0]: del a
        tar=target&common; back=bg&common
        # Native PET CNR is a separate grid comparison, not identical voxel sampling.
        pref=make_ref(pet,roi['corners'],args.outer+2)
        # CT body definition sampled in physical space on native PET grid.
        ct_on_pet=resample(ct,pref)
        nt,nb=regions(roi,segments,pref,args.inner,args.outer,np.isfinite(ct_on_pet)&(ct_on_pet>-350))
        nat=sitk.GetArrayFromImage(pref); nt&=np.isfinite(nat); nb&=np.isfinite(nat)
        nv=cnr_values(nat,nt,nb); ov=cnr_values(original,tar,back)
        native_rows.append(dict(roi_id=roi['id'],native_pet_cnr=nv[0],ct_grid_original_cnr=ov[0],native_pet_mean=nv[1],ct_grid_original_mean=ov[1],native_pet_max=nv[2],ct_grid_original_max=ov[2],native_target_n=int(nt.sum()),native_background_n=int(nb.sum()),ct_target_n=int(tar.sum()),ct_background_n=int(back.sum()),pet_units=units))
        for t,m,known,est,h in maps:
            a=original if m==METHODS[0] else resample(pet,ref,h)
            q=c if m==METHODS[0] else roi_position(known,est,c); e=q-c
            cnr,mean,peak,bmean,bsd=cnr_values(a,tar,back)
            status='OK' if np.isfinite(cnr) else 'CNR unavailable: too few common voxels or zero background SD'
            rows.append(dict(test=t,method=m,roi_id=roi['id'],segment_number=roi['segment'],segment_label=roi['label'],native_roi_voxels=roi['voxels'],volume_ml=roi['volume_ml'],ref_x_mm=c[0],ref_y_mm=c[1],ref_z_mm=c[2],output_x_mm=q[0],output_y_mm=q[1],output_z_mm=q[2],residual_x_mm=e[0],residual_y_mm=e[1],residual_z_mm=e[2],centroid_error_mm=np.linalg.norm(e),cnr=cnr,pet_mean=mean,pet_max=peak,background_mean=bmean,background_sd=bsd,target_n=int(tar.sum()),background_n=int(back.sum()),pet_units=units,cnr_status=status))
        print(f"{roi['id']}: volume={roi['volume_ml']:.3f} mL; common target={tar.sum()}, background={back.sum()}")
        # One representative ROI preview (largest chosen below) using fixed CT crop, actual SEG contours.
        if j==0:
            chosen=args.tests[-1]; fig,axs=plt.subplots(1,4,figsize=(15,4))
            z=int(np.clip(round(ref.TransformPhysicalPointToContinuousIndex(tuple(c))[2]),0,ctarr.shape[0]-1))
            positives=original[np.isfinite(original)&(original>0)]; vmax=max(float(np.percentile(positives,99)) if len(positives) else 1.,1e-6)
            extent=[0,ref.GetSize()[0]*ref.GetSpacing()[0],ref.GetSize()[1]*ref.GetSpacing()[1],0]
            for ax,m in zip(axs,METHODS):
                i=next(i for i,(t,mm,*_) in enumerate(maps) if t==chosen and mm==m)
                h=maps[i][-1]; a=original if m==METHODS[0] else resample(pet,ref,h); mr=resample(roi['mask'],ref,h,mask=True)>0
                ax.imshow(ctarr[z],cmap='gray',vmin=-200,vmax=250,extent=extent)
                ax.imshow(np.ma.masked_where(~np.isfinite(a[z])|(a[z]<=.05*vmax),a[z]),cmap='hot',vmin=0,vmax=vmax,alpha=.5,extent=extent)
                if target[z].any(): ax.contour(target[z],levels=[.5],colors=['lime'],linewidths=1.2,extent=extent,origin='upper')
                if mr[z].any(): ax.contour(mr[z],levels=[.5],colors=['cyan'],linewidths=1,linestyles='dashed',extent=extent,origin='upper')
                ax.set_title(m,fontsize=10);ax.axis('off')
            fig.suptitle(f'{args.patient} {roi["id"]} {chosen}: fixed SEG (green), transformed SEG (cyan dashed) | axial CT crop')
            fig.tight_layout();fig.savefig(out/'04_roi_overlay.png',dpi=180,bbox_inches='tight')
            if args.no_show: plt.close(fig)
    write_csv(out/'roi_metrics.csv',rows);write_csv(out/'native_vs_ct_grid.csv',native_rows)
    summary=[]
    for t in args.tests:
        for m in METHODS:
            rr=[r for r in rows if r['test']==t and r['method']==m]; err=np.array([r['centroid_error_mm'] for r in rr]); cn=np.array([r['cnr'] for r in rr]); cn=cn[np.isfinite(cn)]
            summary.append(dict(test=t,method=m,n_roi=len(rr),mean_centroid_error_mm=float(err.mean()),median_centroid_error_mm=float(np.median(err)),max_centroid_error_mm=float(err.max()),n_valid_cnr=len(cn),median_cnr=float(np.median(cn)) if len(cn) else np.nan))
    write_csv(out/'roi_summary.csv',summary)
    info.update({'patient':args.patient,'pet_units':units,'tests':args.tests,'background_shell_mm':[args.inner,args.outer],'centroid':'analytic native binary SEG centroid; inv(T_est) @ S @ c, not clinical TRE','cnr':'fixed CT-grid ROI, shell excluding all annotated SEG and CT <= -350; common finite samples across all tests/methods; sample SD; native PET comparison has different sampling','connectivity':26,'note':'No SUV conversion. SEG annotations are research references, not independent CT landmarks. CNR shell is not guaranteed homogeneous tissue. Native vs CT grid differs in sample counts.'})
    (out/'roi_protocol.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nTest / method / mean centroid mm / median CNR / valid CNR')
    for r in summary: print(f"{r['test']:3} {r['method']:22} {r['mean_centroid_error_mm']:9.4f} {r['median_cnr']:9.4f} {r['n_valid_cnr']}/{r['n_roi']}")
    (out/'roi_status.json').write_text(json.dumps({'status':'OK','patient':args.patient,'n_roi':len(rois),'tests':args.tests},indent=2),encoding='utf-8')
    print('Saved:',out.resolve());plot_results(rows,out,args.tests,not args.no_show)


def self_test():
    s=np.eye(4); angle=.12; s[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]];s[:3,3]=[4,-3,2]
    c=np.array([100.,-193.,-759.5]); assert np.allclose(roi_position(s,s,c),c)
    assert np.allclose(roi_position(s,np.eye(4),c),apply(s,c))
    a=np.arange(30,dtype=float);t=np.arange(30)<5;b=np.arange(30)>=5
    assert np.isclose(cnr_values(a,t,b)[0],abs(a[t].mean()-a[b].mean())/a[b].std(ddof=1))
    print('SELF-TEST PASSED: inverse transform, rigid composition, CNR formula (no DICOM).')


def locate_study(root):
    if not root.is_dir(): raise FileNotFoundError(f'Patient folder not found: {root}')
    def valid(p):
        names=[x.name.upper() for x in p.iterdir() if x.is_dir()]
        return all(sum(n.startswith(m+'_') for n in names)==1 for m in ('CT','PT','SEG'))
    if valid(root): return root
    candidates=[p for p in root.rglob('*') if p.is_dir() and valid(p)]
    if len(candidates)!=1: raise ValueError(f'{root}: found {len(candidates)} study folders containing CT_, PT_, SEG_; expected one.')
    return candidates[0]


def main():
    base=Path(__file__).resolve().parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patient',default='patient_01')
    p.add_argument('--data-root',type=Path,default=base/'data')
    p.add_argument('--results-root',type=Path,default=base/'results')
    p.add_argument('--out',type=Path)
    p.add_argument('--tests',nargs='+',default=['T0','T1','T2','T3'])
    p.add_argument('--inner',type=float,default=5.);p.add_argument('--outer',type=float,default=15.)
    p.add_argument('--no-show',action='store_true');p.add_argument('--self-test',action='store_true')
    p.add_argument('--check-only',action='store_true',help='Read SEG and list ROI, no registration results needed')
    p.add_argument('--debug',action='store_true',help='Print complete traceback on errors')
    a=p.parse_args()
    if a.self_test: self_test();return
    if ndi is None: raise ImportError('scipy is missing; run python -m pip install scipy')
    if not 0<a.inner<a.outer: p.error('Require 0 < inner < outer')
    if len(set(a.tests))!=len(a.tests): p.error('Duplicate test names')
    a.data_root=a.data_root.resolve();a.results_root=a.results_root.resolve()
    run(a)

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'ERROR [{type(e).__name__}]: {e}',file=sys.stderr)
        if '--debug' in sys.argv: traceback.print_exc()
        if isinstance(e,ImportError): print('Install: python -m pip install SimpleITK pydicom numpy scipy matplotlib',file=sys.stderr)
        sys.exit(1)
