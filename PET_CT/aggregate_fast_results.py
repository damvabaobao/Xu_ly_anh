"""Aggregate Fast registration and ROI metrics by patient, test and method."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def number(v):
    try:return float(v)
    except (ValueError,TypeError):return np.nan

def stats(rows,keys):
    out=[]
    for t,m in sorted({(r['test'],r['method']) for r in rows}):
        rr=[r for r in rows if r['test']==t and r['method']==m]
        for key in keys:
            a=np.array([number(r.get(key)) for r in rr]);a=a[np.isfinite(a)]
            out.append(dict(test=t,method=m,metric=key,n_patients=len(a),mean=a.mean() if len(a) else np.nan,sd=a.std(ddof=1) if len(a)>1 else np.nan,median=np.median(a) if len(a) else np.nan,q1=np.percentile(a,25) if len(a) else np.nan,q3=np.percentile(a,75) if len(a) else np.nan))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--patients',nargs='+',default=['patient_01','patient_02','patient_03','patient_04']);p.add_argument('--k-group',default='k2_k3_k7');args=p.parse_args()
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    root=args.project.resolve();out=root/'results'/'fast_aggregate'/args.k_group;out.mkdir(parents=True,exist_ok=True);rows=[];rois=[];statuses=[];expected=None
    for patient in args.patients:
        folder=root/'results'/patient/'fast_evaluation'/args.k_group;rr=read(folder/'fast_summary.csv');pairs={(r['test'],r['method']) for r in rr}
        if len(pairs)!=len(rr) or any(r['patient']!=patient for r in rr):raise ValueError(f'{patient}: duplicate or mismatched result rows')
        if expected is None:expected=pairs
        elif pairs!=expected:raise ValueError(f'{patient}: different test/method set; run the same protocol on all patients')
        rows.extend(rr);status=json.loads((folder/'roi_status.json').read_text(encoding='utf-8'))['roi_status'];statuses.append(dict(patient=patient,roi_status=status))
        if status=='OK':rois.extend([dict(patient=patient,**r) for r in read(folder/'roi_summary.csv')])
    write(out/'all_registration.csv',rows);write(out/'all_roi_patient_level.csv',rois);write(out/'roi_availability.csv',statuses)
    regkeys=['absolute_mean_mm','absolute_rmse_mm','relative_mean_mm','relative_rmse_mm','relative_rotation_error_deg','histogram_mi_nats','seconds_total'];roikeys=['mean_centroid_error_mm','median_cnr']
    regstats=stats(rows,regkeys);roistats=stats(rois,roikeys);write(out/'registration_statistics.csv',regstats);write(out/'roi_statistics.csv',roistats)
    speed=[]
    for r in rows:
        if not r['method'].startswith('Fast-'):continue
        full=next((q for q in rows if q['patient']==r['patient'] and q['test']==r['test'] and q['method']=='Full-7'),None)
        if full and number(r['seconds_total'])>0:speed.append(dict(patient=r['patient'],test=r['test'],method=r['method'],speedup=number(full['seconds_total'])/number(r['seconds_total']),delta_absolute_mm=number(r['absolute_mean_mm'])-number(full['absolute_mean_mm']),delta_relative_mm=number(r['relative_mean_mm'])-number(full['relative_mean_mm'])))
    write(out/'paired_speed_accuracy.csv',speed)
    tests=[t for t in ['T0','T1','T2','T3','C1','C2','C3'] if any(r['test']==t for r in rows)]
    for data,keys in [(regstats,regkeys),(roistats,roikeys)]:
        for key in keys:
            if not data:continue
            fig,ax=plt.subplots(figsize=(10,5));methods=sorted({r['method'] for r in data if r['metric']==key})
            for m in methods:
                vals=[next((r for r in data if r['metric']==key and r['method']==m and r['test']==t),None) for t in tests];means=[v['mean'] if v else np.nan for v in vals];sd=[v['sd'] if v and np.isfinite(v['sd']) else 0 for v in vals];ax.errorbar(tests,means,yerr=sd,marker='o',capsize=3,label=m)
            ax.set(xlabel='Experiment',ylabel=key,title='Across-patient mean ± sample SD (see CSV for n)');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/(key+'.png'),dpi=160);plt.close(fig)
    print('Saved:',out,'| ROI availability:',statuses)
if __name__=='__main__':main()
