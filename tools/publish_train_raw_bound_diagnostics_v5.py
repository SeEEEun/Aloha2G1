#!/usr/bin/env python3
"""All fixed TRAIN11 raw morphology LOWER bounds, not fabricated feasibility."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_hinge_orbit_bounds import orbit_lower_bounds

def main():
    verified_oracle_contract();g,c,n=model();g.assign(n);bounds=full_chain_enclosures(g)
    ids=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')['selection']['qualification_ids']
    folder=OUT/'07_paper_artifacts/diagnostic_position_v5';rows=[];aggregate={}
    for mode in ('WRIST','INTERACTION'):
        values=[];certs=[]
        for ep in ids:
            case=f'{mode}_EP{ep:03d}';t,h,ts,source=load_input(case,g,n)
            lb=np.array([orbit_lower_bounds(v,bounds) for v in t]);cert=lb.max(axis=1)>.01000001
            values.append(lb);certs.append(cert)
            rows.append(dict(case=case,source=source,frames=len(t),certified_beyond_10mm_frames=np.flatnonzero(cert).tolist(),
                certified_lower_bound_mm=(1000*lb).tolist(),not_certified_is_not_proven_reachable=True))
        v=np.concatenate(values)*1000;cc=np.concatenate(certs)
        aggregate[mode]=dict(frames=len(cc),certified_beyond_10mm_frames=int(cc.sum()),certified_fraction=float(cc.mean()),
            lower_bound_mm=dict(mean=float(v.mean()),p95=float(np.quantile(v,.95)),max=float(v.max())),
            correction_distribution='NOT ESTABLISHED across all trajectories; these are lower bounds only')
    p=folder/'RAW_CERTIFIED_MORPHOLOGY_DIAGNOSTICS.json'
    atomic_json(p,dict(scope='ALL_FIXED_TRAIN11_RAW_TARGETS',aggregate=aggregate,rows=rows,
        model_certificate=file_record(ST5/'full_chain_certificate/MODEL_ONLY_CERTIFICATE.json'),
        no_target_modification=True,no_outcome_input=True,warning='No certificate is not proof of reachability; lower bounds are not achieved corrections.'))
    lines=['# Raw morphology certificates — fixed TRAIN11\n',
        '| Metric | A WRIST | B INTERACTION |','|---|---:|---:|']
    a,b=aggregate['WRIST'],aggregate['INTERACTION']
    lines.append(f"| Certified beyond the 10 mm raw gate | {a['certified_beyond_10mm_frames']}/{a['frames']} | {b['certified_beyond_10mm_frames']}/{b['frames']} |")
    lines.append(f"| Certified fraction | {100*a['certified_fraction']:.3f}% | {100*b['certified_fraction']:.3f}% |")
    for stat in ('mean','p95','max'):lines.append(f"| Cartesian error lower bound {stat} (mm, pooled wrists/frames) | {a['lower_bound_mm'][stat]:.6f} | {b['lower_bound_mm'][stat]:.6f} |")
    lines.extend(['','These values are rigorous outer-enclosure error lower bounds, not numerical solver residuals, achieved corrections, or task outcomes. Non-certified frames remain unclassified until a valid witness or stronger certificate exists. No raw targets, timing, or physical limits were changed.'])
    atomic_text(folder/'TABLE_RAW_CERTIFIED_MORPHOLOGY_DIAGNOSTICS.md','\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    colors=['#3274a1','#e1812c'];labels=['A — WRIST','B — INTERACTION']
    vals=[100*aggregate[k]['certified_fraction'] for k in ('WRIST','INTERACTION')]
    axes[0].bar(labels,vals,color=colors);axes[0].set_ylim(0,max(vals)*1.25);axes[0].set_ylabel('Frames certified beyond 10 mm (%)')
    for i,k in enumerate(('WRIST','INTERACTION')):
        axes[0].text(i,vals[i]+.4,f"{aggregate[k]['certified_beyond_10mm_frames']}/7588\n{vals[i]:.3f}%",ha='center')
        data=np.concatenate([np.max(r['certified_lower_bound_mm'],axis=1) for r in rows if r['case'].startswith(k+'_')])
        data=np.sort(data);axes[1].plot(data,np.arange(1,len(data)+1)/len(data),color=colors[i],label=labels[i])
    axes[1].axvline(10,color='black',ls='--',lw=1);axes[1].set_xlabel('Per-frame maximum certified lower bound (mm)')
    axes[1].set_ylabel('Cumulative fraction of all TRAIN11 frames');axes[1].legend();axes[1].set_ylim(0,1.02)
    fig.suptitle('Raw morphology lower bounds — fixed TRAIN11, not physical task outcomes',weight='bold')
    for suffix in ('png','pdf','svg'):fig.savefig(folder/f'FIGURE_RAW_CERTIFIED_LOWER_BOUNDS.{suffix}',dpi=220)
    plt.close(fig)
    print('RAW_CERTIFIED_TRAIN11',aggregate,flush=True)

if __name__=='__main__':main()
