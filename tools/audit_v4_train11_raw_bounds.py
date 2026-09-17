#!/usr/bin/env python3
"""Raw TRAIN11 geometry bounds, not executable qualification or DEV statistics."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.run_reference_motion_scientific_reset import atomic_csv

def run():
    verified_oracle_contract();g,c,n=model();g.assign(n);bounds=orbit_enclosures(g)
    split=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json');ids=split['selection']['qualification_ids'];rows=[];inputs=[]
    for mode in ('WRIST','INTERACTION'):
        for ep in ids:
            case=f'{mode}_EP{ep:03d}';t,h,ts,src=load_input(case,g,n);inputs.append(src)
            lb=np.array([orbit_lower_bounds(p,bounds) for p in t]);gap=lb.max(axis=1)*1000;cert=gap>10.00001
            indices=np.flatnonzero(cert);longest=max([len(a) for a in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)],default=0)
            rows.append(dict(case=case,representation_mode=mode,episode=ep,frames=len(t),certified_unreachable=int(cert.sum()),
                certified_fraction=float(cert.mean()),remaining_frames_reachability='NOT_CERTIFIED_BY_OUTER_BOUND; no inside-bound reachability inference',
                certified_gap_lower_mean_mm=float(gap[cert].mean()) if cert.any() else 0.,certified_gap_lower_p95_mm=float(np.quantile(gap[cert],.95)) if cert.any() else 0.,
                certified_gap_lower_max_mm=float(gap.max()),longest_certified_interval=longest))
    folder=STAGE/'train11_raw_bounds';result=dict(scope='Predeclared TRAIN11 raw geometry bounds only, not executable trajectory qualification or DEV35 physical performance',rows=rows,inputs=inputs,
        split=file_record(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json'),implementation=file_record(ROOT/'tools/common_hinge_orbit_bounds.py'))
    atomic_json(folder/'TRAIN11_RAW_BOUNDS.json',result);atomic_csv(folder/'TRAIN11_RAW_BOUNDS.csv',rows)
    text='# Predeclared TRAIN11 raw geometry bounds\n\nOnly outside-enclosure targets receive an infeasibility certificate. Inside-enclosure targets are not declared reachable. No source trajectory or physical task pass is inferred.\n\n| Representation | Episode | Frames | Certified unreachable | Fraction | Correction lower mean/p95/max mm |\n|---|---:|---:|---:|---:|---|\n'
    for r in rows:text+=f"| {r['representation_mode']} | {r['episode']} | {r['frames']} | {r['certified_unreachable']} | {r['certified_fraction']:.4%} | {r['certified_gap_lower_mean_mm']:.3f}/{r['certified_gap_lower_p95_mm']:.3f}/{r['certified_gap_lower_max_mm']:.3f} |\n"
    atomic_text(folder/'TRAIN11_RAW_BOUNDS.md',text)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True);x=np.arange(len(ids))
    for mode,offset,color in [('WRIST',-.2,'#46779b'),('INTERACTION',.2,'#cf8a3b')]:
        rr=[r for r in rows if r['representation_mode']==mode];ax.bar(x+offset,[100*r['certified_fraction'] for r in rr],.4,label=mode,color=color)
    ax.set_xticks(x,ids);ax.set_xlabel('Predeclared TRAIN episode');ax.set_ylabel('Certified-unreachable raw frames (%)');ax.legend();ax.set_title('TRAIN11 raw morphology bounds — not physical task performance')
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'FIGURE_TRAIN11_CERTIFIED_RAW_BOUNDS.{ext}',dpi=180)
    plt.close(fig)
    atomic_json(folder/'MANIFEST.json',dict(inputs=inputs,artifacts=[file_record(p) for p in sorted(folder.iterdir()) if p.is_file()]))
    print('TRAIN11_RAW_BOUNDS_RECORDED',len(rows),flush=True)

if __name__=='__main__':run()
