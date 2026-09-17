#!/usr/bin/env python3
"""All-source model-only lower bounds; no new IK searches or target changes."""
from pathlib import Path
import csv,io,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_position_run import *
from tools.common_hinge_orbit_bounds import orbit_lower_bounds

def main():
    verify();g,c,n=model();g.assign(n);bounds=full_chain_enclosures(g);records=[]
    for row in cases():
        t,rot,h,ts,opened=inputs(row,g,n)
        lower=np.array([orbit_lower_bounds(v,bounds) for v in t]);flag=lower.max(axis=1)>.01000001
        edges=np.diff(np.r_[False,flag,False].astype(int));intervals=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1))
        path=DEST/'raw_certificate'/f"{row['key']}.npz"
        atomic_npz(path,RAW_REPRESENTATION_TARGET=t,CERTIFIED_LOWER_BOUND_MM=1000*lower,CERTIFIED_RAW_UNREACHABLE=flag,source_timestamp=ts)
        records.append(dict(case=row,source_frames=len(t),certified_raw_unreachable_frames=int(flag.sum()),certified_fraction=float(flag.mean()),longest_certified_interval_frames=max((int(b-a+1) for a,b in intervals),default=0),intervals=[[int(a),int(b)] for a,b in intervals],lower_bound_mm=dict(mean=float(lower.max(axis=1).mean()*1000),p95=float(np.quantile(lower.max(axis=1),.95)*1000),max=float(lower.max()*1000)),artifact=file_record(path)))
    report=dict(status='MODEL_ONLY_RAW_CERTIFICATES_COMPLETE',cases=150,freeze=file_record(FREEZE),bounds=bounds,records=records,
        interpretation='A positive bound is a necessary Cartesian correction, not a solved minimum. Bound >10mm certifies raw fidelity infeasibility. Bound <=10mm does not establish reachability. No certificate here proves absence of a permitted closest-feasible trajectory.')
    atomic_json(DEST/'raw_certificate/FULL_COHORT_RAW_CERTIFICATE_AUDIT.json',report)
    out=io.StringIO();w=csv.writer(out);w.writerow(['key','frames','certified_raw_unreachable','longest_interval_frames','lower_bound_mean_mm','lower_bound_p95_mm','lower_bound_max_mm'])
    for r in records:w.writerow([r['case']['key'],r['source_frames'],r['certified_raw_unreachable_frames'],r['longest_certified_interval_frames'],r['lower_bound_mm']['mean'],r['lower_bound_mm']['p95'],r['lower_bound_mm']['max']])
    atomic_text(DEST/'raw_certificate/FULL_COHORT_RAW_CERTIFICATE_AUDIT.csv',out.getvalue())
    print('ALL150_RAW_CERTIFICATES_SAVED_NO_IK_RERUN',flush=True)

if __name__=='__main__':main()
