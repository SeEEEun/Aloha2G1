"""Common G1 posture seed bank; numerical q/FK only, no method or task score."""
import numpy as np
from scipy.spatial import cKDTree
from tools.common_fast_hard_witness_v5 import hard_witness

def generate(oracle,solver,targets,hands,timestamps,initial,bank_q,bank_position,allowances,reverse=False,progress=None,resume=None):
    tree=cKDTree(np.asarray(bank_position).reshape(len(bank_q),6));q=np.full((len(targets),14),np.nan)
    previous=initial.copy();reports=[]
    if resume is not None:
        q=resume['q'].copy();reports=list(resume['reports'])
        if reports:previous=q[reports[-1]['frame']].copy()
    completed={r['frame'] for r in reports}
    dt=float(np.median(np.diff(timestamps)))
    order=range(len(q)-1,-1,-1) if reverse else range(len(q))
    for f in order:
        if f in completed:continue
        _,nearest=tree.query(targets[f].reshape(-1),k=min(8,len(bank_q)))
        seeds=[previous,*[bank_q[int(k)] for k in np.atleast_1d(nearest)],solver.natural]
        candidates=[];attempts=[]
        for sid,seed in enumerate(seeds):
            value,fit=oracle.fit(targets[f],hands[f],seed,max_evaluations=180,reference=previous,posture_weight=.0001)
            value[[6,13]]=previous[[6,13]]
            pos=solver.pose_jacobian(value,hands[f])[0];res=np.linalg.norm(pos-targets[f],axis=1)
            witness=None;geom=None
            if np.all(res<=allowances[f]):
                try:witness=hard_witness(solver.collision,value,hands[f])
                except Exception as exc:witness=dict(classification='UNRESOLVED_GEOMETRY',reason=str(exc))
                if witness is None:geom=solver.collision.inspect(value,*hands[f])
            valid=geom is not None and not any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in geom)
            step=float(np.max(np.abs(value-previous)));distance=float(np.linalg.norm(value-previous))
            item=dict(seed_index=sid,position_residual_m=res.tolist(),geometry_valid=valid,hard_witness=witness,q_step_max_rad=step,nfev=fit['nfev'])
            attempts.append(item)
            candidates.append(((not valid,float(np.max(np.maximum(res-allowances[f],0))),distance,sid),value,item,geom))
            if valid and step<=solver.config['maximum_velocity_rad_s']*dt:break
        key,value,item,geom=min(candidates,key=lambda x:x[0]);q[f]=value;previous=value
        reports.append(dict(frame=f,selected=item,attempts=attempts,geometry=geom))
        if progress is not None and (f%25==0 or len(reports)==len(targets)):progress(q,reports)
        if f%25==0:print('COMMON_TRAIN_SEED_BANK',f,'valid',item['geometry_valid'],'seeds',len(attempts),'step',item['q_step_max_rad'],flush=True)
    return q,dict(frames=reports,all_frames_geometry_witnessed=all(r['selected']['geometry_valid'] for r in reports),
        source_time_not_changed=True,method_or_success_inputs=False)
