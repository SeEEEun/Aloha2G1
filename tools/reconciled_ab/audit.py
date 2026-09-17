"""Read-only historical array reconciliation; writes only the new audit root."""
import collections, sys
import numpy as np
from scipy.spatial.transform import Rotation
from .common import *
from tools.final_paper_position_run import model, inputs, QUAL, INITIAL, RESET
from tools.common_physical_position_v4 import temporal_metrics
from tools.master_autonomous_common import preparation_path

def stats(x): return dict(mean=float(np.mean(x)),p95=float(np.quantile(x,.95)),max=float(np.max(x)))
def fk(g,q):
    pos=[];rot=[]
    for v in q:
        g.assign(v)
        pos.append([g.data.xpos[g.wrist_ids[s]].copy() for s in ('left','right')])
        rot.append([g.data.xmat[g.wrist_ids[s]].reshape(3,3).copy() for s in ('left','right')])
    return np.array(pos),np.array(rot)
def assess(g,q,ts,t,r,full,geometry):
    p,o=fk(g,q);err=np.linalg.norm(p-t,axis=2)*1000
    angle=Rotation.from_matrix((r@o.transpose(0,1,3,2)).reshape(-1,3,3)).magnitude().reshape(-1,2)
    tm=temporal_metrics(full,float(np.median(np.diff(ts))),read(QUAL))
    limits=int(np.count_nonzero((full<g.arm_limits[:,0])|(full>g.arm_limits[:,1])))
    gates=[]
    if not np.isfinite(full).all():gates.append('NONFINITE')
    if limits:gates.append('HARD_LIMIT')
    if geometry.get('HARD_SELF_COLLISION',0):gates.append('HARD_COLLISION')
    if geometry.get('UNRESOLVED_GEOMETRY',0):gates.append('UNRESOLVED_GEOMETRY')
    if not tm['pass_temporal']:gates.append('TEMPORAL')
    return dict(position_mm=stats(err),bilateral_position_mm=stats(err.max(axis=1)),orientation_rad=stats(angle),
        raw_frame_fraction_10mm=float(np.mean(err.max(axis=1)<=10)),historical_95percent_fidelity=bool(np.mean(err.max(axis=1)<=10)>=.95),
        temporal=tm,hard_limit_scalars=limits,geometry_counts=geometry,layer_b_gates=gates,layer_b_valid=not gates,
        geometry_evidence='CACHED_DEPENDENCY_HASHED_CLASSIFIER; independent sampled recheck is separate',
        natural_q0_max_diff=float(np.max(np.abs(full[0]-read(INITIAL)['g1_14_arm_initial_q_rad']))))

def main():
    g,c,n=model();natural=np.array(read(INITIAL)['g1_14_arm_initial_q_rad']);cases=read(PRIOR/'CASE_MANIFEST.json')
    rows=[];rich=[];invocations=[];newby={}
    for row in cases:
        t,r,h,ts,opened=inputs(row,g,n);key=row['key'];prior=read(PRIOR/'position'/key/'RESULT.json')
        for stage,candidates in [('POSITION',prior['families']),('FULL6D',[read(PRIOR/'full6d'/key/'RESULT.json')])]:
            for f in candidates:
                if 'trajectory' not in f:
                    if stage=='FULL6D': invocations.append(dict(key=key,status='NOT_RUN_UPSTREAM',historical_outcome=f['outcome']))
                    continue
                assert record(f['trajectory']['path'])==f['trajectory']
                assert record(f['geometry']['path'])==f['geometry']
                z=np.load(f['trajectory']['path']);q=z['q'];full=z['full_q']
                assert np.array_equal(z['RAW_REPRESENTATION_TARGET'],t)
                assert np.array_equal(z['source_timestamp'],ts)
                a=assess(g,q,ts,t,r,full,f['geometry_counts']);a.update(case=row,stage=stage,family=f.get('family','6D'),trajectory=f['trajectory'],geometry=f['geometry'],historical_outcome=f['outcome'],historical_rejecting_frames=f['metrics']['failed_frames'])
                if 'EXECUTABLE_FK_POSITION' in z: a['saved_fk_max_diff_m']=float(np.max(np.abs(z['EXECUTABLE_FK_POSITION']-fk(g,q)[0])))
                rich.append(a);newby.setdefault(key,[]).append(a)
                gates=a['layer_b_gates']; allg=gates+(['TRACKING_CONTRACT'] if f['metrics']['failed_frames'] else [])+(['ANGULAR_FIDELITY'] if a['orientation_rad']['max']>.75 and stage=='FULL6D' else [])
                rows.append(dict(family='CURRENT_FROZEN',key=key,recording=row['source_recording_id'],stage=stage,candidate=a['family'],historical_verdict=f['outcome'],first_layer_b_reject=gates[0] if gates else 'NONE',all_layer_b_rejects=';'.join(gates),all_rejects=';'.join(allg),raw_10mm_fraction=a['raw_frame_fraction_10mm'],position_mean_mm=a['position_mm']['mean'],position_max_mm=a['position_mm']['max'],angular_max_rad=a['orientation_rad']['max'],qdot_max=a['temporal']['maximum_velocity_rad_s'],qddot_max=a['temporal']['maximum_acceleration_rad_s2'],trajectory=f['trajectory']['path']))
                if stage=='FULL6D':invocations.append(dict(key=key,status='RUN_RETURNED_VALID_CANDIDATE' if a['layer_b_valid'] else 'RUN_RETURNED_NO_VALID_CANDIDATE',historical_outcome=f['outcome'],input=f['position_input'],result=f['trajectory'],gates=allg,position_mm=a['position_mm'],orientation_rad=a['orientation_rad']))
        if len(newby)%10==0: print('AUDIT_CURRENT',len(newby),'/150',flush=True)
    intermediate=[]
    diagnostics=read(OLD/'07_paper_artifacts/diagnostic_position_v5/TRAIN11_POSITION_DIAGNOSTICS.json')
    for d in diagnostics['rows']:
        qp=d.get('selected_qualification'); result=read(qp['path']) if qp else {}
        tr=result.get('trajectory',result.get('input'))
        if not isinstance(tr,dict) or not str(tr.get('path','')).endswith('.npz'):
            intermediate.append(dict(case=d['case'],qualified=d['qualified'],status='NOT_COMPARABLE_WITHOUT_ADDITIONAL_PROVENANCE',qualification=qp));continue
        z=np.load(tr['path']);q=z['q'];row=next(x for x in cases if x['key']==f"TRAIN40_{d['mode']}_{d['episode']:03d}")
        t,r,h,ts,opened=inputs(row,g,n)
        savedt=z['RAW_REPRESENTATION_TARGET'] if 'RAW_REPRESENTATION_TARGET' in z else None
        same=savedt is not None and np.array_equal(t,savedt) and len(q)==len(ts)
        if not same:
            intermediate.append(dict(case=d['case'],qualified=d['qualified'],status='NOT_COMPARABLE_WITHOUT_ADDITIONAL_PROVENANCE',trajectory=tr));continue
        full=np.vstack((preparation_path(natural,q[0],21)[:-1],q))
        gc=dict(HARD_SELF_COLLISION=len(d['hard_collision_frames'] or []),UNRESOLVED_GEOMETRY=len(d['unresolved_geometry_frames'] or []) if d['unresolved_geometry_frames'] is not None else 1)
        a=assess(g,q,ts,t,r,full,gc);a.update(case=row,qualified=d['qualified'],trajectory=tr,target_compatibility='EXACT_CURRENT_RAW_ARRAY',qualification=qp)
        intermediate.append(a)
        gates=a['layer_b_gates'];current=newby[row['key']]
        rows.append(dict(family='INTERMEDIATE_QUALIFIED',key=row['key'],recording=row['source_recording_id'],stage='POSITION',candidate='historical_selected_portfolio',historical_verdict='QUALIFIED' if d['qualified'] else 'NOT_QUALIFIED',first_layer_b_reject=gates[0] if gates else 'NONE',all_layer_b_rejects=';'.join(gates),all_rejects=';'.join(gates),raw_10mm_fraction=a['raw_frame_fraction_10mm'],position_mean_mm=a['position_mm']['mean'],position_max_mm=a['position_mm']['max'],angular_max_rad=a['orientation_rad']['max'],qdot_max=a['temporal']['maximum_velocity_rad_s'],qddot_max=a['temporal']['maximum_acceleration_rad_s2'],trajectory=tr['path']))
    old=[]
    for entry in read(ROOT/'outputs/paper_core_ab/train40_manifest.json')['entries']:
        for mode,letter in [('WRIST','a'),('INTERACTION','b')]:
            p=entry[f'{letter}_trajectory_path'];z=np.load(p);q=z['g1_arm_qpos'].astype(float);ts=z['timestamp']
            assert np.array_equal(z['g1_arm_joint_names'],g.arm_joint_names)
            assert str(z['source_directory_name'])==entry['original_source_recording_id']
            row=next(x for x in cases if x['group']=='TRAIN40' and x['representation_mode']==mode and x['source_recording_id']==entry['original_source_recording_id'])
            t=np.stack([z[f'target_{s}_wrist_position_model'] for s in ('left','right')],axis=1)
            r=np.stack([z[f'target_{s}_wrist_rotation_model'] for s in ('left','right')],axis=1)
            ct,cr,ch,cts,opened=inputs(row,g,n)
            # Own archived targets only. Never score OLD q against CURRENT targets.
            a=assess(g,q,ts,t,r,q,{})
            a['geometry_evidence']='NOT_REEVALUATED: old collision contract must not be silently replaced'
            a['layer_b_valid']=None;a['natural_q0_max_diff_diagnostic_only']=a.pop('natural_q0_max_diff')
            a.update(case=row,original_episode_index=entry['original_episode_index'],stable_episode_id=entry['stable_episode_id'],trajectory=record(p),
                own_target_semantics='archived target_*_wrist_*_model; feasible projection fields separately retained',
                old_common_config_sha256=str(z['common_config_sha256']),old_implementation_sha256=str(z['implementation_sha256']),
                cross_version_status='NOT_COMPARABLE_WITHOUT_ADDITIONAL_PROVENANCE',
                cross_version_reason='Old representation/morphology projection and source event contract differ; no motion correction applied',
                timestamp_equal=bool(np.array_equal(ts,cts)),raw_target_equal=bool(t.shape==ct.shape and np.array_equal(t,ct)),
                target_array_delta_mm=stats(np.linalg.norm(t-ct,axis=2)*1000) if t.shape==ct.shape else None,
                events=dict(zip(z['event_names'].astype(str),z['event_frames'].astype(int).tolist())),
                archived_projection_mm=stats(z['feasibility_projection_translation_m']*1000))
            old.append(a)
            rows.append(dict(family='OLD_TRAINING',key=row['key'],recording=row['source_recording_id'],stage='ARCHIVED_6D',candidate='checkpoint_supervision_ancestor',historical_verdict='TRAINED_NOT_PHYSICAL_VALIDITY_PROOF',first_layer_b_reject='NOT_COMPARABLE_WITHOUT_ADDITIONAL_PROVENANCE',all_layer_b_rejects='NOT_COMPARABLE_WITHOUT_ADDITIONAL_PROVENANCE',all_rejects='OWN_TARGET_METRICS_ONLY',raw_10mm_fraction=a['raw_frame_fraction_10mm'],position_mean_mm=a['position_mm']['mean'],position_max_mm=a['position_mm']['max'],angular_max_rad=a['orientation_rad']['max'],qdot_max=a['temporal']['maximum_velocity_rad_s'],qddot_max=a['temporal']['maximum_acceleration_rad_s2'],trajectory=p))
    save(RUN/'audit/CURRENT_CANDIDATE_AUDIT.json',rich);save(RUN/'audit/INTERMEDIATE_AUDIT.json',intermediate);save(RUN/'audit/OLD_TRAINING_LINEAGE.json',old)
    save(RUN/'audit/FULL6D_INVOCATIONS.json',invocations);csvsave(RUN/'REJECTION_REASON_MATRIX.csv',rows)
    counts={}
    for group in ('TRAIN40','DEV35'):
        for mode in ('WRIST','INTERACTION'):
            for stage in ('POSITION','FULL6D'):
                counts[f'{group}_{mode}_{stage}']=len({x['case']['key'] for x in rich if x['case']['group']==group and x['case']['representation_mode']==mode and x['stage']==stage and x['layer_b_valid']})
    save(RUN/'audit/AUDIT_COUNTS.json',counts)
    text(RUN/'LINEAGE_AND_CONTRACT_RECONCILIATION.md','# Lineage and contract reconciliation\n\nThe earlier TRAIN11 A7/11, B10/11 used selected trajectories from multiple common recovery stages, including model-certified selection and local repairs. The latest frozen batch generated a new uniform v9 posture-pool forward/reverse portfolio with fixed budgets. These are not the same q arrays or search portfolio. Raw targets and timestamps are checked explicitly in the accompanying JSON; where exact compatibility is established, independent FK and temporal checks use the same model.\n\nThe fall to A0/40 is not explained merely by an aggregate 0.179 rad gate: that gate was already internal-only. New candidates have measured per-joint motion, hard collision, or unresolved geometry failures. First and all gates are retained in REJECTION_REASON_MATRIX.csv. Historical selected trajectories are not silently installed as new frozen results.\n\nOLD_TRAINING uses different raw/projected target generation and source timing, no 0.700s natural preparation, and its own archived wrist targets. Own-target FK diagnostics are reported, not a rejection against current targets. Cross-version physical verdict: NOT_COMPARABLE_WITHOUT_ADDITIONAL_PROVENANCE. Trained does not mean physically qualified. Archived projection amounts remain distinct from raw errors.\n\nCurrent 6D was invoked only for B: 19 TRAIN and18 DEV. It returned complete trajectories. A was NOT_RUN_UPSTREAM, not an observed orientation failure. The new fidelity/validity separation reveals candidates rejected solely by tracking gates; this is a new acceptance interpretation, not a correction to the historical counts.\n\nRecomputed Layer-B-only counts (before final P14 command/runtime qualification):\n\n```json\n'+json.dumps(counts,indent=2)+'\n```\n')
    log('SAVED_ARRAY_RECONCILIATION','COMPLETE','Independently recomputed saved-array FK and physical temporal diagnostics; historical reports unchanged.',artifacts=[RUN/'audit/AUDIT_COUNTS.json',RUN/'REJECTION_REASON_MATRIX.csv'],next_stage='6D_KNOWN_ANSWER_AND_RUNTIME_COMPONENT_AUDIT')
    print(counts,flush=True)

if __name__=='__main__':main()
