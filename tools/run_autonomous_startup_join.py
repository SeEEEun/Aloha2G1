#!/usr/bin/env python3
"""Recheck common TRAIN preparation against actual current source-start states."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.master_autonomous_diagnostics import choose_candidate
from tools.run_autonomous_dual_position import *
from tools.master_autonomous_common import preparation_duration,preparation_path
from tools.run_reference_motion_scientific_reset import atomic_text


def write_saved_report(result):
    atomic_text(MASTER/'STARTUP_SOURCE_JOIN_CALIBRATION_V2.md',f'# Updated common preparation and actual source joins\n\n{result["status"]}\n\nCommon duration {result["PREP_DURATION_SECONDS"]:.9f}s, {result["PREP_NUM_FRAMES"]} preparation frames. All22 TRAIN first-target paths use the same duration and numerical rules. {result["source_join_count"]} actual current source joins checked. G1 ARM14 natural q0 is exact; Dex3 uses explicit common OPEN. Original raw source targets/timestamps remain separate; no raw tracking is claimed during preparation. This is kinematic self-collision qualification, not loaded articulation or physical object-contact evidence. Final freeze remains pending complete source TRAIN11 and6D qualification.\n')


def run():
    saved=MASTER/'STARTUP_SOURCE_JOIN_CALIBRATION_V2.json'
    if saved.exists():
        previous=read(saved)
        for row in previous['inputs']:
            if row['current_source_trajectory']:
                assert file_record(Path(row['current_source_trajectory']['path']))==row['current_source_trajectory']
        if not (MASTER/'STARTUP_SOURCE_JOIN_CALIBRATION_V2.md').exists():
            write_saved_report(previous)
            atomic_json(MASTER/'STARTUP_JOIN_REPORT_REPAIR.json',dict(action='Imported missing atomic_text and published the report from preserved JSON; no trajectory/path rerun',
                source=file_record(saved),original_implementation=previous['implementation'],updated_implementation=file_record(Path(__file__))))
        print('REUSE_PERSISTED_STARTUP_JOIN_V2; use a new version for changed source trajectories',flush=True)
        return
    verified_oracle_contract();g,c,natural=model();cfg=read(QUAL);initial=np.array(read(INITIAL)['g1_14_arm_initial_q_rad'])
    calibration=read(MASTER/'STARTUP_CALIBRATION.json');opened=np.load(RUN/'startup/WRIST_EP000_COMMON_PREFIX.npz')['common_hand_q'][0].copy()
    rows=[];sources={};dt=1/30
    for old in calibration['rows']:
        case=old['case'];end=np.array(old['first_task_q']);origin=old['source'];source_trajectory=None
        # Availability, not identity, selects a current completed source candidate.
        try:
            qp,rp,record,passed=choose_candidate(case)
        except RuntimeError:pass
        else:
            z=np.load(qp);end=z['q'][0].copy();source_trajectory=file_record(qp);sources[case]=(qp,passed)
            np.testing.assert_allclose(z['common_hand_q'][0],opened,atol=1e-14,rtol=0)
        n,q,minimum=preparation_duration(initial,end,dt,cfg)
        rows.append(dict(case=case,end_q=end.tolist(),required_intervals=n,path_minimum_duration_s=minimum,
            source=origin,current_source_trajectory=source_trajectory))
    n=max(r['required_intervals'] for r in rows);duration=n*dt;folder=RUN/'startup_source_join_v2';checks=[]
    for row in rows:
        case=row['case'];prefix=preparation_path(initial,np.array(row['end_q']),n)
        rr=[dict(frame=f,records=c.inspect(q,*opened)) for f,q in enumerate(prefix)]
        blocked=[r['frame'] for r in rr if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        pm=temporal_metrics(prefix,dt,cfg);assert np.array_equal(prefix[0],initial)
        entry=dict(case=case,prefix_geometry=rr,prefix_blocked_frames=blocked,prefix_temporal=pm,
            source_join=None,source_trajectory_qualified=False)
        if case in sources:
            qp,passed=sources[case];z=np.load(qp);q=z['q'];ts=z['source_timestamp'];hands=z['common_hand_q']
            full=np.vstack((prefix[:-1],q));execution_time=np.r_[np.arange(n)*dt,duration+(ts-ts[0])]
            # Evaluate just the preparation/source boundary separately from
            # already reported mid-task invalidity in diagnostic candidates.
            boundary=full[:n+min(3,len(q))]
            join=temporal_metrics(boundary,float(np.min(np.diff(execution_time[:len(boundary)]))),cfg)
            entry['source_join']=join;entry['source_trajectory_qualified']=passed
            atomic_npz(folder/f'{case}_PREPARED.npz',q=full,common_hand_q=np.concatenate((np.repeat(opened[None],n,axis=0),hands)),
                execution_timestamp=execution_time,source_timestamp=ts,RAW_REPRESENTATION_TARGET=z['RAW_REPRESENTATION_TARGET'],
                raw_tracking_mask=np.r_[np.zeros(n,dtype=bool),np.ones(len(q),dtype=bool)],
                source_frame_index=np.r_[np.full(n,-1),np.arange(len(q))],PREP_DURATION_SECONDS=np.array(duration))
            np.testing.assert_array_equal(full[0],initial)
        checks.append(entry);print('PREPARATION_JOIN',case,'blocked',len(blocked),'join',entry['source_join'],flush=True)
    result=dict(status='KINEMATIC_STARTUP_AND_SOURCE_JOINS_PASS' if all(not r['prefix_blocked_frames'] and r['prefix_temporal']['pass_temporal'] and (r['source_join'] is None or r['source_join']['pass_temporal']) for r in checks) else 'COMMON_PREPARATION_RECOVERY_REQUIRED',
        PREP_DURATION_SECONDS=duration,PREP_NUM_FRAMES=n,natural_arm_q0_exact=True,Dex3_OPEN_identical=True,
        full_28_matches_legacy_state=False,calibration_source='TRAIN11 x both representations; use actual completed source candidates where available, otherwise saved first-target witnesses',
        final_freeze=False,remaining='Complete TRAIN11 source realization and full6D may require common recalibration before execution freeze',
        inputs=rows,checks=checks,source_join_count=sum(r['source_join'] is not None for r in checks),
        simulation_object_contact_tested=False,implementation=file_record(Path(__file__)))
    atomic_json(MASTER/'STARTUP_SOURCE_JOIN_CALIBRATION_V2.json',result)
    atomic_text(MASTER/'STARTUP_SOURCE_JOIN_CALIBRATION_V2.md',f'# Updated common preparation and actual source joins\n\n{result["status"]}\n\nCommon duration {duration:.9f}s, {n} preparation frames. All22 TRAIN first-target paths use the same duration and numerical rules. {result["source_join_count"]} actual current source joins checked. G1 ARM14 natural q0 is exact; Dex3 uses explicit common OPEN. Original raw source targets/timestamps remain separate; no raw tracking is claimed during preparation. This is kinematic self-collision qualification, not loaded articulation or physical object-contact evidence. Final freeze remains pending complete source TRAIN11 and6D qualification.\n')
    print('STARTUP_JOIN_COMPLETE',result['status'],duration,flush=True)


if __name__=='__main__':run()
