"""Independent saved observation/model-output/executed-command alignment checks."""
from pathlib import Path
import hashlib
import numpy as np
from .io import ROOT,read,record,atomic_json


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run(out,version='v1'):
    import cv2
    results=[]
    for method in ('A','B'):
        folder=out/'ACT_interface_diagnostics'/version/f'LEGACY_ACT_{method}'
        cfg=read(folder/'ACT_RUNTIME.json');summary=read(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json')
        trace=dict(np.load(folder/'event_log.npz'));f=trace['control_frame'];steps=sorted((folder/'policy_steps').glob('*.json'))
        errors=[];queries=[];roundtrip=[];alignment=[];chunk_error=[]
        for path in steps:
            row=read(path);frame=row['control_frame'];worker=row['worker']
            rgb=cv2.cvtColor(cv2.imread(row['image']['path']),cv2.COLOR_BGR2RGB)
            assert digest(rgb)==worker['rgb_sha256']
            state=np.asarray(row['measured_state'],np.float32)
            assert digest(state)==worker['state_sha256']
            if frame==0:expected=np.asarray(cfg['initial_q_rad'])
            else:expected=trace['MEASURED_Q'][np.flatnonzero(f==frame-1)[-1]]
            alignment.append(float(np.max(np.abs(state-expected))))
            query=frame//50*50;chunk=dict(np.load(folder/'chunks'/f'query_{query:06d}.npz'))
            chunk_error.append(float(np.max(np.abs(np.asarray(row['raw_action'])-chunk['physical'][frame-query]))))
            if worker['policy_query_this_call']:
                queries.append(frame);roundtrip.append(worker['normalization_round_trip_max_abs_rad'])
            ids=np.flatnonzero(f==frame)
            if row['executed_command'] is None:
                assert not len(ids)
            else:
                assert len(ids)==8
                errors.append(float(np.max(np.abs(trace['EXECUTED_COMMAND'][ids]-np.asarray(row['executed_command'])))))
                assert np.array_equal(trace['RAW_POLICY_COMMAND'][ids[0]],np.asarray(row['raw_action']))
        assert max(errors,default=0.)<1e-7
        assert max(alignment,default=0.)<1e-6
        assert max(chunk_error,default=0.)<1e-7
        assert queries==list(range(0,len(steps),50))
        assert summary['reset_response']['status']=='PASS'
        assert not any(summary[k] for k in ['teacher_reference_used','source_event_clock_used','high_level_planner_used','event_driven_demo_controller_used'])
        specs=read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json')['joint_specs']
        q=trace['MEASURED_Q'];lower=np.asarray([r['minimum'] for r in specs]);upper=np.asarray([r['maximum'] for r in specs])
        excess=np.maximum(np.maximum(lower-q,q-upper),0.)
        result=dict(method=method,status='SAVED_CAUSAL_INTERFACE_VERIFIED',trace=record(folder/'event_log.npz'),
            checkpoint=cfg['checkpoint'],model_sha256=cfg['model_sha256'],physical_rows=len(f),
            executed_control_frames=int(len(np.unique(f))),observation_calls=len(steps),query_frames=queries,
            observation_state_alignment_max_error_rad=max(alignment,default=0.),
            executed_command_trace_max_error_rad=max(errors,default=0.),
            official_chunk_action_max_error_rad=max(chunk_error,default=0.),
            normalization_roundtrip_max_error_rad=max(roundtrip,default=0.),
            raw_measured_joint_limit_excess_rad=float(excess.max()),raw_excursion_samples=int(np.count_nonzero(excess)),
            safety_stop=summary['aborted'],ACT_primary_trial=False,compatible_policy_checkpoint=False,
            source_conditioning='Source-conditioned initial scene only; ACT sees current RGB and measured joints, no source events or reference path.',
            physical_full_task_outcome='NOT_EVALUATED_SHORT_INTERFACE_DIAGNOSTIC')
        atomic_json(folder/'INDEPENDENT_INTERFACE_VERIFICATION.json',result);results.append(result)
    atomic_json(out/'ACT_interface_diagnostics'/version/'INDEPENDENT_VERIFICATION.json',results)
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--version',default='v1')
    a=p.parse_args();print(run(a.run_dir,a.version))
