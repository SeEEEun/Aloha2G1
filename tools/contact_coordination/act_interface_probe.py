"""Bounded paired legacy-checkpoint dynamic TRAIN interface validation."""
import os,subprocess,time,shutil
from pathlib import Path
import numpy as np
from .io import ROOT,ISAAC,read,record,atomic_json,atomic_npz


def run(out,version='v1',resume=False):
    audit=read(out/'effective_data_audit/EFFECTIVE_SUPERVISION_AUDIT.json')
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    prefix=out/'prototype'/sid/'morphology_acquisition_v4'
    a=dict(np.load(prefix/'COMMANDS.npz'));q0=a['common_initial_q_rad'];names=list(a['joint_names'])
    results=[]
    for method in ('A','B'):
        folder=out/'ACT_interface_diagnostics'/version/f'LEGACY_ACT_{method}'
        if (folder/'PROCESS.json').exists():
            if not resume:raise FileExistsError(folder)
            for dep in read(folder/'ACT_RUNTIME.json')['dependencies']:
                assert record(dep['path'])['sha256']==dep['sha256'], 'Interface dependency drift: '+dep['path']
            result=read(folder/'PROCESS.json');results.append(result)
            if not result['complete_interface_trace']:break
            continue
        if folder.exists():raise FileExistsError('Interrupted diagnostic is immutable: '+str(folder))
        folder.mkdir(parents=True)
        ckpt=Path(audit['methods'][method]['checkpoint'])
        checker_dir=folder/'geometry_workspace'
        shutil.copytree(out/'target_repair/runtime_bin150',checker_dir/'target_repair/runtime_bin150')
        physics=read(out/'common_control/left_full_state/DEPENDENCIES.json')
        deps=[record(ROOT/p) for p in (
            'tools/contact_coordination/act_physics.py','tools/contact_coordination/act_policy_runtime.py',
            'tools/contact_coordination/runtime_hulls.py','tools/act_b_inference_worker.py',
            'tools/run_act_b_isaac_diagnostic.py','tools/common_jerk_limited_otg.py',
            'tools/common_deployment_safety_projection.py','tools/deployment_camera_config.py')]
        deps+=audit['methods'][method]['checkpoint_files']
        physics['files']+=deps
        atomic_json(folder/'DEPENDENCIES.json',physics)
        cfg=dict(runtime_mode='CURRENT_RGB_MEASURED_STATE_TO_ACT_TO_COMMON_OTG',
            output_dir=str(folder),run_dir=str(out),checker_dir=str(checker_dir),
            checkpoint=str(ckpt),model_sha256=record(ckpt/'model.safetensors')['sha256'],
            execution='e0',control_fps=30,maximum_control_frames=101,initial_q_rad=q0,joint_names=names,
            camera=str(ROOT/'outputs/policy_b_isaac_validation/camera/source_like_cam_high.json'),
            deployment_projection=str(ROOT/'outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json'),
            otg=str(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json'),
            dependencies=deps,main_study_eligible=False,reason='Existing checkpoint training supervision is incompatible; validate actual causal interface only.')
        atomic_json(folder/'ACT_RUNTIME.json',cfg)
        atomic_npz(folder/'ADMINISTRATIVE_LOOP.npz',commanded_q_rad=np.tile(q0,(101,1)),joint_names=np.asarray(names),
            stage=np.repeat('POLICY',101),control_fps_hz=np.asarray(30.))
        shutil.copy2(prefix/'SOURCE_SCENE.json',folder/'SOURCE_SCENE.json')
        args=read(out/'common_control/left_full_state/INVOCATION.json')['command']
        args[1]=str(ROOT/'tools/contact_coordination/act_physics.py')
        for key,value in [('--output-dir',folder),('--direct-freeze-manifest',folder/'DEPENDENCIES.json'),('--scripted-command-path',folder/'ADMINISTRATIVE_LOOP.npz')]:
            args[args.index(key)+1]=str(value)
        args+=['--episode-registration-manifest',str(folder/'SOURCE_SCENE.json'),'--episode-stable-id',sid]
        atomic_json(folder/'INVOCATION.json',dict(command=args,source_id=sid,primary_evaluation=False,checkpoint_compatible=False,timeout_s=420))
        start=time.monotonic()
        with (folder/'engine.log').open('w') as log:
            p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
            atomic_json(folder/'ACTIVE_PROCESS.json',dict(pid=p.pid,command=args))
            try:rc=p.wait(timeout=420)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                rc='INFRASTRUCTURE_TIMEOUT'
        summary=folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json'
        complete=summary.exists() and (folder/'event_log.npz').exists()
        result=dict(method=method,returncode=rc,wall_seconds=time.monotonic()-start,
            complete_interface_trace=complete,policy_steps=len(list((folder/'policy_steps').glob('*.json'))),
            summary=read(summary) if summary.exists() else None,primary_ACT_trials=0)
        atomic_json(folder/'PROCESS.json',result);results.append(result);print(result,flush=True)
        if not complete:break
    atomic_json(out/'ACT_interface_diagnostics'/version/'RESULT.json',results)
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--version',default='v1');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.run_dir.resolve(),a.version,a.resume)
