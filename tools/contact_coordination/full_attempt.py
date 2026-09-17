"""Launch isolated nominal/diagnostic controls without modifying source plans."""
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from .io import ROOT, read, record, atomic_json, atomic_npz
from .abc_contract import require, METHODS


def prepare(out, source_plan, name, method_key, channel='OFFICIAL_NOMINAL',
            prefix_frames=None, injected_failure_frame=None, contract_test=False):
    require(out,'ARCHITECTURE_REPAIR' if (out/'ARCHITECTURE_REPAIR.json').exists() else 'TRAIN_control' if contract_test else 'full_diagnostics' if channel!='OFFICIAL_NOMINAL' else 'TRAIN_calibration')
    source_plan=Path(source_plan);assert method_key in METHODS
    if not contract_test and channel=='OFFICIAL_NOMINAL':
        assert read(source_plan/'FINAL_COMMAND_VALIDATION.json')['status']=='VALID'
    if prefix_frames is not None and not contract_test:
        raise ValueError('Only short contract controls may truncate a stored command')
    folder=out/'video_contract_controls'/name if contract_test else out/'physical_attempts'/name
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True);plan=folder/'input';plan.mkdir()
    with np.load(source_plan/'COMMANDS.npz',allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    original=len(arrays['stage']);nominal=min(prefix_frames,original) if prefix_frames else original
    observation_s=read(out/'ABC_STUDY.json')['nominal_settle_observation_s'];tail=int(round(observation_s*30))
    for key,value in arrays.items():
        if value.ndim and len(value)==original:
            value=value[:nominal];arrays[key]=np.concatenate([value,np.repeat(value[-1:],tail,axis=0)])
    arrays['stage']=arrays['stage'].astype('U80');arrays['stage'][nominal:]='POST_HORIZON_OBSERVATION'
    if bool(arrays.get('runtime_right_three_digit_gate_required',False)):
        raise ValueError('Legacy special release guard needs a separate qualified full-horizon adapter')
    if not np.isfinite(arrays['commanded_q_rad']).all():raise ValueError('Nonfinite source command')
    sid=read(source_plan/'PLAN.json')['source_id']
    atomic_npz(plan/'COMMANDS.npz',**arrays)
    for item in ('SOURCE_SCENE.json','PLAN.json'):
        shutil.copy2(source_plan/item,plan/item)
    cfg=dict(schema='full_attempt_recording_v1',output_dir=str(folder),source_id=sid,method_key=method_key,
        evidence_channel=channel,nominal_frames=nominal,observation_s=observation_s,
        contract_test_only=contract_test,injected_nonfatal_failure_frame=injected_failure_frame,
        source_commands=record(source_plan/'COMMANDS.npz'),source_scene=record(source_plan/'SOURCE_SCENE.json'),
        no_arm_path_retiming=True,contact_feedback_retiming='COMMON_VELOCITY_ACCELERATION_LIMITS',continue_existing_command_after_task_failure=True,
        architecture_repair_verification_only=(out/'ARCHITECTURE_REPAIR.json').exists(),
        tail='hold last executed command only after true nominal horizon')
    atomic_json(plan/'FULL_ATTEMPT_CONFIG.json',cfg)
    old=ROOT/'outputs/contact_coordination_hybrid/20260907T073437Z/common_control/scripted_captured'
    deps=read(old/'DEPENDENCIES.json')
    for item in deps['files']:
        assert record(item['path'])['sha256']==item['sha256'],item['path']
    extra=[ROOT/'tools/contact_coordination'/n for n in ('full_attempt.py','full_attempt_runtime.py','full_attempt_physics.py','pregrasp_protection.py','phase_physics.py','phase_clock_runtime.py','execution_timing.py','contact_command_retiming.py','physics_capture.py')]
    extra += list(plan.iterdir());deps['files'] += [record(x) for x in extra]
    deps['files'].append(record(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json'))
    deps.update(purpose='Full recording contract control' if contract_test else channel,
                evidence_channel=channel,not_eligible_for_training=contract_test or channel!='OFFICIAL_NOMINAL' or cfg['architecture_repair_verification_only'])
    atomic_json(folder/'DEPENDENCIES.json',deps)
    args=read(old/'INVOCATION.json')['command']
    args[1]=str(ROOT/'tools/contact_coordination/full_attempt_physics.py')
    for flag,value in [('--direct-freeze-manifest',folder/'DEPENDENCIES.json'),('--output-dir',folder),('--scripted-command-path',plan/'COMMANDS.npz')]:
        args[args.index(flag)+1]=str(value)
    i=args.index('--object-registration-config');args[i:i+2]=['--episode-registration-manifest',str(plan/'SOURCE_SCENE.json'),'--episode-stable-id',sid]
    atomic_json(folder/'INVOCATION.json',dict(command=args,timeout_s=1200,evidence_channel=channel,contract_test_only=contract_test))
    return folder


def launch(folder,resume=False):
    folder=Path(folder);receipt=folder/'PROCESS.json'
    for item in read(folder/'DEPENDENCIES.json')['files']:
        assert record(item['path'])['sha256']==item['sha256'],item['path']
    if receipt.exists():
        if resume:return read(receipt)
        raise FileExistsError(receipt)
    invocation=read(folder/'INVOCATION.json');start=time.monotonic()
    with (folder/'engine.log').open('w') as stream:
        try:
            result=subprocess.run(invocation['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                timeout=invocation['timeout_s'],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
            code=result.returncode
        except subprocess.TimeoutExpired:code='TIMEOUT'
    value=dict(returncode=code,wall_s=time.monotonic()-start,
        trace_exists=(folder/'event_log.npz').exists(),nominal_outcome_not_scored_by_launcher=True)
    atomic_json(receipt,value);return value
