"""Existing PhysX engine, matched environments, explicit safe branch handling."""
import subprocess,sys,os,time
from pathlib import Path
import numpy as np
from .common import *
from .score import STAGES,score
from tools.final_paper_reference_physics import ISAAC,PHYSICS_CONFIG,REG
from tools.final_paper_position_run import RESET,load_common_config,load_scene
from tools.run_reference_motion_scientific_reset import source_object

PF=RUN/'freeze/PHYSICS_PROTOCOL.json'
def protocol(pilot=False):
    comp=read(RUN/'components/COMPONENT_QUALIFICATION.json');assert comp['status']=='PASS'
    base=read(RUN/'components/COMPONENT_FREEZE.json');paths={Path(r['path']) for r in base['files']}
    paths.update([Path(REG),RUN/'freeze/CONSTRUCTION_PROTOCOL.json',Path(__file__),ROOT/'tools/reconciled_ab/score.py',ROOT/'tools/score_episode_registered_physical_eval35_run.py',ROOT/'outputs/final_episode_registered_eval35/01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json'])
    paths.update(Path(p) for p in read(RUN/'components/known_control/RUNTIME_COLLISION_MODEL.json')['usd_layers'])
    current=read(OLD/'01_registration/DEV35_EPISODE_TASK_ENVIRONMENTS.json')['entries'];reg=read(REG)['entries']
    for e in current:assert e['T_object_A']==e['T_object_B']==next(r for r in reg if r['stable_episode_id']==e['stable_episode_id'])['target_object_pose']
    value=dict(status='PRE_EVAL35_QUALIFICATION_PROVISIONAL' if pilot else 'FROZEN_BEFORE_EVAL35',graspability_classifier_used=False,files=[record(p) for p in sorted(paths)],
      component_qualification=record(RUN/'components/COMPONENT_QUALIFICATION.json'),created_at=now(),scope='TRAIN_REFERENCE_PILOT' if pilot else 'DEV35 DEVELOPMENT EVALUATION',
      same_environment_pose_count=35,natural_q0=True,prep_s=.7,physics='Unchanged dynamic PhysX doll, table, bin, gravity; no post-initialization root-pose writes, parenting or attachments.',
      runtime_self_collision_response=False,offline_command_and_measured_geometry='Common detailed classifier,10um; unresolved fails closed, distinct from runtime response.',
      scoring='Existing topology-neutral retained grasp/ownership,5mm confirmed lift,existing transport/bin/1s settle rules. Contact/enclosure candidate alone is not grasp. NO_ACQUISITION release taxonomy bug corrected.',
      timing='Unchanged source clock; controller contact waits/delays are recorded without arm rescue.',
      denominator_contract='construction/35; cumulative/35 with upstream zero and physical stages NOT_ATTEMPTED; observed tasks/valid actually executed. No execution => NOT MEASURED. Invalid/incomplete outcomes unknown.',
      normal_failure_retry=False,implementation_retry_budget=3,search_budget=record(RUN/'freeze/CONSTRUCTION_PROTOCOL.json'))
    fp=RUN/'components/TRAIN_PILOT_FREEZE.json' if pilot else PF
    if fp.exists():
        prior=read(fp)
        for dep in prior['files']:assert record(dep['path'])==dep
        return fp
    save(fp,value);return fp

def run_one(row,pilot=False):
    key=row['key'];construction=read(RUN/'construction'/key/'RESULT.json');folder=RUN/('train_pilot' if pilot else 'reference_physics')/key;rp=folder/'RESULT.json'
    if rp.exists():return read(rp)
    if construction['outcome']=='INFRASTRUCTURE_INVALID':raise RuntimeError('Unresolved construction infrastructure')
    if not construction['selected']:
        r=dict(case=row,outcome='NO_EXECUTABLE_TRAJECTORY_UNDER_FIXED_PROTOCOL',actual_physics_executed=False,physical_valid=None,physical_trace=None,
               physical_stages={s:'NOT_ATTEMPTED' for s in STAGES},cumulative_pipeline={s:False for s in STAGES},first_failure_stage='TRAJECTORY_CONSTRUCTION',construction=record(RUN/'construction'/key/'RESULT.json'))
        save(rp,r);return r
    fp=protocol(pilot);cp=Path(construction['selected']['trajectory']['path']);assert record(cp)==construction['selected']['trajectory']
    folder.mkdir(parents=True,exist_ok=True)
    args=[ISAAC,str(ROOT/'tools/reconciled_ab/physics_engine.py'),'--direct-freeze-manifest',str(fp),'--config',str(PHYSICS_CONFIG),'--side','right','--geometry','INTERMEDIATE_PLUSH_PROXY','--profile','P14','--output-dir',str(folder),'--scripted-command-path',str(cp),'--object-spawn-side','left','--audit-robot-bin','--full-task-audit','--bin-height-m','0.150','--bin-rim-bevel-m','0.003','--headless']
    if pilot:
        args+=['--qualification-mode'];scene=load_scene(load_common_config(RESET/'config/common_config.json'));pos=source_object(scene)
        registration=dict(schema_version='common_task_frame_registration_v1',status='QUALIFICATION_ONLY',method_independent=True,episode_independent=True,changes_object_physics=False,
          base_physics_config=str(PHYSICS_CONFIG),base_physics_config_sha256=record(PHYSICS_CONFIG)['sha256'],registered_doll_center_world_xy_m=pos[:2].tolist(),registered_doll_center_world_z_m=float(pos[2]),registered_doll_orientation_quaternion_xyzw=[0.,0.,0.,1.],
          purpose='TRAIN source task geometry used by authoritative raw reference generation; not policy or outcome derived',source=record(RESET/'config/common_config.json'))
        regpath=folder/'TRAIN_TASK_REGISTRATION.json';save(regpath,registration);args+=['--object-registration-config',str(regpath)]
        entry=dict(A_B_identical_object_pose=True,source_recording_id=row['source_recording_id'],target_object_pose=dict(position_xyz_m=pos.tolist(),quaternion_xyzw=[0.,0.,0.,1.]),scope='TRAIN_PILOT')
        stable=row['source_recording_id']
    else:
        entry=next(x for x in read(REG)['entries'] if x['eval_index']==row['index']);stable=entry['stable_episode_id'];args+=['--episode-registration-manifest',str(REG),'--episode-stable-id',stable]
    inv=dict(command=args,case=row,method='REFERENCE_'+row['representation_mode'],eval_index=row['index'],stable_episode_id=stable,episode_registration=entry,commands=record(cp),freeze=record(fp),is_dev35=not pilot)
    save(folder/'INVOCATION_MANIFEST.json',inv)
    required=['event_log.npz','trial_result.json','DIRECT_EXECUTION_RUNTIME_SUMMARY.json','robot_bin_contacts.npz']
    for retry in range(3):
        if all((folder/x).exists() for x in required):break
        with (folder/f'ENGINE_{retry}.log').open('w') as logf:proc=subprocess.run(args,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=logf,stderr=subprocess.STDOUT)
        print('PHYSICS_ENGINE',key,retry,proc.returncode,flush=True)
    if not all((folder/x).exists() for x in required):
        r=dict(case=row,outcome='INFRASTRUCTURE_INVALID',actual_physics_executed=(folder/'event_log.npz').exists(),physical_valid=False,physical_stages={s:None for s in STAGES},cumulative_pipeline={s:None for s in STAGES},first_failure_stage='INFRASTRUCTURE_INVALID',reason='Incomplete trace or metadata after bounded retries');save(rp,r);return r
    return score(folder)

def main():
    rows=read(PRIOR/'CASE_MANIFEST.json')
    if sys.argv[1]=='pilot':
        while not (RUN/'components/COMPONENT_QUALIFICATION.json').exists():time.sleep(15)
        keys=['TRAIN40_WRIST_000','TRAIN40_INTERACTION_010']
        reports=[run_one(next(r for r in rows if r['key']==key),True) for key in keys]
        valid=all(r['outcome'] not in ('INFRASTRUCTURE_INVALID','PHYSICAL_EXECUTION_INVALID_OUTCOME_UNKNOWN') for r in reports)
        save(RUN/'components/TRAIN_PILOT_RESULT.json',dict(status='PASS' if valid else 'PHYSICS_VALIDITY_REVIEW_REQUIRED',results=[record(RUN/'train_pilot'/k/'RESULT.json') for k in keys],physical_rollouts=sum(bool(r['actual_physics_executed']) for r in reports),task_success_not_required=True));return
    assert read(RUN/'components/TRAIN_PILOT_RESULT.json')['status']=='PASS'
    protocol()
    for row in [r for r in rows if r['group']=='DEV35']:
        while not (RUN/'construction'/row['key']/'RESULT.json').exists():time.sleep(15)
        r=run_one(row);print('REFERENCE_CASE_COMPLETE',row['key'],r['outcome'],flush=True)
    save(RUN/'REFERENCE_BATCH_COMPLETE.json',dict(coverage=70,results=[record(p) for p in sorted((RUN/'reference_physics').glob('*/RESULT.json'))]))

if __name__=='__main__':main()
