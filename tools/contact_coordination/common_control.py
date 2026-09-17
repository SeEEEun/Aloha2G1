"""Fresh, bounded replay of common embodiment controls; never a method result."""
import os, subprocess, time
from pathlib import Path
from .io import ROOT, ISAAC, atomic_json, read, record, fingerprint

OLD=ROOT/'outputs/final_episode_registered_eval35/00_qualification'

def run(out,resume=False):
    folder=out/'common_control';folder.mkdir(parents=True,exist_ok=True)
    old=read(OLD/'PROVISIONAL_QUALIFICATION_FREEZE.json')
    deps=[Path(r['path']) for r in old['files']]
    deps += [ROOT/'tools/common_execution_layer.py',ROOT/'tools/finalize_common_dex3_grasp_qualification.py']
    key,files=fingerprint(deps)
    frozen=dict(old,files=files,purpose='Fresh common-control qualification for contact-coordination v1; no source-method result')
    freeze=folder/f'CURRENT_DEPENDENCIES_{key[:12]}.json';atomic_json(freeze,frozen)
    prior=read(ROOT/'outputs/final_single_variable_ab/00_contract/OFFLINE_EXECUTION_ENVIRONMENT.json')
    drift=[dict(path=r['path'],historical=r['sha256'],current=record(r['path'])['sha256']) for r in old['files'] if record(r['path'])['sha256']!=r['sha256']]
    atomic_json(folder/'HISTORICAL_COMPATIBILITY.json',dict(dependency_drift=drift,fresh_runs_required=True,model_files=prior['models']))
    from tools.finalize_common_dex3_grasp_qualification import standalone,full_task
    results=[]
    for side in ['left','right','scripted']:
        dest=folder/f'{side}_{key[:12]}';dest.mkdir(exist_ok=True)
        result_path=dest/'QUALIFICATION.json'
        if resume and result_path.exists():
            results.append(read(result_path));continue
        command=OLD/'contact_seeking_commands'/('scripted_full_task_contact_seeking.npz' if side=='scripted' else f'{side}_standalone_contact_seeking.npz')
        args=[ISAAC,str(ROOT/'tools/run_direct_physical_execution_isaac.py'),'--qualification-mode','--direct-freeze-manifest',str(freeze),'--config',str(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json'),'--side','right' if side=='scripted' else side,'--geometry','INTERMEDIATE_PLUSH_PROXY','--profile','P14','--output-dir',str(dest),'--scripted-command-path',str(command),'--object-spawn-side','left' if side=='scripted' else side,'--headless']
        if side=='scripted':args+=['--object-registration-config',str(OLD/'SCRIPTED_QUALIFICATION_OBJECT_REGISTRATION.json'),'--audit-robot-bin','--full-task-audit','--bin-height-m','0.150','--bin-rim-bevel-m','0.003']
        atomic_json(dest/'INVOCATION.json',dict(command=args,cache_key=key,command_file=record(command),type='COMMON_CONTROL',method=None))
        attempts=[]
        for attempt in range(3):
            start=time.monotonic()
            with (dest/f'engine_{attempt}.log').open('w') as log:
                try:
                    proc=subprocess.run(args,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,timeout=360)
                    code=proc.returncode
                except subprocess.TimeoutExpired:code='TIMEOUT'
            attempts.append(dict(attempt=attempt,returncode=code,wall_sec=time.monotonic()-start))
            atomic_json(dest/'ATTEMPTS.json',attempts)
            if all((dest/n).exists() for n in ['trial_result.json','event_log.npz','DIRECT_EXECUTION_RUNTIME_SUMMARY.json']):break
        if not (dest/'event_log.npz').exists():
            result=dict(status='INFRASTRUCTURE_INVALID',run=str(dest),attempts=attempts)
        else:
            result=full_task(dest) if side=='scripted' else standalone(dest,side)
            result.update(control_type=side,cache_key=key,attempts=attempts,method_result=False)
        atomic_json(result_path,result);results.append(result)
        print('COMMON_CONTROL',side,result['status'],flush=True)
    result=dict(status='VALIDATED' if all(x['status']=='PASS' for x in results) else 'UNVERIFIED',results=results,cache_key=key,physical_runs=sum(len(r.get('attempts',[])) for r in results),natural_start_qualification=False)
    atomic_json(folder/'RESULT.json',result)
    return result
