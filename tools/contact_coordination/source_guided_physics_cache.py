"""Reuse actual execution only for identical inputs within this repair run.

Output-directory names are provenance, not physical inputs. Every array,
scene value, method/source identity and frozen execution dependency must agree.
Physical success is never a cache key or a selection criterion.
"""
from pathlib import Path
import shutil
import numpy as np
from .io import read,record,atomic_json


def same_commands(first,second):
    with np.load(first,allow_pickle=False) as a,np.load(second,allow_pickle=False) as b:
        return set(a.files)==set(b.files) and all(a[k].dtype==b[k].dtype and np.array_equal(a[k],b[k]) for k in a.files)


def scene_value(value):
    if isinstance(value,dict):
        if set(value)=={'path','bytes','sha256'}:
            if record(value['path'])!=value:raise ValueError('Changed scene provenance dependency')
            return dict(bytes=value['bytes'],sha256=value['sha256'])
        return {k:scene_value(v) for k,v in value.items()}
    if isinstance(value,list):return [scene_value(v) for v in value]
    return value


def find_equivalent(out,plan,method):
    out=Path(out);plan=Path(plan);sid=read(plan/'PLAN.json')['source_id']
    for cfg_path in sorted((out/'physical_attempts').glob('*/input/FULL_ATTEMPT_CONFIG.json')):
        folder=cfg_path.parent.parent;cfg=read(cfg_path)
        if (cfg['source_id'],cfg['method_key'],cfg['evidence_channel'])!=(sid,method,'OFFICIAL_NOMINAL'):continue
        if cfg.get('contract_test_only') or cfg.get('injected_nonfatal_failure_frame') is not None:continue
        if not (folder/'PROCESS.json').exists() or read(folder/'PROCESS.json')['returncode']!=0:continue
        if not (folder/'FULL_ATTEMPT_RECORDING.json').exists():continue
        recording=read(folder/'FULL_ATTEMPT_RECORDING.json')
        if recording['recording_end_reason']!='COMPLETE_DECLARED_HORIZON':continue
        try:
            deps=read(folder/'DEPENDENCIES.json')['files']
            if not deps or any(record(d['path'])!=d for d in deps):continue
            original=cfg['source_commands']
            if record(original['path'])!=original or not same_commands(original['path'],plan/'COMMANDS.npz'):continue
            if scene_value(read(folder/'input/SOURCE_SCENE.json'))!=scene_value(read(plan/'SOURCE_SCENE.json')):continue
            with np.load(plan/'COMMANDS.npz') as data:nominal=len(data['stage'])
            if cfg['nominal_frames']!=nominal or cfg['observation_s']!=read(out/'ABC_STUDY.json')['nominal_settle_observation_s']:continue
            if recording['nominal_frames']!=nominal or recording['requested_recording_frames']!=nominal+round(cfg['observation_s']*30):continue
        except (OSError,ValueError,KeyError):continue
        # Preserve previously derived audits before the common scorer rechecks
        # this raw trace against the current plan's geometry provenance.
        from .scientific_cache import digest
        archive=folder/'PRE_REBIND_AUDITS'/digest(str(plan))[:12];archive.mkdir(parents=True,exist_ok=True)
        saved=[]
        for name in ('ABC_NOMINAL_SCORE.json','HYBRID_SCORE.json','MEASURED_GEOMETRY_AUDIT.json',
            'ADAPTIVE_COMMAND_GEOMETRY_AUDIT.json','PREGRASP_OBJECT_PROTECTION.json','ISSUED_COMMAND_TIMING_AUDIT.json'):
            src=folder/name;dst=archive/name
            if src.exists() and not dst.exists():shutil.copy2(src,dst)
            if dst.exists():saved.append(record(dst))
        receipt=plan/'PHYSICS_EXECUTION_EQUIVALENCE.json'
        atomic_json(receipt,dict(status='IDENTICAL_EXECUTION_INPUTS',source_id=sid,method=method,
            current_commands=record(plan/'COMMANDS.npz'),actual_executed_source_commands=original,
            exact_array_keys_dtypes_values=True,scene_values_and_source_evidence_equal=True,
            all_frozen_execution_dependencies_current=True,dependencies=deps,
            actual_trace=record(folder/'event_log.npz'),actual_recording=record(folder/'FULL_ATTEMPT_RECORDING.json'),
            preserved_prior_audits=saved,same_repair_run_only=True,physical_outcomes_used_for_reuse=False,
            selection_rule='First deterministic matching source/method trial; outcome not inspected',fresh_physx_invocation=False))
        return folder,receipt
    return None
