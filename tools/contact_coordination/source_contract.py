"""Reuse source evidence and express Wrist priors in the shared functional TCP."""
from pathlib import Path
import numpy as np
from .io import read,record,atomic_json,atomic_npz
from .source_phase import COMMON,load_source,phase_record


def registered_functional_wrists(wrists,old_wrist_to_tool,shared_wrist_to_tool):
    """Preserve every registered source tool pose; apply fixed tool change once."""
    return np.asarray(wrists)@np.asarray(old_wrist_to_tool)@np.linalg.inv(shared_wrist_to_tool)


def run(out,split_name='TRAIN40',source_ids=None):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));cal=read(out/'target_repair/CONTACT_CALIBRATION.json')
    normalization_path=out/'target_repair/TASK_FRAME_NORMALIZATION.json'
    normalization=read(normalization_path) if normalization_path.exists() else None
    if split_name not in ('TRAIN40','DEV35'):raise ValueError('Unknown authorized source split')
    ids=read(out/'SPLIT_CONTRACT.json')['authorized_training_source_ids' if split_name=='TRAIN40' else 'evaluation_source_ids'];entries=read(out/'bootstrap/SPLITS.json')['entries'];results=[]
    if source_ids is not None:
        if not set(source_ids)<=set(ids):raise ValueError('Source subset outside declared split')
        ids=list(source_ids)
    for sid in ids:
        entry=next(e for e in entries if e['source_recording_id']==sid)
        assert bool(entry['TRAIN40'])==(split_name=='TRAIN40') and bool(entry['DEV35_DIAGNOSTIC35'])==(split_name=='DEV35')
        folder=out/'source_phase'/sid
        try:
            raw=load_source(entry)
            from .scientific_cache import digest,content_hash,file_payload
            from .source_phase import CALIBRATION,PHYSICS
            from .io import ROOT
            registration=read(CALIBRATION)
            additional=read(ROOT/'outputs/doll_handoff_dataset_b_final/new_episode_conversion/source_image_object_estimates.json')
            measurement=next((v for v in registration['per_episode'] if v['source_name']==sid),None)
            if measurement is None:measurement=next((v for v in additional.values() if sid in str(v['observations'])),None)
            source_key=digest(dict(source_id=sid,raw_sha256=entry['source_parquet_sha256'],source_arrays=raw,
                registration=registration,physics=read(PHYSICS),common=cfg,task_normalization=normalization,
                additional_registration=additional,scene_layout=read(ROOT/'isaaclab_doll_handoff_scene/scene_layout.json'),
                image_hashes=[content_hash(ROOT/v['image']) for v in measurement['observations']] if measurement else [],
                source_code={name:content_hash(ROOT/'tools/contact_coordination'/name) for name in ('source_phase.py','source_contract.py')},
                registration_code=content_hash(ROOT/'tools/build_eval35_episode_object_registration.py'),
                model_code=content_hash(ROOT/'tools/doll_handoff_retargeting/models.py'),
                model_xml=content_hash(cfg['models']['g1_xml'])))
            cache=folder/'SOURCE_CACHE_CONTRACT.json'
            if cache.exists() and read(cache)['key']==source_key:
                for artifact in read(cache)['artifacts']:
                    if record(artifact['path'])!=artifact:raise ValueError('Changed source cache result')
                phase=read(folder/'PHASE_RECORD.json');pri=dict(np.load(folder/'SOURCE_PRIORS.npz'));wrists={s:pri[s+'_wrist_world'] for s in ('left','right')}
                assert phase['source_id']==sid and phase['provenance']['raw']['sha256']==entry['source_parquet_sha256']
            else:
                phase,wrists,objects=phase_record(entry,raw,g,normalization);atomic_json(folder/'PHASE_RECORD.json',phase)
                atomic_npz(folder/'SOURCE_PRIORS.npz',source_timestamp=raw['source_timestamp'],left_wrist_world=wrists['left'],right_wrist_world=wrists['right'],inferred_object_from_left=objects,**{k:v for k,v in raw.items() if k.startswith('source_fk_')})
                atomic_json(cache,dict(key=source_key,artifacts=[record(folder/name) for name in ('PHASE_RECORD.json','SOURCE_PRIORS.npz')]))
            adapted={};errors={};transforms={}
            for side in ('left','right'):
                old=np.asarray(raw[f'WRIST_{side}_wrist_to_tool']);new=np.asarray(cal['contacts']['handoff_'+side]['T_wrist_H'])
                adapted[side]=registered_functional_wrists(wrists[side],old,new)
                difference=adapted[side]@new-wrists[side]@old
                errors[side]=float(np.max(np.abs(difference)));assert errors[side]<1e-12
                transforms[side]=dict(original_wrist_to_tool=old,shared_wrist_to_tool=new)
            atomic_npz(folder/'FUNCTIONAL_WRIST_PRIORS.npz',source_timestamp=raw['source_timestamp'],**{s+'_wrist_world':v for s,v in adapted.items()})
            audit=dict(status='REGISTERED_FUNCTIONAL_TOOL_POSES_PRESERVED',source_id=sid,source_reference=record(entry['current_rebuild_raw_cartesian_reference']['path']),phase=record(folder/'PHASE_RECORD.json'),calibration=record(out/'target_repair/CONTACT_CALIBRATION.json'),transforms=transforms,max_SE3_matrix_difference=errors,independent_hand_rebasing=False,spatial_contact_optimization=False,source_reference_path_changed=False,interpretation='Wrist origins change only to express the same registered functional tool poses with the current shared calibrated G1 tool frame.')
            atomic_json(folder/'FUNCTIONAL_FRAME_AUDIT.json',audit);results.append(dict(source_id=sid,status='SOURCE_EVIDENCE_AVAILABLE',phase=record(folder/'PHASE_RECORD.json'),frame_audit=record(folder/'FUNCTIONAL_FRAME_AUDIT.json')))
        except ValueError as error:
            if 'SOURCE_EVIDENCE_MISSING' not in str(error):raise
            results.append(dict(source_id=sid,status='SOURCE_EVIDENCE_MISSING',reason=str(error)))
    name=f'{split_name}_SOURCE_CONTRACT.json' if source_ids is None else f'{split_name}_SOURCE_SUBSET_{digest(ids)[:12]}.json'
    atomic_json(out/'source_phase'/name,dict(rows=results,source_count=len(ids),source_membership_unchanged=True,physical_or_converter_outcomes_used=False))
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r=run(a.run_dir.resolve());print({s:sum(x['status']==s for x in r) for s in {x['status'] for x in r}})
