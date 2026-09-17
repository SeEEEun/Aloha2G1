"""Thin LeRobot realization of verified dynamic G1 observation/action traces.

The input manifest is frozen after TRAIN conversion. It contains the identical
ordered source IDs for A and B, and references complete aligned physical runs.
This module never fills a missing observation, action, or source episode.
"""
from pathlib import Path
import hashlib
import numpy as np
from .io import read,record,atomic_json

CAM='observation.images.cam_high'


def checked_episode(row):
    for key in ('alignment','score'):
        if record(row[key]['path'])!=row[key]:raise ValueError('Changed '+key)
    alignment=read(row['alignment']['path']);score=read(row['score']['path'])
    from .demo_alignment import scored_task_success
    success=scored_task_success(score)
    if alignment.get('schema_version')!=2:raise ValueError('Explicit boolean alignment schema required')
    if not alignment['physical_validity'] or not alignment['complete_command_sequence']:
        raise ValueError('Incomplete or invalid physical supervision')
    if alignment['task_success'] is not success:raise ValueError('Task labels disagree')
    commands=read(Path(alignment['commands']['path']).parent/'PLAN.json')
    if commands.get('diagnostic',False) or commands.get('full_task') is False:
        raise ValueError('A phase diagnostic cannot stand in for a complete demonstration')
    for key in ('observations','physics_trace','commands'):
        if record(alignment[key]['path'])!=alignment[key]:raise ValueError('Changed '+key)
    data=dict(np.load(alignment['observations']['path']))
    if len(data['measured_state'])!=len(alignment['images']):raise ValueError('Missing target RGB')
    if str(np.load(alignment['commands']['path'])['stable_episode_id'])!=row['source_id']:
        raise ValueError('Source identity mismatch')
    return alignment,data,success


def build(manifest_path,condition,destination):
    from .generalization_gate import require
    require(Path(manifest_path).parent, 'target_dataset')
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig
    manifest_path=Path(manifest_path);destination=Path(destination)
    manifest=read(manifest_path)
    if manifest['status']!='FROZEN_MATCHED_TARGET_SUPERVISION':raise ValueError('Unfrozen membership')
    ids=manifest['paired_source_ids'];rows=manifest['conditions'][condition]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Expected a nonempty unique paired source set')
    if [r['source_id'] for r in rows]!=ids:raise ValueError('Condition membership differs')
    if any([r['source_id'] for r in rr]!=ids for rr in manifest['conditions'].values()):
        raise ValueError('A/B membership differs')
    contract=dict(manifest=record(manifest_path),condition=condition,implementation=record(__file__))
    receipt=destination.parent/(destination.name+'_BUILD_CONTRACT.json')
    if receipt.exists() and read(receipt)!=contract:raise ValueError('Resume dependencies changed')
    atomic_json(receipt,contract)
    verified=[checked_episode(r) for r in rows]
    names=verified[0][1]['joint_names'].tolist()
    if any(a['joint_names'].tolist()!=names for _,a,_ in verified):raise ValueError('Joint order changed')
    features={k:dict(dtype='float32',shape=(28,),names=names) for k in ('action','observation.state')}
    features[CAM]=dict(dtype='video',shape=(480,640,3),names=['height','width','channels'])
    kwargs=dict(repo_id='local/hybrid_g1_'+condition.lower(),root=destination,video_backend='torchcodec',
                image_writer_threads=2,encoder_threads=1,rgb_encoder=RGBEncoderConfig(vcodec='h264',crf=18,preset='veryfast'))
    dataset=(LeRobotDataset.resume(**kwargs) if (destination/'meta/info.json').exists()
             else LeRobotDataset.create(fps=30,features=features,robot_type='g1_dex3_dynamic',metadata_buffer_size=1,**kwargs))
    completed=int(dataset.meta.total_episodes)
    if completed>len(rows):raise ValueError('Unexpected existing episode count')
    lineage=[]
    try:
        for index,(row,(alignment,data,success)) in enumerate(zip(rows,verified,strict=True)):
            digest=hashlib.sha256()
            for key in ('measured_state','executed_command','timestamp_s'):
                digest.update(key.encode());digest.update(np.ascontiguousarray(data[key],dtype='<f8').tobytes())
            lineage.append(dict(source_id=row['source_id'],dataset_episode_index=index,frames=len(data['measured_state']),
                                full_task_success=success,expert_label=success,effective_numeric_sha256=digest.hexdigest(),
                                alignment=row['alignment']))
            if index<completed:continue
            for state,action,image in zip(data['measured_state'],data['executed_command'],alignment['images'],strict=True):
                if record(image['path'])!=image:raise ValueError('Changed target observation')
                with Image.open(image['path']) as im:rgb=np.asarray(im.convert('RGB'))
                dataset.add_frame({'task':manifest['task_instruction'],'action':action.astype(np.float32),
                                   'observation.state':state.astype(np.float32),CAM:rgb})
            dataset.save_episode(parallel_encoding=False)
            atomic_json(destination.parent/(destination.name+'_PROGRESS.json'),dict(contract=contract,episodes=lineage))
            print('G1_DATASET_EPISODE',condition,index+1,len(rows),row['source_id'],flush=True)
    finally:
        dataset.finalize()
    result=dict(contract=contract,root=str(destination.resolve()),episodes=lineage,
                observation_alignment='Current G1 RGB and measured named28 before the executed absolute named28 command at30Hz',
                privileged_source_and_contact_metadata_consumed_by_ACT=False,
                task_failure_labels_retained_separately=True,
                files=[record(p) for p in sorted(destination.rglob('*')) if p.is_file()])
    atomic_json(destination.parent/(destination.name+'_MANIFEST.json'),result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--condition',choices=['A','B'],required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();build(a.manifest,a.condition,a.destination)
