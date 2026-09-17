"""Audit the tensors ACT will consume against real G1 supervision lineage."""
from pathlib import Path
import hashlib
import numpy as np
from .io import read,record,atomic_json,atomic_text
from .target_dataset import CAM


def numeric_alignment(raw,observed):
    """Serialization equality is checked in the declared learning dtype."""
    for stored,evidence in [('action','executed_command'),('observation.state','measured_state')]:
        a=np.asarray(raw[stored],dtype=np.float32);b=np.asarray(observed[evidence],dtype=np.float32)
        if a.shape!=b.shape or not np.array_equal(a,b):raise ValueError('Effective tensor mismatch: '+stored)
    timestamp=np.asarray(raw['timestamp'],dtype=np.float32)
    expected=np.asarray(observed['timestamp_s'],dtype=np.float32)
    if not np.allclose(timestamp,expected,rtol=0,atol=1e-4):raise ValueError('Observation/action timestamp mismatch')
    return {k:hashlib.sha256(np.ascontiguousarray(raw[k],dtype='<f4').tobytes()).hexdigest()
            for k in ('action','observation.state')}


def audit(out,manifest_a,manifest_b):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    paths={'A':Path(manifest_a),'B':Path(manifest_b)};reports={};schemas={};memberships={}
    for condition,path in paths.items():
        manifest=read(path)
        for dependency in [*manifest['files'],manifest['contract']['manifest']]:
            if record(dependency['path'])!=dependency:raise ValueError('Changed dataset lineage')
        root=Path(manifest['root']);info=read(root/'meta/info.json')
        schemas[condition]={k:info['features'][k] for k in ('action','observation.state',CAM)}
        memberships[condition]=[r['source_id'] for r in manifest['episodes']]
        dataset=LeRobotDataset('local/hybrid_g1_'+condition.lower(),root=root,video_backend='torchcodec',
            delta_timestamps={'action':[i/30 for i in range(50)]})
        raw=dataset.select_columns(['action','observation.state','timestamp','episode_index','frame_index'])[:]
        episodes=[]
        for index,row in enumerate(manifest['episodes']):
            alignment=read(row['alignment']['path'])
            if record(row['alignment']['path'])!=row['alignment']:raise ValueError('Changed alignment')
            if record(alignment['observations']['path'])!=alignment['observations']:raise ValueError('Changed measured supervision')
            observed=dict(np.load(alignment['observations']['path']))
            positions=np.flatnonzero(np.asarray(raw['episode_index'])==index)
            if not np.array_equal(np.asarray(raw['frame_index'])[positions],np.arange(row['frames'])):
                raise ValueError('Episode/frame mapping differs from measured trace')
            selected={k:np.asarray(raw[k])[positions] for k in ('action','observation.state','timestamp')}
            hashes=numeric_alignment(selected,observed);examples=[]
            for local in sorted({0,len(positions)//2,len(positions)-1}):
                item=dataset[int(positions[local])]
                rgb=item[CAM].numpy();action=item['action'].numpy();mask=item['action_is_pad'].numpy()
                if rgb.shape!=(3,480,640) or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
                    raise ValueError('Actual ACT image decoder interface differs')
                if action.shape!=(50,28) or mask.shape!=(50,):raise ValueError('ACT chunk/mask schema differs')
                expected_index=np.minimum(local+np.arange(50),len(positions)-1)
                np.testing.assert_array_equal(action,np.asarray(observed['executed_command'],dtype=np.float32)[expected_index])
                np.testing.assert_array_equal(mask,local+np.arange(50)>=len(positions))
                examples.append(dict(frame=local,decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                                     image_range=[float(rgb.min()),float(rgb.max())],pad_frames=int(mask.sum())))
            episodes.append(dict(source_id=row['source_id'],frames=len(positions),effective_tensors=hashes,decoded_examples=examples))
        reports[condition]=dict(episodes=episodes,total_frames=len(dataset))
    if not memberships['A'] or memberships['A']!=memberships['B'] or schemas['A']!=schemas['B']:
        raise ValueError('A/B source membership or tensor schema differs')
    result=dict(status='VERIFIED_EFFECTIVE_TARGET_DATASET_PARITY',dataset_manifests={k:record(p) for k,p in paths.items()},
        paired_source_ids=memberships['A'],schemas=schemas,conditions=reports,implementation=record(__file__),
        actual_actions_states_images_may_differ=True,privileged_metadata_consumed=False,
        alignment='Pre-action G1 RGB/measured named28 -> executed absolute named28 at30Hz',
        chunk='50 future actions; episode-end replication masked by installed ACT dataset loader',
        legacy_source_image_checkpoint_reuse=False)
    atomic_json(out/'effective_data_audit/TARGET_DATASET_PARITY.json',result)
    atomic_text(out/'DATASET_PARITY.md',f'# Effective target-domain dataset parity\n\nMatched sources: {len(memberships["A"])} of authorized TRAIN40. All serialized action/state tensors and timestamps match their measured G1 lineage in float32. Installed ACT loading was checked at first, middle and final frames of every episode, including50-action chunks and boundary masks. G1 RGB is decoded from actual observation capture. Outcome labels remain metadata. A/B share schema, membership and alignment; their physical observations/actions may differ. Legacy source-image policies are not reused.\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--manifest-a',type=Path,required=True);p.add_argument('--manifest-b',type=Path,required=True)
    a=p.parse_args();print(audit(a.run_dir,a.manifest_a,a.manifest_b)['status'])
