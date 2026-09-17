#!/usr/bin/env python3
"""One paired-source ACT builder. Never substitutes a failed action target."""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_io import DEST,read,atomic_json,atomic_text,file_record,cases
AUDIT=DEST/'action_dataset_audit';BASE=DEST/'act_datasets'
CAM='observation.images.cam_high'

def build(representation_mode):
    paired=read(AUDIT/'PAIRED_TRAIN_SET_MANIFEST.json')['included_ids']
    if not paired:raise RuntimeError('No paired executable targets; no dataset is fabricated')
    folder=BASE/representation_mode;done=BASE/f'{representation_mode}_DATASET_MANIFEST.json'
    if done.exists():return read(done)
    old=ROOT/'datasets/doll_handoff_fair_a_train40';info=read(old/'meta/info.json')
    features={k:copy.deepcopy(info['features'][k]) for k in ('action','observation.state',CAM)}
    features[CAM].pop('info',None)
    source_rows={r['index']:r for r in cases() if r['group']=='TRAIN40' and r['representation_mode']==representation_mode}
    src_info=read(ROOT/'raw_recordings'/source_rows[paired[0]]['source_recording_id']/'meta/info.json')
    names=src_info['features']['observation.state'].get('names') or [f'source_state_{i}' for i in range(14)]
    if isinstance(names,dict):names=next(iter(names.values()))
    assert len(names)==14
    features['observation.state']['names']=list(names)+[f'zero_padding_{i}' for i in range(14)]
    tasks=pq.read_table(old/'meta/tasks.parquet').to_pylist();assert len(tasks)==1
    task=tasks[0]['__index_level_0__']
    repo_id='local/final_single_variable_'+representation_mode.lower()
    encoder=RGBEncoderConfig(vcodec='h264',crf=18,preset='veryfast')
    kwargs=dict(repo_id=repo_id,root=folder,video_backend='torchcodec',image_writer_threads=2,encoder_threads=1,rgb_encoder=encoder)
    if (folder/'meta/info.json').exists():dataset=LeRobotDataset.resume(**kwargs)
    else:dataset=LeRobotDataset.create(fps=30,features=features,robot_type='g1_dex3_source_conditioned_offline',metadata_buffer_size=1,**kwargs)
    complete=int(dataset.meta.total_episodes);assert complete<=len(paired)
    rows=[]
    try:
        for index_in_dataset,index in enumerate(paired):
            row=source_rows[index];target=AUDIT/'corrected_targets'/f'{representation_mode}_{index:03d}.npz'
            with np.load(target) as z:actions=z['action'];state=z['observation_state']
            assert len(actions)==len(state) and actions.shape[1:]==state.shape[1:]==(28,)
            source=ROOT/'raw_recordings'/row['source_recording_id']/'images'/CAM/'episode_000000'
            images=sorted(source.glob('frame_*.png'));assert len(images)==len(actions)-21
            digest=hashlib.sha256()
            for image in images:digest.update(bytes.fromhex(file_record(image)['sha256']))
            record=dict(dataset_episode_index=index_in_dataset,source_index=index,source_recording_id=row['source_recording_id'],target=file_record(target),source_image_sequence_sha256=digest.hexdigest(),source_image_count=len(images),prefix_frames=21,frames=len(actions))
            rows.append(record)
            if index_in_dataset<complete:continue
            for f,(a,s) in enumerate(zip(actions,state)):
                image=images[max(0,f-21)]
                with Image.open(image) as im:rgb=np.asarray(im.convert('RGB'))
                dataset.add_frame({'task':task,'action':a.astype(np.float32),'observation.state':s.astype(np.float32),CAM:rgb})
            dataset.save_episode(parallel_encoding=False)
            atomic_json(BASE/f'{representation_mode}_PROGRESS.json',dict(completed=index_in_dataset+1,paired_ids=paired,episodes=rows))
            print('PAIRED_DATASET_EPISODE',representation_mode,index,flush=True)
        dataset.finalize()
    finally:
        dataset.finalize()
    records=[file_record(p) for p in sorted(folder.rglob('*')) if p.is_file() and 'images' not in p.relative_to(folder).parts]
    result=dict(root=str(folder),repo_id=repo_id,episodes=rows,paired_ids=paired,records=records,builder=file_record(Path(__file__)),source_state_contract='Same original ALOHA14 plus14 zeros; not measured target G1 state',task=task)
    atomic_json(done,result);return result

def parity():
    a=read(BASE/'WRIST_DATASET_MANIFEST.json');b=read(BASE/'INTERACTION_DATASET_MANIFEST.json')
    assert a['paired_ids']==b['paired_ids'] and a['task']==b['task']
    left=pq.read_table(sorted((Path(a['root'])/'data').rglob('*.parquet')));right=pq.read_table(sorted((Path(b['root'])/'data').rglob('*.parquet')))
    assert left.column_names==right.column_names and len(left)==len(right)
    common=[]
    for k in left.column_names:
        if k=='action':continue
        assert left[k].equals(right[k]),f'Unintended paired tensor confound: {k}'
        common.append(k)
    av=np.asarray(left['action'].to_pylist(),np.float32);bv=np.asarray(right['action'].to_pylist(),np.float32)
    assert np.array_equal(av[:,14:],bv[:,14:]),'Dex3 timing/command confound'
    for x,y in zip(a['episodes'],b['episodes']):
        assert x['source_recording_id']==y['source_recording_id'] and x['source_image_sequence_sha256']==y['source_image_sequence_sha256'] and x['frames']==y['frames']
    sa=read(Path(a['root'])/'meta/stats.json');sb=read(Path(b['root'])/'meta/stats.json')
    for k in sa:
        if k!='action':assert sa[k]==sb[k],f'Non-action normalization confound: {k}'
    videos_a=sorted((Path(a['root'])/'videos').rglob('*.mp4'));videos_b=sorted((Path(b['root'])/'videos').rglob('*.mp4'));assert len(videos_a)==len(videos_b)
    video_equal=all(file_record(x)['sha256']==file_record(y)['sha256'] for x,y in zip(videos_a,videos_b))
    if not video_equal:
        # Container metadata can differ without pixels differing. Decode exact
        # frames only when needed, using the same decoder for both streams.
        import av as pyav
        for x,y in zip(videos_a,videos_b):
            with pyav.open(str(x)) as ca,pyav.open(str(y)) as cb:
                xa=ca.decode(video=0);xb=cb.decode(video=0)
                import itertools
                for fa,fb in itertools.zip_longest(xa,xb):assert fa is not None and fb is not None and np.array_equal(fa.to_ndarray(format='rgb24'),fb.to_ndarray(format='rgb24'))
    result=dict(status='PASS',unintended_dataset_confounds=0,common_columns=common,frames=len(av),episodes=len(a['paired_ids']),changed_action_scalars=int((av!=bv).sum()),
        action_difference_class='INTENDED_REPRESENTATION_DIFFERENCE through identical frozen realization',dex3_actions_identical=True,observations_and_state_identical=True,video_container_hash_equality=video_equal,
        normalization='Same official LeRobot procedure; action statistics may differ because executable action targets differ; all non-action statistics identical',chunk_size=50,sampling_hz=30,
        manifests=[file_record(BASE/f'{m}_DATASET_MANIFEST.json') for m in ('WRIST','INTERACTION')])
    atomic_json(BASE/'A_B_CORRECTED_DATASET_PARITY_AUDIT.json',result);atomic_text(BASE/'A_B_CORRECTED_DATASET_PARITY_AUDIT.md','# Corrected paired ACT dataset parity\n\n'+json.dumps(result,indent=2)+'\n')
    print('PAIRED_DATASET_PARITY_PASS',len(a['paired_ids']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('representation_mode',choices=['WRIST','INTERACTION','PARITY']);args=p.parse_args()
    parity() if args.representation_mode=='PARITY' else build(args.representation_mode)
