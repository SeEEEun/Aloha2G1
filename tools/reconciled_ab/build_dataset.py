"""One ACT builder: real source observations, method-consistent G1 state."""
import copy,hashlib,sys
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig
from .common import *
CAM='observation.images.cam_high'

def build(mode):
    paired=read(RUN/'act/PAIRED_TRAIN_SET_MANIFEST.json')['included_ids'];assert paired
    folder=RUN/'datasets'/mode;done=RUN/f'act/{mode}_DATASET_MANIFEST.json'
    if done.exists():return
    old=ROOT/'datasets/doll_handoff_fair_a_train40';info=read(old/'meta/info.json');features={k:copy.deepcopy(info['features'][k]) for k in ('action','observation.state',CAM)};features[CAM].pop('info',None)
    tasks=pq.read_table(old/'meta/tasks.parquet').to_pylist();assert len(tasks)==1;task=tasks[0]['__index_level_0__']
    repo='local/reconciled_'+mode.lower();kwargs=dict(repo_id=repo,root=folder,video_backend='torchcodec',image_writer_threads=2,encoder_threads=1,rgb_encoder=RGBEncoderConfig(vcodec='h264',crf=18,preset='veryfast'))
    dataset=LeRobotDataset.resume(**kwargs) if (folder/'meta/info.json').exists() else LeRobotDataset.create(fps=30,features=features,robot_type='g1_dex3_source_conditioned_offline',metadata_buffer_size=1,**kwargs)
    complete=int(dataset.meta.total_episodes);rows=[]
    try:
        for number,index in enumerate(paired):
            result=read(RUN/'construction'/f'TRAIN40_{mode}_{index:03d}'/'RESULT.json');assert result['selected'];row=result['case'];target=result['selected']['trajectory']
            with np.load(target['path']) as z:actions=z['commanded_q_rad'][21:].astype(np.float32)
            states=np.vstack((actions[:1],actions[:-1]));source=ROOT/'raw_recordings'/row['source_recording_id']/'images'/CAM/'episode_000000';images=sorted(source.glob('frame_*.png'));assert len(images)==len(actions)
            digest=hashlib.sha256()
            for p in images:digest.update(bytes.fromhex(record(p)['sha256']))
            item=dict(dataset_episode_index=number,index=index,recording=row['source_recording_id'],frames=len(actions),action_target=target,source_image_sequence_sha256=digest.hexdigest(),synthetic_observations=0,learned_preparation_frames=0,external_execution_preparation_s=.7)
            rows.append(item)
            if number<complete:continue
            for image,a,s in zip(images,actions,states):
                with Image.open(image) as im:rgb=np.asarray(im.convert('RGB'))
                dataset.add_frame({'task':task,'action':a,'observation.state':s,CAM:rgb})
            dataset.save_episode(parallel_encoding=False);save(RUN/f'act/{mode}_DATASET_PROGRESS.json',dict(completed=number+1,rows=rows));print('DATASET_EPISODE',mode,index,flush=True)
        dataset.finalize()
    finally:dataset.finalize()
    save(done,dict(root=str(folder),repo_id=repo,paired_ids=paired,episodes=rows,task=task,files=[record(p) for p in sorted(folder.rglob('*')) if p.is_file() and 'images' not in p.relative_to(folder).parts],builder=record(Path(__file__)),state_semantics='state[0]=action[0]; state[t]=action[t-1], method-consistent28D absolute G1/Dex3 q'))

def parity():
    a=read(RUN/'act/WRIST_DATASET_MANIFEST.json');b=read(RUN/'act/INTERACTION_DATASET_MANIFEST.json');assert a['paired_ids']==b['paired_ids'] and a['task']==b['task']
    ta=pq.read_table(sorted((Path(a['root'])/'data').rglob('*.parquet')));tb=pq.read_table(sorted((Path(b['root'])/'data').rglob('*.parquet')));assert ta.column_names==tb.column_names and len(ta)==len(tb)
    for k in ta.column_names:
        if k not in ('action','observation.state'):assert ta[k].equals(tb[k]),k
    aa=np.array(ta['action'].to_pylist(),np.float32);ab=np.array(tb['action'].to_pylist(),np.float32);assert np.array_equal(aa[:,14:],ab[:,14:])
    for table in (ta,tb):
        acts=np.array(table['action'].to_pylist(),np.float32);state=np.array(table['observation.state'].to_pylist(),np.float32);ep=np.array(table['episode_index'])
        for e in np.unique(ep):
            rows=ep==e;v=acts[rows];assert np.array_equal(state[rows],np.vstack((v[:1],v[:-1])))
    for x,y in zip(a['episodes'],b['episodes']):assert x['recording']==y['recording'] and x['source_image_sequence_sha256']==y['source_image_sequence_sha256'] and x['frames']==y['frames']
    # Same deterministic encoder, verify actual decoded pixels if container bytes differ.
    va=sorted((Path(a['root'])/'videos').rglob('*.mp4'));vb=sorted((Path(b['root'])/'videos').rglob('*.mp4'));assert len(va)==len(vb)
    for x,y in zip(va,vb):
        if record(x)['sha256']==record(y)['sha256']:continue
        import av,itertools
        with av.open(str(x)) as ca,av.open(str(y)) as cb:
            for fa,fb in itertools.zip_longest(ca.decode(video=0),cb.decode(video=0)):assert fa is not None and fb is not None and np.array_equal(fa.to_ndarray(format='rgb24'),fb.to_ndarray(format='rgb24'))
    save(RUN/'act/DATASET_PARITY.json',dict(status='PASS',unintended_confounds=0,paired_count=len(a['paired_ids']),state_schema_equal=True,state_values_consistent_with_own_actions=True,source_observations_identical=True,dex3_nominal_commands_identical=True,normalization='Same official procedure; per-method state/action statistics legitimately differ',changed_action_scalars=int(np.sum(aa!=ab)),manifests=[record(RUN/f'act/{m}_DATASET_MANIFEST.json') for m in ('WRIST','INTERACTION')]))

if __name__=='__main__':parity() if sys.argv[1]=='PARITY' else build(sys.argv[1])
