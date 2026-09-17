"""Existing measured-state renderer with explicit current config and ACT labels."""
from pathlib import Path
import os
import subprocess
import json
import numpy as np
from .io import ROOT,read,record,atomic_json


def decoded_video(path,frames=None):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames','-of','json',str(path)],text=True))['streams'][0]
    if frames is not None and int(data['nb_read_frames'])!=frames:raise ValueError('Video frame count differs from trace')
    if data['avg_frame_rate']!='30/1':raise ValueError('Physical replay timing changed')
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    return data


def encode_captured(folder,target,label):
    """Exact captured real-state images; no physics or fabricated object motion."""
    expected=dict(trace=record(folder/'event_log.npz'),label=label,adapter=record(__file__))
    receipt=target.with_suffix('.json')
    if target.exists() and receipt.exists():
        old=read(receipt)
        if old.get('dependencies')==expected and old['video']==record(target):
            decoded_video(target,old['frames']);return old
        raise ValueError('Changed captured replay dependencies; preserve the old evidence')
    trace=dict(np.load(folder/'event_log.npz'));frames=len(np.unique(trace['control_frame']))
    images=folder/'observations';paths=sorted(images.glob('frame_*.png'))
    if len(paths)!=frames:raise ValueError('Captured image count is not the measured control-frame count')
    label_path=target.with_suffix('.txt');label_path.write_text(label+'\nActual measured G1 / dynamic object; physical time preserved\n')
    temporary=target.with_name(target.stem+'.incomplete.mp4')
    args=['ffmpeg','-v','error','-y','-framerate','30','-i',str(images/'frame_%06d.png'),
          '-vf',f'drawbox=x=0:y=0:w=iw:h=52:color=white@0.85:t=fill,drawtext=textfile={label_path}:x=8:y=6:fontsize=13:fontcolor=black:line_spacing=3',
          '-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(temporary)]
    subprocess.run(args,check=True);decoded=decoded_video(temporary,frames);temporary.replace(target)
    result=dict(video=record(target),trace=record(folder/'event_log.npz'),frames=frames,decoded=decoded,dependencies=expected,
                rendering='Actual observation images captured before actions; saved simulation states only',physics_rerun=False)
    atomic_json(receipt,result);return result


def renderer_class():
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    from tools import render_final_episode_registered_physical_evidence as legacy
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from .source_phase import COMMON,PHYSICS
    class Renderer(legacy.PhysicalRenderer):
        def __init__(self):
            # Explicit config adapter; no global mutation of legacy settings.
            self.common=load_common_config(COMMON);self.layout=load_scene(self.common);self.config=read(PHYSICS)
            self.g1=G1Kinematics(self.common,self.layout);self.model=self.g1.model;self.data=mujoco.MjData(self.model)
            self.renderer=mujoco.Renderer(self.model,width=540,height=405)
            contract=read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json')
            names=[r['joint_name'] for r in contract['joint_specs']]
            model_names=[*self.g1.arm_joint_names,*self.g1.hand_joint_names['left'],*self.g1.hand_joint_names['right']]
            if set(names)!=set(model_names):raise ValueError('Named renderer joint mismatch')
            self.reorder=np.asarray([names.index(n) for n in model_names])
            presets=self.layout['camera']['presets']
            self.cameras={k:legacy.camera(presets[k]['eye_world_xyz_m'],presets[k]['target_world_xyz_m']) for k in ('top','overview')}
            self.root_position=np.asarray(self.layout['g1']['root_position_world_xyz_m'])
            self.root_quaternion=np.asarray(self.layout['g1']['root_orientation_world_wxyz'])
            self.model.vis.headlight.ambient[:]=(.52,.52,.52);self.model.vis.headlight.diffuse[:]=(.78,.78,.78);self.model.vis.headlight.specular[:]=(.08,.08,.08)
        def measured_state(self,trace,index):
            self.set_state(trace['MEASURED_Q'][index],trace['object_position_world_m'][index],trace['object_quaternion_xyzw'][index])
            for name,value in zip(trace['all_joint_names'],trace['all_measured_q_rad'][index],strict=True):
                j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,str(name))
                if j<0:raise ValueError('Unmapped measured articulation joint')
                self.data.qpos[self.model.jnt_qposadr[j]]=value
            mujoco.mj_forward(self.model,self.data)
    return Renderer


def mosaic(out,condition,view,rows,source_ids):
    import cv2
    target=out/'replays'/f'ACT_{condition}_DEV35_{view.upper()}_35SPLIT.mp4'
    keyed={r['source_id']:r for r in rows if r['condition']==condition};traces=[];indices=[]
    for sid in source_ids:
        row=keyed.get(sid);path=Path(row['attempt'])/'event_log.npz' if row and row.get('attempt') else None
        if path is not None and path.exists():
            a=dict(np.load(path))
            if not len(a['control_frame']):traces.append(None);indices.append(None);continue
            if not {'MEASURED_Q','all_measured_q_rad','all_joint_names'}<=a.keys():raise ValueError('Full measured policy state missing')
            ix=np.r_[np.flatnonzero(np.diff(a['control_frame'])!=0),len(a['control_frame'])-1]
            traces.append(a);indices.append(ix)
        else:traces.append(None);indices.append(None)
    actual=sum(t is not None for t in traces)
    if actual==0:return dict(file=str(target),status='NOT_RENDERED_NO_ACT_PHYSICAL_TRACES',all_card_video_created=False)
    frames=max(len(i) for i in indices if i is not None)
    expected=dict(trace_hashes=[record(Path(keyed[sid]['attempt'])/'event_log.npz') for sid,t in zip(source_ids,traces) if t is not None],
                  adapter=record(__file__),camera=view,ordered_source_ids=source_ids,ledger=record(out/'ACT_DEV35/LEDGER.json'))
    receipt=target.with_suffix('.json')
    if receipt.exists() and target.exists():
        old=read(receipt)
        if old['dependencies']==expected and old['video']==record(target):return old
        raise ValueError('Replay dependencies changed; preserve the old video')
    renderer=renderer_class()();temporary=target.with_name(target.stem+'.incomplete.mp4')
    args=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r','30','-i','-',
          '-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(temporary)]
    process=subprocess.Popen(args,stdin=subprocess.PIPE)
    try:
        for frame in range(frames):
            board=np.full((1080,1920,3),247,np.uint8)
            cv2.putText(board,f'ACT-{condition} | DEV35 development | {view} | physical t={frame/30:.2f}s',(18,29),cv2.FONT_HERSHEY_SIMPLEX,.7,(25,25,25),1,cv2.LINE_AA)
            for i,(sid,a,ix) in enumerate(zip(source_ids,traces,indices)):
                row=keyed.get(sid);tile=np.full((200,270,3),226,np.uint8)
                if a is not None:
                    j=int(ix[min(frame,len(ix)-1)]);renderer.measured_state(a,j)
                    tile=cv2.resize(renderer.view(view),(270,200),interpolation=cv2.INTER_AREA)
                else:cv2.putText(tile,'NOT EXECUTED',(40,100),cv2.FONT_HERSHEY_SIMPLEX,.5,(45,45,45),1,cv2.LINE_AA)
                cv2.rectangle(tile,(0,0),(270,42),(244,244,244),-1)
                label='SUCCESS' if row and row.get('full_task_success') else str(row.get('first_failure',row.get('terminal','UNKNOWN'))) if row else 'NOT_ATTEMPTED_UPSTREAM'
                cv2.putText(tile,f'{i+1:02d} {sid.replace("GoPark_","")}',(5,15),cv2.FONT_HERSHEY_SIMPLEX,.36,(20,20,20),1,cv2.LINE_AA)
                cv2.putText(tile,label[:35],(5,33),cv2.FONT_HERSHEY_SIMPLEX,.34,(20,20,20),1,cv2.LINE_AA)
                if ix is not None and frame>=len(ix):cv2.putText(tile,'TRACE ENDED',(8,187),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,180),1)
                y,x=divmod(i,7);board[48+y*200:248+y*200,15+x*270:285+x*270]=tile
            cv2.putText(board,'Measured full articulation + dynamic object; visual replay, no physics rerun; all outcomes retained',(18,1072),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,30,30),1)
            process.stdin.write(board.tobytes())
            if frame%300==0:print('ACT_MEASURED_REPLAY',condition,view,frame,frames,flush=True)
        process.stdin.close()
        if process.wait(timeout=60)!=0:raise RuntimeError('Replay encoder failed')
    finally:
        renderer.close()
        if process.poll() is None:process.terminate();process.wait(timeout=10)
    decoded=decoded_video(temporary,frames);temporary.replace(target)
    result=dict(status='ACTUAL_POLICY_MEASURED_REPLAY_VERIFIED',video=record(target),dependencies=expected,decoded=decoded,
                actual_trace_count=actual,nonexecuted_cards=35-actual,frames=frames,physics_rerun=False,
                limitation='Existing visual geometry replay; not a contact or soft-body certificate.')
    atomic_json(receipt,result);return result


def run(out):
    area=out/'replays';area.mkdir(exist_ok=True)
    milestone=read(out/'FIRST_SOURCE_CONDITIONED_FULL_TASK.json')
    if record(milestone['video']['path'])!=milestone['video']:raise ValueError('Changed development replay')
    decoded_video(Path(milestone['video']['path']),milestone['frames'])
    ledger_path=out/'ACT_DEV35/LEDGER.json';rows=read(ledger_path)['rows'] if ledger_path.exists() else []
    source_ids=read(out/'SPLIT_CONTRACT.json')['evaluation_source_ids'];products=[]
    if rows:
        for c in ('A','B'):
            for view in ('top','overview'):products.append(mosaic(out,c,view,rows,source_ids))
    else:
        products=[dict(file=str(area/f'ACT_{c}_DEV35_{view}_35SPLIT.mp4'),status='NOT_RENDERED_NO_ACT_PHYSICAL_TRACES',all_card_video_created=False) for c in ('A','B') for view in ('TOP','OVERVIEW')]
    comparison=area/'ACT_A_B_MATCHED_COMPARISON.mp4'
    if all(p.get('status')=='ACTUAL_POLICY_MEASURED_REPLAY_VERIFIED' for p in products):
        temporary=comparison.with_name(comparison.stem+'.incomplete.mp4')
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(area/'ACT_A_DEV35_OVERVIEW_35SPLIT.mp4'),'-i',str(area/'ACT_B_DEV35_OVERVIEW_35SPLIT.mp4'),
            '-filter_complex','[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack=inputs=2[v]','-map','[v]',
            '-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(temporary)],check=True)
        decoded_video(temporary);temporary.replace(comparison);matched=dict(status='ACTUAL_MATCHED_POLICY_REPLAY_VERIFIED',video=record(comparison))
    else:matched=dict(file=str(comparison),status='NOT_RENDERED_NO_MATCHED_ACT_PHYSICAL_REPLAYS')
    illustrations=[]
    generation=read(out/'TRAIN40_conversion/LEDGER.json')
    # Fixed illustration rule: lowest source ID per condition/terminal category.
    categories={}
    for row in generation['rows']:
        if row.get('physical_run'):
            key=(row['condition'],row['terminal'])
            if key not in categories or row['source_id']<categories[key]['source_id']:categories[key]=row
    for (c,terminal),row in sorted(categories.items()):
        target=area/f'TRAIN40_{c}_{terminal}_ILLUSTRATION.mp4'
        illustrations.append(encode_captured(Path(row['attempt']),target,f'TRAIN illustration {c} {row["source_id"]} | {terminal}'))
    reference_path=out/'reference_coupling10/LEDGER.json'
    reference_clip=dict(status='NOT_RENDERED_NO_PAIRED_REFERENCE_PHYSICAL_TRACES')
    reference_illustrations=[]
    if reference_path.exists():
        rr={(r['source_id'],r['condition']):r for r in read(reference_path)['rows']}
        # Preserve actual one-condition evidence even if the paired condition
        # has no plan. This never becomes a paired physical effect estimate.
        reference_categories={}
        for row in rr.values():
            if not row.get('physical_run'):continue
            key=(row['condition'],row['terminal'])
            if key not in reference_categories or row['source_id']<reference_categories[key]['source_id']:
                reference_categories[key]=row
        for (c,terminal),row in sorted(reference_categories.items()):
            target=area/f'REFERENCE_{c}_{row["source_id"]}_ILLUSTRATION.mp4'
            clip=encode_captured(Path(row['attempt']),target,f'Reference {c} {row["source_id"]} | {terminal}')
            reference_illustrations.append(dict(clip,source_id=row['source_id'],condition=c,terminal=terminal,
                selection_rule='Lowest source ID per condition/terminal category with an actual physical trace',
                paired_physical_effect=False,ACT_ablation=False))
        candidates=sorted(sid for sid in {k[0] for k in rr} if all(rr.get((sid,c),{}).get('physical_run') for c in ('B','B_NO_COUPLING')))
        if candidates:
            sid=candidates[0];clips=[];starts=[]
            for c in ('B','B_NO_COUPLING'):
                row=rr[sid,c];folder=Path(row['attempt']);a=dict(np.load(folder/'event_log.npz'))
                handoff=np.flatnonzero(a['stage']=='RECEIVER_APPROACH')
                starts.append(float(a['control_frame'][handoff[0]])/30 if len(handoff) else 0.)
                target=area/f'REFERENCE_{c}_{sid}_ILLUSTRATION.mp4'
                clips.append(encode_captured(folder,target,f'Reference {c} {sid} | {row["terminal"]}'))
            start=max(0.,min(starts)-1.);target=area/'REFERENCE_COUPLING_PAIRED_HANDOFF.mp4';tmp=target.with_name(target.stem+'.incomplete.mp4')
            subprocess.run(['ffmpeg','-v','error','-y','-ss',str(start),'-i',clips[0]['video']['path'],'-ss',str(start),'-i',clips[1]['video']['path'],
                '-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-t','20','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(tmp)],check=True)
            decoded_video(tmp);tmp.replace(target)
            reference_clip=dict(status='ACTUAL_PAIRED_REFERENCE_ILLUSTRATION',source_id=sid,video=record(target),
                underlying_replays=clips,physical_start_s=start,selection_rule='Lowest source ID with both actual reference traces; common physical-time20s window starting1s before earliest receiver approach, or initial20s when neither reached handoff.',
                ACT_ablation=False,typical_case_estimate=False)
    sanity=[]
    for c in ('A','B'):
        folder=out/'ACT_policy/TRAIN_sanity'/c
        if (folder/'INDEPENDENT_INTERFACE_VERIFICATION.json').exists():
            sanity.append(encode_captured(folder,area/f'ACT_{c}_TRAIN_SANITY.mp4',f'ACT-{c} TRAIN causal-interface sanity | no task-success gate'))
    result=dict(status='ACT_REPLAY_AVAILABILITY_VERIFIED',products=products,matched_comparison=matched,
                development_full_task=milestone['video'],all_required_ACT_replays_verified=matched['status']=='ACTUAL_MATCHED_POLICY_REPLAY_VERIFIED',
                TRAIN_failure_illustrations=illustrations,reference_coupling_clip=reference_clip,policy_sanity=sanity,
                reference_illustrations=reference_illustrations,
                no_all_card_physical_rollout_claim=True,implementation=record(__file__))
    atomic_json(area/'CURRENT_REPLAY_STATUS.json',result);return result
