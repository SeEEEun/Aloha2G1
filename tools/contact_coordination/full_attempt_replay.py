"""Solid measured-state replay whose horizon never depends on task failure."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import subprocess
import json
import numpy as np
from .io import read, record, atomic_json
from .physical_failure_evidence import recorded_failures

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
DIAGNOSTIC_LABEL='진단용 전체 실행 | 공식 결과 아님'
RECONSTRUCTED_LABEL='재구성된 진단 명령 | 원래 계획 아님 | 공식 평가 제외'


def replay_indices(trace):
    frames=np.asarray(trace['control_frame'])
    if not len(frames):raise ValueError('FULL_DIAGNOSTIC_UNAVAILABLE: no measured states')
    return np.r_[np.flatnonzero(np.diff(frames)!=0),len(frames)-1]


def render(folder,target,resume=False):
    import mujoco,cv2
    from PIL import Image,ImageDraw,ImageFont
    from fontTools.ttLib import TTFont
    from .paper_replays import renderer_class,decoded_video
    from tools.render_final_episode_registered_physical_evidence import camera
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    trace_path=folder/'event_log.npz';rec=read(folder/'FULL_ATTEMPT_RECORDING.json')
    contract=dict(trace=record(trace_path),recording=record(folder/'FULL_ATTEMPT_RECORDING.json'),
                  failure_evidence_code=record(Path(__file__).with_name('physical_failure_evidence.py')),
                  legacy_validity=record(folder/'LEGACY_CONTROL_SCORER.json') if (folder/'LEGACY_CONTROL_SCORER.json').exists() else None,
                  renderer=record(__file__),shared_renderer=record(Path(__file__).parent/'paper_replays.py'),font=record(FONT),
                  geometry_audit=record(folder/'MEASURED_GEOMETRY_AUDIT.json') if (folder/'MEASURED_GEOMETRY_AUDIT.json').exists() else None,
                  finite_validity=record(folder/'FINITE_STATE_VALIDITY_TELEMETRY.json') if (folder/'FINITE_STATE_VALIDITY_TELEMETRY.json').exists() else None,
                  final_score=record(folder/'ABC_NOMINAL_SCORE.json') if (folder/'ABC_NOMINAL_SCORE.json').exists() else None)
    metadata_path=folder/'SOURCE_GUIDED_RENDER_METADATA.json'
    metadata=read(metadata_path) if metadata_path.exists() else None
    contract['source_guided_metadata']=record(metadata_path) if metadata else None
    contract['protected_object_audit']=record(folder/'PREGRASP_OBJECT_PROTECTION.json') if (folder/'PREGRASP_OBJECT_PROTECTION.json').exists() else None
    receipt=target.with_suffix('.json')
    if resume and target.exists() and receipt.exists():
        prior=read(receipt)
        if prior['dependencies']==contract and prior['video']==record(target):return prior
        raise ValueError('Changed replay inputs; preserve previous output')
    glyphs=TTFont(FONT,fontNumber=1).getBestCmap()
    assert all(ord(c) in glyphs for c in DIAGNOSTIC_LABEL+RECONSTRUCTED_LABEL+'기록 종료' if not c.isspace())
    font=ImageFont.truetype(FONT,21,index=1);small=ImageFont.truetype(FONT,18,index=1)
    plan=read(folder/'input/PLAN.json')
    selection=plan.get('selected_candidate_id','See saved selected phase candidates')
    if plan.get('contact_selection'):
        selected=read(plan['contact_selection']['path'])
        selection=' + '.join(v.split(':HANDOFF:')[-1] for v in selected.get('selected_candidate_ids',[str(selected.get('candidate_id',selection))]))
    a=dict(np.load(trace_path,allow_pickle=False));ix=replay_indices(a)
    renderer=renderer_class()();renderer.renderer.close()
    renderer.model.vis.global_.offwidth=max(960,renderer.model.vis.global_.offwidth)
    renderer.model.vis.global_.offheight=max(520,renderer.model.vis.global_.offheight)
    renderer.renderer=mujoco.Renderer(renderer.model,width=960,height=520)
    renderer.model.geom_rgba[:,3]=1.
    renderer.cameras['overview']=camera([1.1,.9,1.7],[.42,.1,1.0])
    temp=target.with_name(target.stem+'.incomplete.mp4')
    command=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x810','-r','30','-i','-',
             '-c:v','libx264','-threads','2','-preset','veryfast','-crf','19','-pix_fmt','yuv420p',str(temp)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE);screen=target.parent/'screenshots';screen.mkdir(exist_ok=True)
    screenshots=[];score_path=folder/'ABC_NOMINAL_SCORE.json'
    outcome=read(score_path) if score_path.exists() else None
    failure,task_failure=recorded_failures(folder,rec,outcome,len(ix),a)
    try:
        for frame,j in enumerate(ix):
            renderer.measured_state(a,int(j));rgb=cv2.cvtColor(renderer.view('overview'),cv2.COLOR_BGR2RGB)
            board=Image.new('RGB',(960,810),(246,246,246));board.paste(Image.fromarray(rgb),(0,90));draw=ImageDraw.Draw(board)
            channel=rec['evidence_channel']
            title=RECONSTRUCTED_LABEL if channel=='RECONSTRUCTED_DIAGNOSTIC' else DIAGNOSTIC_LABEL if channel!='OFFICIAL_NOMINAL' else 'OFFICIAL NOMINAL'
            if rec.get('architecture_repair_verification_only') and channel=='OFFICIAL_NOMINAL':
                title='ARCHITECTURE VERIFICATION | ACTUAL PHYSX'
            if metadata and channel=='OFFICIAL_NOMINAL':title='SOURCE-GUIDED RRT VERIFICATION | ACTUAL PHYSX'
            phase_selection=(metadata or {}).get('phase_selection',{}).get(str(a['NOMINAL_PHASE'][j]),{})
            if rec.get('synthetic_contract_test'):title += ' | 기록 검증 전용 · 점수/학습 제외'
            for text,y,f in [(title,5,font),(rec['method_key']+' | '+rec['source_id'],34,font),
                (f"physical time {a['timestamp_s'][j]:.6f} s | frame {frame+1}/{len(ix)}",63,small),
                ('CURRENT: '+str(a['FULL_ATTEMPT_RUNTIME_PHASE'][j]),617,font),
                ('TASK FAILURE: '+task_failure['reason'] if task_failure and frame>=task_failure['control_frame'] else
                    'NOMINAL INTENT: '+str(a['NOMINAL_PHASE'][j]),647,small),
                ('FIRST FAILURE: '+(failure['reason'] if failure and frame>=failure['control_frame'] else 'NONE')+(' | 기록 종료' if frame==len(ix)-1 else ''),675,small),
                ('TRACE TYPE: MEASURED PHYSX | SOLID ROBOT',704,small),
                ('CANDIDATE: '+str(phase_selection.get('candidate',selection))+' | PATH: '+str(phase_selection.get('path','CONTACT HOLD / SEE PLAN')) if metadata else 'SELECTED HANDOFF: '+str(selection),733,small),
                ('PLANNER: COMPLETE VALIDATED PATH | TASK: '+('FAILURE RECORDED' if failure and frame>=failure['control_frame'] else
                    'SUCCESS' if frame==len(ix)-1 and outcome and outcome['stages']['FULL_TASK'] else 'IN PROGRESS'),762,small)]:
                box=draw.textbbox((0,0),text,font=f)
                if box[2]>=950:f=ImageFont.truetype(FONT,max(11,int(f.size*940/box[2])),index=1)
                draw.text((8,y),text,font=f,fill=(25,25,25))
            proc.stdin.write(np.asarray(board).tobytes())
            if frame in (0,len(ix)//2,len(ix)-1):
                path=screen/f'{target.stem}_{frame:04d}.png';board.save(path);screenshots.append(record(path))
        proc.stdin.close();assert proc.wait(timeout=60)==0
    finally:
        renderer.close()
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    probe=decoded_video(temp,len(ix));temp.replace(target)
    value=dict(status='MEASURED_FULL_RECORDING_RENDERED',video=record(target),dependencies=contract,
        frames=len(ix),last_rendered_timestamp_s=float(a['timestamp_s'][ix[-1]]),
        true_trace_last_timestamp_s=float(a['timestamp_s'][-1]),probe=probe,screenshots=screenshots,
        failure_timestamp_used_for_cutoff=False,solid_robot=True,object_motion='MEASURED_DYNAMIC_STATE_ONLY',
        Korean_font_glyphs_verified=True,visual_inspection='PENDING',physics_rerun=False)
    assert value['last_rendered_timestamp_s']==value['true_trace_last_timestamp_s']
    atomic_json(receipt,value);return value


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    controls=read(a.run_dir/'VIDEO_CONTRACT_CONTROL.json')['controls'];results=[]
    for c in controls:
        folder=Path(c['trace']['path']).parent
        results.append(render(folder,a.run_dir/'video_contract_replays'/(c['name']+'.mp4'),a.resume))
    atomic_json(a.run_dir/'VIDEO_CONTRACT_REPLAYS.json',dict(status='AUTOMATED_PASS_VISUAL_PENDING',videos=results))
