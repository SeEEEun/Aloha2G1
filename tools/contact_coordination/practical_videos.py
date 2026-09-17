"""All80 final identities, full measured motion, honest no-plan/end cards."""
from pathlib import Path
import os,subprocess
import numpy as np
from .io import read,record,atomic_json,atomic_text
from .architecture_videos import probe,writer,card,preserve_previous,inspect_frames,composite
from .source_guided_videos import comparison,concatenate
from .practical_planning import csv_file

SIZE=(960,870)
TITLES={'A':'PAPER_A | Wrist-Centric Retargeting','B':'PAPER_B | Interaction-Centric Retargeting (Ours)'}


def render_case(row,target):
    """Same camera/model as repaired measured-state replay; no extra physics."""
    os.environ.setdefault('MUJOCO_GL','egl')
    import cv2,mujoco
    from PIL import Image,ImageDraw,ImageFont
    from .paper_replays import renderer_class
    from .full_attempt_replay import replay_indices
    from .physical_failure_evidence import recorded_failures
    from tools.render_final_episode_registered_physical_evidence import camera
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);p=row['physical'];code=row['paper_method'][-1]
    dependencies=dict(result=row,renderer=record(__file__),common_renderer=record(Path(__file__).parent/'paper_replays.py'),
        failure_evidence=record(Path(__file__).parent/'physical_failure_evidence.py'))
    receipt=target.with_suffix('.json')
    if target.exists() and receipt.exists():
        old=read(receipt)
        if old['dependencies']==dependencies and old['video']==record(target):return old
        preserve_previous(target)
    temp=target.with_name(target.stem+'.incomplete.mp4');proc=writer(temp,SIZE)
    if not p['physics_executed']:
        failure=p.get('first_failure');reason=(failure.get('cause','UNKNOWN')+' @ '+failure.get('phase','UNKNOWN')) if isinstance(failure,dict) else str(failure)
        board=card(SIZE,[TITLES[code],row['source_id'],'NO_COMPLETE_PLAN','NO_COMPLETE_COMMAND',
            'CURRENT PHASE: '+(failure.get('phase','NOT REACHED') if isinstance(failure,dict) else 'NOT REACHED'),
            'FIRST FAILURE: '+reason,'TASK STATUS: NOT EXECUTED','PLAN STATUS: NO COMPLETE PLAN',
            'PHYSICAL TIME: N/A','FINAL RESULT: PLANNING FAILURE','TRACE TYPE: NONE','No official physical execution exists'])
        for _ in range(90):proc.stdin.write(board.tobytes())
        proc.stdin.close();assert proc.wait(timeout=60)==0;temp.replace(target)
        value=dict(status='NO_PLAN_STATUS_CARD',video=record(target),dependencies=dependencies,frames=90,
            source_id=row['source_id'],paper_method=row['paper_method'],measured_physics=False,true_command_horizon=None,
            diagnostic_reconstruction=False,probe=probe(target))
        atomic_json(receipt,value);return value
    folder=Path(p['folder']);trace=dict(np.load(folder/'event_log.npz',allow_pickle=False));rec=read(folder/'FULL_ATTEMPT_RECORDING.json')
    indices=replay_indices(trace);score=read(folder/'ABC_NOMINAL_SCORE.json') if (folder/'ABC_NOMINAL_SCORE.json').exists() else None
    first,task=recorded_failures(folder,rec,score,len(indices),trace)
    if p.get('status')=='NUMERICAL_ABORT' and first is None:
        abort=read(folder/'NUMERICAL_ABORT.json')
        first=dict(reason='NUMERICAL ABORT: '+str(abort['reason']),
            control_frame=min(int(abort['control_frame']),len(indices)-1))
    if p.get('status')!='NUMERICAL_ABORT':assert len(indices)==rec['requested_recording_frames']
    final='FULL TASK SUCCESS' if p.get('stages',{}).get('FULL_TASK') else 'NUMERICAL ABORT' if p.get('status')=='NUMERICAL_ABORT' else 'PHYSICAL TASK FAILURE'
    renderer=renderer_class()();renderer.renderer.close()
    renderer.model.vis.global_.offwidth=max(960,renderer.model.vis.global_.offwidth)
    renderer.model.vis.global_.offheight=max(520,renderer.model.vis.global_.offheight)
    renderer.renderer=mujoco.Renderer(renderer.model,width=960,height=520);renderer.model.geom_rgba[:,3]=1.
    renderer.cameras['overview']=camera([1.1,.9,1.7],[.42,.1,1.0])
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(font_path,20)
    try:
        for frame,j in enumerate(indices):
            renderer.measured_state(trace,int(j));rgb=cv2.cvtColor(renderer.view('overview'),cv2.COLOR_BGR2RGB)
            board=Image.new('RGB',SIZE,(246,246,246));board.paste(Image.fromarray(rgb),(0,92));draw=ImageDraw.Draw(board)
            failed=first and frame>=first['control_frame']
            phase=str(trace['NOMINAL_PHASE'][j]);runtime=str(trace['FULL_ATTEMPT_RUNTIME_PHASE'][j])
            task_status='FAILURE RECORDED; COMMANDS CONTINUE' if failed else 'SUCCESS' if frame==len(indices)-1 and final=='FULL TASK SUCCESS' else 'IN PROGRESS'
            if frame==len(indices)-1 and failed:task_status='GENUINE PHYSICS ABORT' if p.get('status')=='NUMERICAL_ABORT' else 'FAILURE RECORDED; COMMAND HORIZON ENDED'
            lines=[(TITLES[code],5),(row['source_id']+' | '+row['internal_method'],34),
                (f'PHYSICAL TIME: {trace["timestamp_s"][j]:.6f} s | frame {frame+1}/{len(indices)}',63),
                ('CURRENT PHASE: '+runtime,620),('NOMINAL PHASE: '+phase,650),
                ('FIRST FAILURE: '+(first['reason'] if failed else 'NONE'),680),
                ('TASK STATUS: '+task_status,710),('PLAN STATUS: COMPLETE VALIDATED PLAN',740),
                ('FINAL RESULT: '+final,770),('TRACE TYPE: MEASURED PHYSX | SOLID ROBOT',800),
                ('TRUE RECORDED END' if frame==len(indices)-1 else 'Existing command sequence continues after task failure',830)]
            for text,y in lines:
                f=font;width=draw.textlength(text,font=f)
                if width>942:f=ImageFont.truetype(font_path,max(11,int(font.size*942/width)))
                draw.text((8,y),text,font=f,fill=(24,24,24))
            proc.stdin.write(np.asarray(board).tobytes())
        proc.stdin.close();assert proc.wait(timeout=60)==0
    finally:
        renderer.close()
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    temp.replace(target);info=probe(target);assert info['frames']==len(indices)
    assert float(trace['timestamp_s'][indices[-1]])==float(trace['timestamp_s'][-1])
    value=dict(status='MEASURED_FULL_RECORDING_RENDERED',video=record(target),dependencies=dependencies,frames=len(indices),probe=info,
        source_id=row['source_id'],paper_method=row['paper_method'],measured_physics=True,solid_robot=True,
        measured_object=True,first_failure_used_as_cutoff=False,true_command_horizon=rec['requested_recording_frames'],
        genuine_abort=p.get('status')=='NUMERICAL_ABORT',last_rendered_time_s=float(trace['timestamp_s'][indices[-1]]),
        actual_trace_end_time_s=float(trace['timestamp_s'][-1]),diagnostic_reconstruction=False,
        trace=record(folder/'event_log.npz'),recording=record(folder/'FULL_ATTEMPT_RECORDING.json'))
    atomic_json(receipt,value);return value


def run(out):
    from .practical_finalize import verify_freeze
    verify_freeze(out);ids=read(out/'PRACTICAL_STUDY.json')['train_source_ids'];videos=out/'videos';all_rows=[];lookup={}
    for code in ('A','B'):
        rows=read(out/('PAPER_'+code+'_RESULTS.json'))['rows'];assert [r['source_id'] for r in rows]==ids
        all_rows+=rows
        for row in rows:lookup[(row['source_id'],code)]=row
    outputs=[];individual=[];manifest_rows=[];paths={};renders=[]
    for index,row in enumerate(all_rows):
        code=row['paper_method'][-1];sid=row['source_id'];target=videos/('PAPER_'+code)/(sid+'_FULL.mp4')
        atomic_json(out/'WORK_PROGRESS.json',dict(stage='videos',source_id=sid,method=row['internal_method'],
            substage='Full measured render or honest no-plan status card',completed_work=index,remaining_work=80-index))
        result=render_case(row,target);renders.append(result);outputs.append(target);individual.append(target)
        # Give no-plan tiles persistent status cards throughout overview timing.
        paths[(sid,code)]=target if result['measured_physics'] else None
        info=inspect_frames(target,target.parent)
        manifest_rows.append(dict(source_id=sid,paper_method=row['paper_method'],internal_method=row['internal_method'],video_path=str(target),
            trace_type='MEASURED_PHYSX' if result['measured_physics'] else 'NO_COMPLETE_COMMAND_STATUS_CARD',frames=result['frames'],
            expected_command_frames=result['true_command_horizon'],last_rendered_physical_time_s=result.get('last_rendered_time_s'),
            true_trace_end_time_s=result.get('actual_trace_end_time_s'),genuine_abort=result.get('genuine_abort',False),
            full_horizon=result.get('last_rendered_time_s')==result.get('actual_trace_end_time_s') if result['measured_physics'] else None,
            solid_robot=result.get('solid_robot'),ffprobe_pass=True,sha256=result['video']['sha256']))
        atomic_json(out/'TRAIN40_VIDEO_PROGRESS.json',dict(individual_count=len(individual),last=info,renders=renders))
    assert len(individual)==80
    for code in ('A','B'):
        actual=list((videos/('PAPER_'+code)).glob('*_FULL.mp4'))
        assert {p.name[:-len('_FULL.mp4')] for p in actual}==set(ids) and len(actual)==40
    def labels(sid,code):
        row=lookup[(sid,code)];p=row['physical'];failure=p.get('first_failure')
        return [TITLES[code],sid,'FIRST FAILURE: '+str(failure),'FINAL: '+('SUCCESS' if p.get('stages',{}).get('FULL_TASK') else 'FAIL / NOT EXECUTED')]
    pairs={}
    for index,sid in enumerate(ids):
        target=videos/'pairs'/(sid+'_PAPER_AB.mp4');target.parent.mkdir(exist_ok=True)
        pairs[sid]=comparison([paths[(sid,c)] for c in ('A','B')],[labels(sid,c) for c in ('A','B')],target,tile=SIZE);outputs.append(target)
        atomic_json(out/'WORK_PROGRESS.json',dict(stage='videos',source_id=sid,method='PAPER_AB',substage='Matched full-horizon pair',completed_work=index+1,remaining_work=40-index-1))
    for begin,end,label in [(0,20,'01_20'),(20,40,'21_40')]:
        subset=ids[begin:end]
        for code in ('A','B'):
            target=videos/f'PAPER_{code}_TRAIN40_{label}_FULL.mp4'
            outputs.append(composite([paths[(sid,code)] for sid in subset],[labels(sid,code) for sid in subset],target,5,4,tile=(640,580)))
            alias=videos/f'PAPER_{code}_TRAIN40_{label}.mp4'
            if alias.exists() and record(alias)['sha256']!=record(target)['sha256']:preserve_previous(alias)
            if not alias.exists():os.link(target,alias)
            outputs.append(alias)
        outputs.append(concatenate([pairs[sid] for sid in subset],videos/f'PAPER_AB_TRAIN40_{label}_SIDE_BY_SIDE.mp4'))
    inspected=[]
    for p in outputs:inspected.append(inspect_frames(p,p.parent))
    # Side-by-side boundaries are also recorded so all identities can be
    # reviewed without confusing global montage timestamps with physical time.
    manifest=out/'TRAIN40_AB_VIDEO_MANIFEST.csv';csv_file(manifest,manifest_rows)
    report=out/'TRAIN40_AB_VIDEO_VERIFICATION.json';atomic_json(report,dict(status='AUTOMATED_PASS_VISUAL_PENDING',
        expected_A_sources=ids,expected_B_sources=ids,A_individual_count=40,B_individual_count=40,
        no_missing_or_duplicate_sources=True,individual_render_receipts=[record(p.with_suffix('.json')) for p in individual],
        videos=inspected,no_diagnostic_physics_results=True,all_actual_horizons_verified=True,
        source_camera_shared=True,overview_layout='5 columns x4 rows;640x580 per tile',
        no_plan_policy='Explicit status cards with no physical time; not fabricated motion',
        comparison_policy='Full measured playback at30Hz; shorter trace switches to explicit end card'))
    markdown=out/'TRAIN40_AB_VIDEO_VERIFICATION.md';atomic_text(markdown,'# TRAIN40 paper A/B full-motion video verification\n\n'
        'Exactly40 PAPER_A=A_WRIST and40 PAPER_B=C_COUPLED individual source videos are present with no duplicate or missing identities. '
        'Actual PhysX states are replayed with a solid robot and measured object through the true stored horizon, including post-failure commands. '
        'No-plan cases are explicitly labelled NO_COMPLETE_PLAN / NO_COMPLETE_COMMAND status cards, not physical conversions. '
        'Four5x4 overview videos and two sequential matched20-source comparison videos preserve fixed source order, camera and30Hz timing. '
        'Shorter recordings use explicit end cards. All MP4s have ffprobe and beginning/middle/end decoding evidence in the JSON manifest. '
        'Final completion separately requires the agent visual inspection receipt matching every current video hash.\n')
    return [manifest,report,markdown,*outputs,*[p.with_suffix('.json') for p in individual]]
