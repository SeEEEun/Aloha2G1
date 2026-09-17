"""Small diagnostic figures and a real measured common-control handoff replay."""
import json
import subprocess
from pathlib import Path
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text


def first_source_full_task(out,attempt):
    """Record the milestone only from an independently valid complete trace."""
    from .demo_alignment import scored_task_success
    score=read(attempt/'HYBRID_SCORE.json')
    if not scored_task_success(score):raise ValueError('A physically invalid/incomplete task cannot satisfy M2')
    alignment=read(attempt/'DEMONSTRATION_ALIGNMENT.json')
    if not alignment['complete_command_sequence'] or not alignment['physical_validity'] or alignment['task_success'] is not True:
        raise ValueError('Complete measured target-observation alignment is required')
    for item in [score['trace'],alignment['observations'],alignment['commands'],*alignment['images']]:
        if record(item['path'])!=item:raise ValueError('Changed physical evidence')
    command=Path(alignment['commands']['path']);plan=read(command.parent/'PLAN.json')
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    if plan['source_id']!=sid or not plan.get('source_conditioned') or not plan.get('full_task') or plan.get('diagnostic'):
        raise ValueError('Expected the fixed source-conditioned full TRAIN prototype')
    target=out/'replays/FIRST_SOURCE_CONDITIONED_FULL_TASK.mp4';milestone=out/'FIRST_SOURCE_CONDITIONED_FULL_TASK.json'
    if milestone.exists():
        old=read(milestone)
        if old['score']!=record(attempt/'HYBRID_SCORE.json') or old['video']!=record(target):raise ValueError('Immutable first milestone differs')
        return old
    caption=target.with_suffix('.txt');atomic_text(caption,'INTERACTION_OURS | fixed source-conditioned TRAIN prototype\nActual dynamic G1/PhysX observations | natural start to bin settle\nReference converter demonstration; no ACT or real hardware\n')
    temporary=target.with_suffix('.incomplete.mp4')
    args=['ffmpeg','-v','error','-y','-framerate','30','-i',str(attempt/'observations/frame_%06d.png'),
          '-vf',f'drawbox=x=0:y=0:w=iw:h=73:color=white@0.88:t=fill,drawtext=textfile={caption}:x=10:y=6:fontsize=14:fontcolor=black:line_spacing=3',
          '-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(temporary)]
    subprocess.run(args,check=True);subprocess.run(['ffmpeg','-v','error','-i',str(temporary),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(temporary)]))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    if int(stream['nb_frames'])!=alignment['frames'] or stream['r_frame_rate']!='30/1':raise ValueError('Replay time/frame count mismatch')
    temporary.replace(target)
    value=dict(status='SOURCE_CONDITIONED_FULL_TASK_DEMONSTRATED',source_id=sid,
        score=record(attempt/'HYBRID_SCORE.json'),trace=score['trace'],video=record(target),
        alignment=record(attempt/'DEMONSTRATION_ALIGNMENT.json'),plan=record(command.parent/'PLAN.json'),
        source_phase=record(out/'source_phase'/sid/'PHASE_RECORD.json'),connection=plan['connection'],
        frames=alignment['frames'],fps=30,decode_verified=True,encoder_command=args,
        ACT_policy_used=False,privileged_simulation_state_used=True,hardware_used=False,
        selection_rule='First independently physically valid full task on the fixed TRAIN prototype after versioned general repairs; preceding failed attempts retained.',
        measured_fields=['MEASURED_Q','all_measured_q_rad','object_position_world_m','object_quaternion_xyzw','bilateral digit/palm/support contact forces'],
        interpretation='Converter-generation milestone only. Fixed pilot, matched data, both ACT trainings and true ACT35/35 evaluation remain required.')
    atomic_json(milestone,value)
    return value


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read(out/'NUMERIC_SUMMARY.json');folder=out/'figures';folder.mkdir(exist_ok=True)
    source=summary['prototype_source']
    fig,ax=plt.subplots(figsize=(12,3.5));ax.set_axis_off()
    labels=['Existing ALOHA\nFK + image registration\n+ inferred events',
            'Phase/contact goals\nsource relations\n+ shared-pose coupling',
            'Common SE(3) fitting\noverlap + collision checks\nINVALID CANDIDATES',
            'Source-conditioned\nPhysX full task\nNOT DEMONSTRATED']
    for i,label in enumerate(labels):
        ax.text(.125+.25*i,.63,label,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=.7',facecolor='#e9f0f4' if i<2 else '#ffe9df',edgecolor='#52616b'))
        if i<3:ax.annotate('',xy=(.245+.25*i,.63),xytext=(.215+.25*i,.63),arrowprops=dict(arrowstyle='->',color='#52616b'))
    ax.text(.5,.19,'Separate calibration branch: measured dynamic PhysX common control; never counted as Ours.',ha='center',fontsize=11)
    ax.set_title('Hybrid retargeting prototype — implemented stages and unresolved execution gate',fontsize=13)
    fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(folder/f'Fig1_Hybrid_Retargeting_Method.{suffix}',dpi=180)
    plt.close(fig)
    rows=summary['coupling_diagnostics'];fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for enabled,color,label,shift in [('True','#127d8a','Coupling on',-.17),('False','#c47435','Coupling off',.17)]:
        rr=sorted((r for r in rows if r['enable_coupling']==enabled),key=lambda r:r['seed'])
        for ax,key in zip(axs,['overlap_position_disagreement_mm','overlap_orientation_disagreement_rad','modeled_contact_records']):
            ax.bar(np.arange(3)+shift,[r[key] for r in rr],.32,color=color,label=label,hatch='//')
            ax.set_xticks(range(3),['Seed 0','Seed 1','Seed 2']);ax.spines[['top','right']].set_visible(False)
    for ax,title in zip(axs,['Predicted object disagreement (mm)','Predicted rotation disagreement (rad)','Modeled overlap contact records']):ax.set_title(title,fontsize=10)
    axs[0].legend(fontsize=8)
    fig.suptitle('Coupling diagnostic: one TRAIN source, all candidates inadmissible — no physical ablation',fontsize=12)
    fig.text(.5,.015,source+' | 21 interpolated overlap states/candidate | seeds are not independent episodes',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    for suffix in ('png','svg'):fig.savefig(folder/f'Fig3_Coupling_Ablation.{suffix}',dpi=180)
    plt.close(fig)
    atomic_text(folder/'Fig2_Wrist_vs_Ours_Physical_Results.NOT_GENERATED.md',
        'No frozen DEV35 physical observations exist. A physical-results figure and paired matrix are intentionally not fabricated. Use the not-attempted accounting tables only; report package remains PARTIAL.\n')
    prototype=read(out/'prototype/RESULT.json');fig,ax=plt.subplots(figsize=(10,3.5))
    ours=next(r for r in prototype['results'] if r['method']=='INTERACTION_OURS');ik=read(ours['phase_ik']['path'])
    errors=[max(e['position_m'] for e in r['selected_errors'].values())*1000 for r in ik['phases']]
    ax.bar(range(len(errors)),errors,color=['#127d8a' if r['goal_satisfied'] else '#b85943' for r in ik['phases']])
    ax.set_xticks(range(len(errors)),[r['phase'] for r in ik['phases']],rotation=25,ha='right',fontsize=8)
    ax.set_ylabel('Maximum active-hand position error (mm)')
    ax.set_title('Initial source-conditioned phase solve — kinematics only; orientation checked separately')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(folder/f'PROTOTYPE_PHASE_FEASIBILITY.{suffix}',dpi=180)
    plt.close(fig)


def replay(out,resume):
    from tools import render_final_episode_registered_physical_evidence as render
    import cv2
    folder=out/'common_control/scripted_captured';destination=out/'replays';destination.mkdir(exist_ok=True)
    target=destination/'COMMON_CONTROL_MEASURED_HANDOFF.mp4';provenance=destination/'COMMON_CONTROL_REPLAY.json'
    trace_record=record(folder/'event_log.npz')
    if resume and target.exists() and provenance.exists():
        old=read(provenance)
        if old['trace']==trace_record and old['video']==record(target):return old
    control=read(out/'common_control/RESULT.json');score=next(r for r in control['results'] if r['control']=='scripted')
    z=dict(np.load(folder/'event_log.npz',allow_pickle=False));frames=z['control_frame']
    # Fixed event-based illustration rule, independent of success selection.
    event=int(score['giving_hand_release_frame']);start=max(0,event-90);end=min(int(frames[-1]),event+120)
    indices=[np.flatnonzero(frames==frame)[-1] for frame in range(start,end+1)]
    renderer=render.PhysicalRenderer()
    common=read(ROOT/'outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4/config/common_config.json')
    assert record(renderer.g1.path)['sha256']==common['models']['g1_xml_sha256']
    temporary=target.with_name(target.stem+'.incomplete.mp4')
    args=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1080x480','-r','30','-i','-',
          '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(temporary)]
    process=subprocess.Popen(args,stdin=subprocess.PIPE)
    try:
        for count,index in enumerate(indices):
            renderer.set_state(z['measured_q_rad'][index],z['object_position_world_m'][index],z['object_quaternion_xyzw'][index])
            board=np.full((480,1080,3),246,np.uint8)
            for col,view in enumerate(('top','overview')):board[50:455,col*540:(col+1)*540]=renderer.view(view)
            cv2.putText(board,'COMMON SCRIPTED CONTROL | saved measured states | NOT AN OURS TASK',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.62,(25,25,25),1,cv2.LINE_AA)
            cv2.putText(board,f'Physical t = {frames[index]/30:.2f} s | {z["stage"][index]}',(12,475),cv2.FONT_HERSHEY_SIMPLEX,.46,(25,25,25),1,cv2.LINE_AA)
            process.stdin.write(board.tobytes())
            if count==min(90,len(indices)-1):render.atomic_image(out/'figures/COMMON_CONTROL_MEASURED_HANDOFF.png',board)
        process.stdin.close();returncode=process.wait(timeout=60)
        if returncode:raise RuntimeError('ffmpeg failed')
    finally:
        renderer.close()
        if process.poll() is None:process.terminate();process.wait(timeout=10)
    temporary.replace(target)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)],text=True))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert stream['width']==1080 and stream['height']==480 and int(stream['nb_frames'])==len(indices)
    assert abs(float(probe['format']['duration'])-len(indices)/30)<.04
    result=dict(kind='ACTUAL_MEASURED_COMMON_CONTROL_REPLAY',method_result=False,source_conditioned=False,
        trace=trace_record,video=record(target),frame_range=[start,end],frames=len(indices),fps=30,
        duration_s=float(probe['format']['duration']),resolution=[1080,480],camera_order=['top','overview'],
        selection_rule='3 seconds before to 4 seconds after measured giver-release frame',
        renderer=record(Path(render.__file__)),model=record(renderer.g1.path),
        measured_fields=['measured_q_rad','object_position_world_m','object_quaternion_xyzw'],
        render_model_note='existing measured-state mesh renderer; collider display is not a runtime penetration certificate')
    atomic_json(provenance,result);return result


def run(out,resume=False):
    figures(out);video=replay(out,resume)
    unavailable=['WRIST_DEV35_TOP_35SPLIT.mp4','WRIST_DEV35_OVERVIEW_35SPLIT.mp4','OURS_DEV35_TOP_35SPLIT.mp4','OURS_DEV35_OVERVIEW_35SPLIT.mp4']
    atomic_json(out/'replays/REPLAY_STATUS.json',dict(common_control=video,
        dev35_mosaics=[dict(file=n,status='NOT_RENDERED_NO_METHOD_PHYSICAL_ROLLOUTS') for n in unavailable],
        ablation_clip='NOT_RENDERED_NO_PAIRED_PHYSICAL_ROLLOUTS',all_card_videos_created=False))
    atomic_text(out/'PAPER_FIGURE_DIRECTION.md',f'''# 그림 및 원고 인계 — 프로토타입 범위

- Fig. 1, 방법 절: `{out}/figures/Fig1_Hybrid_Retargeting_Method.png` 및 `.svg`. 왼쪽부터 기존 추출, 단계/접촉 목표, 공통 SE(3)·충돌 검사, 미입증된 실행 단계 순서이다. 빨간 미완료 표시를 삭제하지 않는다. 보정 물리 실행은 별도 가지이다.
  Caption: “Prototype hybrid retargeting from existing source FK, registration and inferred events to phase/contact goals and common realization checks. The source-conditioned full task was not demonstrated. Actual dynamic execution shown separately is common scripted calibration with privileged simulated-state access, not an Ours result; no real-robot or VLA claim is made.”

- Fig. 2, DEV35 결과 절: 생성하지 않음. `{out}/TABLE_MAIN_WRIST_VS_OURS.md` 및 `figures/Fig2_Wrist_vs_Ours_Physical_Results.NOT_GENERATED.md`를 사용한다. 35+35의 예정 수와 실제 실행 0을 구분하고 성공률 그래프를 만들지 않는다.
  Caption if reporting accounting: “DEV35 development-evaluation accounting: 35 intended instances per primary method, zero physical rollouts, all upstream-not-attempted. Physical completion is not measured; these counts do not estimate 0% task success.”

- Fig. 3, 제한점/개발 진단 절: `{out}/figures/Fig3_Coupling_Ablation.png` 및 `.svg`. 위치 불일치, 회전 불일치, 모델 접촉 기록 순서이다. 한 TRAIN 녹화의 세 초기값이며 대응 10개 실험 결과가 아니다. 모든 후보가 무효라는 표시를 유지한다.
  Caption: “Diagnostic coupling-on/off fit on one predeclared TRAIN source with identical non-coupling factors and three fixed seeds per condition. Bars show predicted shared-object disagreement over 21 interpolated overlap states and modeled contact records; all candidates were inadmissible and none was physically executed. Seeds are not independent episodes. The planned paired-10 DEV35 physical ablation has observed N=0.”

- 보충 자료: `{out}/figures/PROTOTYPE_PHASE_FEASIBILITY.png` 및 `.svg`는 첫 단계 해의 진단이다. `{out}/figures/COMMON_CONTROL_MEASURED_HANDOFF.png`와 `{out}/replays/COMMON_CONTROL_MEASURED_HANDOFF.mp4`는 실제 측정 상태를 재생한 보정 예시이다. 위에서 본 영상, 전체 시점 순서이며 실제 30 fps 시간을 사용한다. Ours 성공 예시로 넣지 않는다.
  Caption: “Common scripted calibration replay from measured G1 joint states and dynamic PhysX object poses, viewed from top and overview cameras. Illustration window is fixed around the measured giver-release event. This replay is not source-conditioned method evaluation; privileged simulated states are used, and no real-robot/VLA result is claimed.”

4개 DEV35 모자이크와 물리 대응 전달 클립은 생성하지 않았다. 실제 방법 롤아웃이 없으므로 카드 영상이나 가상 성공 동작으로 대체하지 않는다. 보고서 및 수치 표는 이 제한과 함께 인계한다.
''')
    return dict(status='PROTOTYPE_DIAGNOSTIC_MEDIA_VERIFIED',common_control_video=video['video'],physical_study_videos='NOT_AVAILABLE')
