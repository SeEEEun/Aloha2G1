"""Numeric prototype figures, then existing measured-state renderer; no physics."""
import json,subprocess
from pathlib import Path
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=read(out/'NUMERIC_SUMMARY.json');folder=out/'figures';folder.mkdir(exist_ok=True)
    def save(fig,name):
        for suffix in ('png','svg'):fig.savefig(folder/(name+'.'+suffix),dpi=170,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,3.8));ax.set_axis_off()
    labels=['Existing source evidence\nFK / registered scene\nevents + uncertainty','Phase/contact goals\nDex3 contact frames\ncoupled handoff fit','Common IK / connection\nfull hand geometry\nretiming + contact control','Dynamic PhysX evidence\nacquisition / lift / handoff\nFULL TASK NOT DEMONSTRATED']
    for i,label in enumerate(labels):
        ax.text(.12+.255*i,.62,label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.6',facecolor='#e5eef0' if i<3 else '#f6e4dc',edgecolor='#52616b'))
        if i<3:ax.annotate('',xy=(.255+.255*i,.62),xytext=(.222+.255*i,.62),arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.15,'Newly identified common requirement: measured waist/full articulation must enter calibration and replay FK.',ha='center',fontsize=10)
    ax.set_title('Hybrid source-conditioned prototype — implemented path and unresolved calibration',fontsize=13)
    save(fig,'Fig1_Method')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[1,1.3]})
    axes[0].axis('off');axes[0].text(.5,.72,'FULL TASK SUCCESS\nNOT MEASURED',ha='center',va='center',fontsize=20,weight='bold')
    axes[0].text(.5,.31,'Wrist: 0 physical / 35 intended\nOurs: 0 physical / 35 intended\nNo freeze; no paired effect or CI',ha='center',va='center',fontsize=12)
    matrix=np.zeros((35,2));axes[1].imshow(matrix,aspect='auto',cmap=matplotlib.colors.ListedColormap(['#e7e7e7']),vmin=0,vmax=1)
    axes[1].set_xticks([0,1],['Wrist','Ours']);axes[1].set_yticks([0,4,9,14,19,24,29,34],['1','5','10','15','20','25','30','35']);axes[1].set_ylabel('Ordered DEV35 manifest row')
    axes[1].set_title('All 70 primary instances: upstream not attempted',fontsize=10)
    fig.suptitle('DEV35 development accounting — no physical comparison',fontsize=13)
    fig.text(.5,.01,'Grey = NOT_ATTEMPTED_UPSTREAM, not observed failure. Planning and physical completion rates are unmeasured.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));save(fig,'Fig2_AB_Physical_Results')
    fit=Path(read(out/'target_repair/CURRENT_CONTACT_REGION_FIT.json')['path']);on=read(fit/'coupling_True.json');off=read(fit/'coupling_False.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2));x=np.arange(len(on))
    for rows,label,color,shift in [(on,'Coupling on','#197a86',-.18),(off,'Coupling off','#b97142',.18)]:
        for ax,key,scale in zip(axes,['position_disagreement_m','rotation_disagreement_rad'],[1000,1]):ax.bar(x+shift,[r[key]*scale for r in rows],.36,label=label,color=color)
    for ax,title in zip(axes,['Predicted object position disagreement (mm)','Predicted object rotation disagreement (rad)']):
        ax.set_title(title,fontsize=10);ax.set_xticks(x,[f"C{r['contact_candidate']}/S{r['seed']}" for r in on]);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=9);fig.suptitle('One TRAIN source: algebraic coupling diagnostic, physical paired N=0',fontsize=13)
    fig.text(.5,.015,'Same 3-contact / 2-seed banks. Candidate seeds are not independent episodes; inherited contact calibration remains unqualified.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.93));save(fig,'Fig3_Coupling_Ablation')
    sid=n['source_id'];base=out/'prototype'/sid/'full_task_connection/ead64a3c8365';a=np.load(base/'geometry_telemetry_diagnostic/event_log.npz');time=a['timestamp_s'];fig,axes=plt.subplots(3,1,figsize=(11,6.5),sharex=True)
    axes[0].plot(time,a['object_position_world_m'][:,2],color='#197a86');axes[0].axhline(.8375,color='grey',ls=':');axes[0].set_ylabel('Object z (m)');axes[0].set_title('Actual source-conditioned diagnostic: lift, receiver support, then drop')
    for side,color in [('left','#197a86'),('right','#b97142')]:
        support=(a[side+'_thumb_force_n']>=.015)&((a[side+'_index_force_n']>=.015)|(a[side+'_middle_force_n']>=.015))
        axes[1].plot(time,support.astype(float)+(0 if side=='left' else 1.2),label=side,color=color,linewidth=.8)
    axes[1].set_yticks([.5,1.7],['Left','Right']);axes[1].set_ylabel('Opposing contact');axes[1].legend(loc='upper left',fontsize=8)
    j=list(a['all_joint_names']).index('waist_pitch_joint');axes[2].plot(time,a['all_measured_q_rad'][:,j]*180/np.pi,color='#704f88');axes[2].set_ylabel('Waist pitch (deg)');axes[2].set_xlabel('Actual physical time (s)')
    for ax in axes:
        ax.axvline(946/30,color='#b64130',ls='--',linewidth=.8);ax.spines[['top','right']].set_visible(False)
    axes[0].text(946/30,.96,' Giver withdrawal',fontsize=8,color='#b64130',rotation=90)
    fig.tight_layout();save(fig,'PROTOTYPE_MEASURED_STAGES')
    atomic_json(folder/'FIGURE_PROVENANCE.json',dict(summary=record(out/'NUMERIC_SUMMARY.json'),coupling_config=record(fit/'CONFIG.json'),measured_trace=record(base/'geometry_telemetry_diagnostic/event_log.npz'),generator=record(__file__),physical_comparison_measured=False))
    return dict(status='PROTOTYPE_FIGURES_VERIFIED',DEV35_figure='NONEXECUTION_ACCOUNTING',ablation_figure='TRAIN_ALGEBRAIC_DIAGNOSTIC')


def render_one(out,trace_path,name,title,resume=False):
    import cv2,mujoco
    from tools import render_final_episode_registered_physical_evidence as render
    destination=out/'replays';destination.mkdir(exist_ok=True);target=destination/(name+'.mp4');provenance=destination/(name+'.json')
    expected=dict(trace=record(trace_path),renderer=record(render.__file__),adapter=record(__file__))
    if resume and target.exists() and provenance.exists():
        old=read(provenance)
        if old.get('dependencies')==expected and old.get('video')==record(target):return old
    a=dict(np.load(trace_path));frames=a['control_frame'];indices=np.r_[np.flatnonzero(np.diff(frames)!=0),len(frames)-1];renderer=render.PhysicalRenderer();full='all_measured_q_rad' in a
    extra=[]
    if full:
        for i,joint in enumerate(a['all_joint_names']):
            if joint in a['joint_names']:continue
            j=mujoco.mj_name2id(renderer.model,mujoco.mjtObj.mjOBJ_JOINT,str(joint));assert j>=0;extra.append((i,renderer.model.jnt_qposadr[j]))
    temporary=target.with_name(target.stem+'.incomplete.mp4');args=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1080x540','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(temporary)]
    process=subprocess.Popen(args,stdin=subprocess.PIPE);qkey='MEASURED_Q' if 'MEASURED_Q' in a else 'measured_q_rad'
    try:
        for count,i in enumerate(indices):
            renderer.set_state(a[qkey][i],a['object_position_world_m'][i],a['object_quaternion_xyzw'][i])
            if full:
                for source,destination_q in extra:renderer.data.qpos[destination_q]=a['all_measured_q_rad'][i,source]
                mujoco.mj_forward(renderer.model,renderer.data)
            board=np.full((540,1080,3),248,np.uint8)
            for col,view in enumerate(('top','overview')):board[85:490,col*540:(col+1)*540]=renderer.view(view)
            cv2.putText(board,title,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.58,(25,25,25),1,cv2.LINE_AA)
            note='Full measured articulation FK; actual dynamic object; diagnostic, not a replacement primary attempt' if full else '28D measured arms/hands; waist was not logged (FK approximation); actual dynamic object'
            cv2.putText(board,note,(12,49),cv2.FONT_HERSHEY_SIMPLEX,.46,(35,35,35),1,cv2.LINE_AA)
            cv2.putText(board,'TRAIN illustration | FULL TASK NOT DEMONSTRATED | no attachment / object-state transport writes',(12,72),cv2.FONT_HERSHEY_SIMPLEX,.46,(35,35,135),1,cv2.LINE_AA)
            cv2.putText(board,f"Actual physical t={int(frames[i])/30:.2f} s | {a['stage'][i]}",(12,518),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,25,25),1,cv2.LINE_AA)
            process.stdin.write(board.tobytes())
            if full and int(frames[i]) in (300,850,980):render.atomic_image(out/'figures'/f'MEASURED_FULL_STATE_FRAME_{int(frames[i]):04d}.png',board)
            if count%240==0:print('MEASURED_REPLAY',name,count,len(indices),flush=True)
        process.stdin.close();rc=process.wait(timeout=60)
        if rc:raise RuntimeError('ffmpeg failed with '+str(rc))
    finally:
        renderer.close()
        if process.poll() is None:process.terminate();process.wait(timeout=10)
    temporary.replace(target);probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)],text=True));stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert int(stream['nb_frames'])==len(indices) and stream['width']==1080 and stream['height']==540
    assert abs(float(probe['format']['duration'])-len(indices)/30)<.04
    result=dict(kind='ACTUAL_MEASURED_PROTOTYPE_REPLAY',dependencies=expected,video=record(target),frames=len(indices),fps=30,duration_s=float(probe['format']['duration']),resolution=[1080,540],full_named_articulation=full,source_conditioned=True,full_task_success=False,physics_rerun_for_render=False,object_states_written=False,
        fields=[qkey,'object_position_world_m','object_quaternion_xyzw']+(['all_measured_q_rad'] if full else []),camera_order=['top','overview'],selection_rule='All saved frames of each explicitly named development evidence category; no success-based example selection',limitation='Existing visual mesh/ellipsoid renderer; not a collision/soft-body certificate')
    atomic_json(provenance,result);return result


def replays(out,resume=False):
    n=read(out/'NUMERIC_SUMMARY.json');sid=n['source_id'];base=out/'prototype'/sid;specs=[
        (base/'morphology_acquisition_v4/physics_attempt_01/event_log.npz','TRAIN_ACQUISITION_LIFT_RELEASE_MEASURED','Natural-start acquisition / lift / release | summary-write failure retained'),
        (base/'full_task_connection/ead64a3c8365/physics_attempt_03/event_log.npz','TRAIN_FULL_ATTEMPT_03_MEASURED_28D','TRAIN full attempt 03 | temporary receiver support, then drop outside bin'),
        (base/'full_task_connection/ead64a3c8365/geometry_telemetry_diagnostic/event_log.npz','TRAIN_PROTOTYPE_MEASURED_FULL_STATE_DIAGNOSTIC','Read-only full-state diagnostic | exact command prefix of TRAIN attempt 03')]
    products=[render_one(out,*spec,resume=resume) for spec in specs]
    unavailable=['WRIST_DEV35_TOP_35SPLIT.mp4','WRIST_DEV35_OVERVIEW_35SPLIT.mp4','OURS_DEV35_TOP_35SPLIT.mp4','OURS_DEV35_OVERVIEW_35SPLIT.mp4']
    atomic_json(out/'replays/REPLAY_STATUS.json',dict(status='ACTUAL_PROTOTYPE_REPLAYS_VERIFIED',products=products,DEV35=[dict(file=f,status='NOT_RENDERED_NO_DEV35_ROLLOUTS') for f in unavailable],paired_handoff_clip='NOT_RENDERED_PAIRED_PHYSICAL_N0',all_card_videos_created=False,
        prior_common_control_video=record(ROOT/'outputs/contact_coordination_hybrid/20260907T073437Z/replays/COMMON_CONTROL_MEASURED_HANDOFF.mp4'),prior_control_qualification='NOT_A_GEOMETRICALLY_QUALIFIED_OURS_RESULT'))
    return dict(status='ACTUAL_PROTOTYPE_REPLAYS_VERIFIED',videos=len(products),DEV35_rollout_videos=0)


def direction(out):
    atomic_text(out/'PAPER_FIGURE_DIRECTION.md',f'''# 그림·영상 배치 지침

현재 원고는 전체 물리 비교 논문이 아니라 프로토타입 및 실패 원인 보고 범위다. 아래 PNG와 SVG 경로는 모두 실제 생성 파일이다. 결합 후보 그림을 물리 성공률로 해석하지 않는다.

1. Methods: `{out}/figures/Fig1_Method.png`와 `.svg`. 원본 추출 → 접촉 목표 → 공통 연결/시간 조정 → 실제 부분 실행 순서. 미입증 전체 과제와 보정 요구 표시를 유지한다.
   Caption: “Source-conditioned hybrid prototype using existing source extraction, phase/contact goals, shared motion connection and retiming, and dynamic G1 execution. Natural acquisition, lift and partial handoff were observed, but full task completion was not demonstrated. Privileged simulator states are used; no real-robot or VLA claim is made.”

2. Results/accounting: `{out}/figures/Fig2_AB_Physical_Results.png`와 `.svg`, `TABLE_MAIN_WRIST_VS_OURS.md`. 왼쪽에 Full Task Success NOT MEASURED, 오른쪽에 DEV35 순서의 회색 대응 행렬. 회색을 실패 색상으로 바꾸거나 0% 성공률로 표기하지 않는다.
   Caption: “Planned Wrist-versus-Ours DEV35 development comparison: 35 intended instances per method, zero planning attempts and zero physical rollouts. All cells are upstream-not-attempted. Full task success and paired effects are unmeasured; no physical performance comparison is claimed.”

3. Limitations/development diagnostic: `{out}/figures/Fig3_Coupling_Ablation.png`와 `.svg`, `TABLE_COUPLING_ABLATION_PAIRED10.md`. 위치 불일치 다음 회전 불일치. C는 접촉 후보, S는 초기값이다. 동일 원본의 후보를 독립 표본으로 세지 않는다.
   Caption: “TRAIN-only diagnostic of the same contact-region optimizer with cross-hand residuals enabled or disabled. Three contact candidates and two seeds share all other factors. Predicted object-pose disagreement illustrates a nonredundant algebraic mechanism; inherited contact calibration remains unqualified. The predeclared paired-10 DEV35 physical ablation has observed N=0 and supports no physical coupling attribution.”

4. Prototype evidence: `{out}/figures/PROTOTYPE_MEASURED_STAGES.png`와 `.svg`. 물체 높이, 양손 대향 접촉, 측정 허리 각도 순서. 전체 성공 예시로 사용하지 않는다. `FIRST_FAILURE_CLOSEUP.png`는 초기 원본/목표 형상 진단이며 완전한 충돌 인증이 아니다.
   Caption: “Actual measurements from a fresh-reset, read-only telemetry diagnostic using the unchanged prefix of the third source-conditioned TRAIN plan. Object lift and receiver contact are followed by a drop during giver withdrawal. Measured waist deflection explains the omitted-state FK error. This diagnostic does not replace the failed primary attempt.”

Supplementary videos (illustrations, no typical-performance estimate):

- `{out}/replays/TRAIN_ACQUISITION_LIFT_RELEASE_MEASURED.mp4`: 실제 획득·들기·자연 놓기. 마지막 결과 파일 쓰기 오류와 28D 재구성 한계를 표시한다.
- `{out}/replays/TRAIN_FULL_ATTEMPT_03_MEASURED_28D.mp4`: 보존된 전체 세 번째 시도, 낙하 이후도 포함한다. 허리 미기록에 따른 FK 근사 표시를 유지한다.
- `{out}/replays/TRAIN_PROTOTYPE_MEASURED_FULL_STATE_DIAGNOSTIC.mp4`: 전체 측정 관절을 적용한 동일 명령 접두부의 별도 진단. 원래 시도를 대체하지 않는다.

영상은 위/전체 시점, 실제 물리 시간과 대기를 보존한 30 fps이며 1080×540 검토용 출력이다. 카메라만 바꾸고 저장된 물체 상태를 바꾸지 않는다. 이전 스크립트 보정 영상은 이전 run의 원본을 보존하되 Ours 결과나 완전한 기하 검증으로 사용하지 않는다. 4개 DEV35 7×5 모자이크와 대응 결합 전달 영상은 실제 관측이 없어 생성하지 않았고, 카드 영상으로 대체하지 않았다.
''')
