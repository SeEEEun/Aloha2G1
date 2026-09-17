"""Full measured comparisons and explicitly separate geometric diagnostics."""
from pathlib import Path
import subprocess
import numpy as np
from .io import read,record,atomic_json,atomic_text
from .architecture_videos import (probe,writer,card,preserve_previous,inspect_frames,
    contact_sheet,plot_movie,candidate_demo)


def failure_label(value):
    if isinstance(value,dict):
        cause=value.get('cause',value.get('causal_label',value.get('reason','UNKNOWN')))
        return str(cause)+(' @ '+str(value['phase']) if value.get('phase') else '')
    return str(value) if value is not None else 'NONE'


def phase_metadata(selected,sid):
    """A stationary contact phase inherits its actual endpoint candidate."""
    motion={'APPROACH_CLEARANCE':'APPROACH_CLEARANCE','PREGRASP':'PREGRASP','ACQUISITION_INGRESS':'LEFT_ACQUISITION',
        'LIFT_5CM':'LIFT','LEFT_CARRY':'LEFT_CARRY','RECEIVER_APPROACH':'RECEIVER_APPROACH','GIVER_CLEARANCE':'GIVER_CLEARANCE',
        'RECEIVER_DEPARTURE':'RECEIVER_DEPARTURE','RIGHT_TRANSPORT':'RIGHT_TRANSPORT','PLACE':'PLACE','POST_RELEASE_RETREAT':'POST_RELEASE_RETREAT'}
    holds={'PRESHAPE':'LEFT_ACQUISITION','POWER_GRASP':'LEFT_ACQUISITION','GRAVITY_RETENTION':'LEFT_ACQUISITION',
        'HOLD_ELEVATED':'LIFT','LEFT_CARRY_STABILIZE':'LEFT_CARRY','DUAL_SUPPORT':'RECEIVER_APPROACH',
        'GIVER_RELEASE':'RECEIVER_APPROACH','GIVER_OPENING':'RECEIVER_APPROACH','RIGHT_OWNERSHIP_VERIFY':'RECEIVER_APPROACH',
        'RIGHT_HOLD_OVER_BIN':'RIGHT_TRANSPORT','RIGHT_PRE_RELEASE_STABILIZATION':'PLACE','RIGHT_RELEASE':'PLACE',
        'BIN_SETTLE':'POST_RELEASE_RETREAT','POST_HORIZON_OBSERVATION':'POST_RELEASE_RETREAT'}
    result={}
    for nominal,phase in (motion|holds).items():
        matches=[p for p in selected if p['phase']==phase]
        if matches:
            result[nominal]=dict(candidate=matches[-1]['selected_candidate'].replace(sid+':',''),
                path=('HOLD / ' if nominal in holds else '')+' + '.join(p['certificate'].get('selected_path_id') or 'LOCAL_CONTACT' for p in matches))
    return result


def comparison(paths,labels,target,tile=(960,810)):
    """Synchronized 30 Hz playback; never repeat a last physical frame."""
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    contract=dict(inputs=[record(p) if p else None for p in paths],labels=labels,tile=tile,code=record(__file__))
    receipt=target.with_suffix('.comparison.json')
    if target.exists() and receipt.exists() and read(receipt)['contract']==dict(contract,tile=list(tile)) and read(receipt)['video']==record(target):return target
    preserve_previous(target);caps=[cv2.VideoCapture(str(p)) if p else None for p in paths]
    lengths=[probe(p)['frames'] if p else 0 for p in paths];frames=max(lengths) or 90
    height=tile[1]+40;temp=target.with_name(target.stem+'.incomplete.mp4');proc=writer(temp,(tile[0]*len(paths),height))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
    try:
        for frame in range(frames):
            board=Image.new('RGB',(tile[0]*len(paths),height),(22,29,39));draw=ImageDraw.Draw(board)
            for index,(cap,label,n) in enumerate(zip(caps,labels,lengths)):
                if cap is not None and frame<n:
                    ok,picture=cap.read()
                    if not ok:raise RuntimeError('Premature measured comparison end')
                    picture=cv2.cvtColor(cv2.resize(picture,tile),cv2.COLOR_BGR2RGB)
                else:
                    picture=card(tile,[*label,'RECORDED HORIZON ENDED' if cap else 'NO_COMPLETE_COMMAND',
                        'TRACE TYPE: END CARD' if cap else 'TRACE TYPE: NONE',
                        'No physical states after this horizon' if cap else 'No official physical rollout exists'])
                board.paste(Image.fromarray(picture),(index*tile[0],40))
                draw.text((index*tile[0]+12,6),label[0],font=font,fill='white')
            proc.stdin.write(np.asarray(board).tobytes())
        proc.stdin.close();assert proc.wait(timeout=60)==0
    finally:
        for cap in caps:
            if cap:cap.release()
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    temp.replace(target);assert probe(target)['frames']==frames
    atomic_json(receipt,dict(contract=contract,video=record(target),frames=frames,same_camera=True,
        same_physical_timing=True,task_failure_never_cuts_recording=True,shorter_trace_end_card=True))
    return target


def concatenate(paths,target):
    receipt=target.with_suffix('.concat.json');inputs=[record(p) for p in paths]
    if target.exists() and receipt.exists() and read(receipt).get('inputs')==inputs and read(receipt)['video']==record(target):return target
    preserve_previous(target);listing=target.with_suffix('.concat.txt')
    atomic_text(listing,''.join("file '"+str(p)+"'\n" for p in paths));temp=target.with_name(target.stem+'.incomplete.mp4')
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(temp)],check=True)
    temp.replace(target);assert probe(target)['frames']==sum(probe(p)['frames'] for p in paths)
    atomic_json(receipt,dict(inputs=inputs,video=record(target),order_preserved=True));return target


def path_demo(out,target,selection=False):
    """Actual stored G1 plans displayed as FK curves, never as physics."""
    from .source_phase import COMMON
    from .planning_kinematics import G1Kinematics
    from .source_motion_prior import attach,MotionGuide
    from .path_quality import resample
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    evidence=read(out/'SOURCE_GUIDED_PATH_EVIDENCE.json')
    # Predeclared: most retained alternatives, fixed source/method/phase tie
    # order. Distinct phases and numerical paths; no physical outcomes used.
    pool=[r for r in evidence['rows'] if r['version']=='NEW' and r['certificate']['algorithm']=='SOURCE_GUIDED_RRT_CONNECT']
    pool=sorted(pool,key=lambda r:(-len(r['certificate']['path_candidates']),r['source_id'],r['method'],r['phase']))
    cases=[];seen=set()
    for row in pool:
        if row['phase'] in seen:continue
        cases.append(row);seen.add(row['phase'])
        if len(cases)==3:break
    if len(cases)<3:raise RuntimeError('Need three actual G1 phase connections for path visualization')
    contract=dict(evidence=record(out/'SOURCE_GUIDED_PATH_EVIDENCE.json'),selection=selection,code=record(__file__))
    receipt=target.with_suffix('.paths.json')
    if target.exists() and receipt.exists() and read(receipt)['contract']==contract and read(receipt)['video']==record(target):return target
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));draw_data=[]
    for row in cases:
        cert=row['certificate'];q=np.asarray(row['path']);goal=attach(row['goal'],out/'source_phase'/row['source_id'])
        active=np.asarray(cert['active_indices']);goal=dict(goal,active_hands=['left' if min(active)<7 else 'right']) if len(active)==7 else goal
        guide=MotionGuide(g,goal,q[0],q[-1],active)
        def wrist(path):return np.asarray([guide.features(p)['wrists'][0] for p in resample(np.asarray(path),81)])
        candidates=[]
        for candidate in cert['path_candidates']:
            candidates.append(dict(candidate,xyz=wrist(candidate['waypoints'])))
        draw_data.append(dict(row=row,hand=guide.hands[0],prior=guide.anchors[:,0],final=wrist(q),candidates=candidates))
    def draw(fig,frame):
        import textwrap
        from matplotlib.ticker import MaxNLocator
        index=min(frame//150,len(draw_data)-1);local=frame%150;case=draw_data[index];row=case['row'];cert=row['certificate']
        ax=fig.add_subplot(121,projection='3d');final=case['final'];prior=case['prior']
        for candidate in case['candidates']:
            chosen=candidate['path_id']==cert['selected_path_id'];xyz=candidate['xyz']
            ax.plot(*xyz.T,color='#c78521' if chosen else '#9aa2b0',lw=2 if chosen else 1,
                alpha=.9 if chosen else .5,label='Selected RRT, before smoothing' if chosen else None)
        ax.plot(*prior.T,'o--',color='#7652a0',lw=2,markersize=4,label='Compact source soft prior')
        ax.plot(*final.T,color='#007e73',lw=2.5,label='Smoothed / fully revalidated')
        ax.scatter(*final[0],color='black',s=55,label='G1 start');ax.scatter(*final[-1],color='#c44a43',marker='*',s=140,label='Interaction endpoint')
        moving=final[min(80,int(local*80/130))];ax.scatter(*moving,color='#007e73',s=80)
        ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_zlabel('World Z (m)')
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.set_major_locator(MaxNLocator(4))
        ax.view_init(elev=24,azim=-55);ax.legend(loc='upper left',fontsize=7)
        ax.set_title(row['phase']+' | active '+case['hand']+' wrist',fontsize=10)
        chart=fig.add_subplot(222);values=[c['score'] for c in case['candidates']]
        colors=['#007e73' if c['path_id']==cert['selected_path_id'] else '#b1b9c4' for c in case['candidates']]
        chart.barh(range(len(values)),values,color=colors);chart.set_yticks(range(len(values)),[c['path_id'] for c in case['candidates']],fontsize=8)
        chart.set_xlim(0,max(.1,max(values)*1.2));chart.invert_yaxis();chart.set_xlabel('Common normalized path cost (lower is better)',fontsize=9)
        text=fig.add_subplot(224);text.axis('off')
        metrics=cert['final_path_quality']
        lines=[f"Actual candidate: {row['selected_candidate'].replace(row['source_id']+':','')}",
            f"Selected path: {cert['selected_path_id']}",
            f"RRT expanded nodes: {cert['rrt_search_expanded']}",
            f"Source / global samples: {cert['source_guided_samples']} / {cert['global_samples']}",
            f"Final joint travel: {metrics['joint_path_length_rad']:.3f} rad",
            f"Source RMS deviation: {metrics['source_motion_deviation_m']:.3f} m",
            f"Final quality cost: {metrics['score']:.4f}",
            'Endpoints fixed; all changed states / edges checked',
            'Source prior may be left for feasibility or quality']
        wrapped=[part for line in lines for part in textwrap.wrap(line,width=68,break_long_words=False,break_on_hyphens=False)]
        text.text(0,.98,'\n'.join(wrapped),va='top',fontsize=9.5,linespacing=1.5)
        fig.suptitle(('Candidate and path quality selection' if selection else 'Source-guided RRT path visualization')+
            '\n'+row['source_id']+' | '+row['method']+f' | {index+1}/3',fontsize=14)
        fig.text(.03,.025,'TRACE TYPE: GEOMETRIC PLAN DIAGNOSTIC — NOT MEASURED PHYSICS / NOT A TASK RESULT',fontsize=10,color='#a3383d')
        fig.subplots_adjust(left=.05,right=.98,bottom=.09,top=.83,wspace=.35,hspace=.65)
    plot_movie(target,450,draw)
    atomic_json(receipt,dict(contract=contract,video=record(target),cases=cases,
        selection_rule='Most retained alternatives; distinct phase; source/method/phase lexical tie; no physics outcomes',
        trace_type='GEOMETRIC_PLAN_DIAGNOSTIC',executed_physics=False))
    return target


def run(out):
    from .full_attempt_replay import render
    from .source_guided_reports import selected_phases
    fixed=read(out/'COVERAGE8.json');rows=read(out/'PHYSICAL_VERIFICATION.json')['rows']
    plans=read(out/'GOLDEN_PLANNING.json')['rows']+read(out/'COVERAGE8_PLANNING.json')['rows']
    plan_by={(r['source_id'],r['method_key']):r for r in plans};by={(r['source_id'],r['method_key']):r for r in rows}
    old=Path(read(out/'SOURCE_GUIDED_RRT_REBUILD.json')['old_run']);old_rows=read(old/'PHYSICAL_VERIFICATION.json')['rows'];old_by={(r['source_id'],r['method_key']):r for r in old_rows}
    videos=out/'videos';videos.mkdir(exist_ok=True);outputs=[];sheets=[];mapping={}
    for index,row in enumerate(rows):
        sid=row['source_id'];method=row['method_key'];key=(sid,method);code='B' if method=='B_INDEPENDENT' else 'C';mapping[key]=None
        name=f'GOLDEN_SOURCE_GUIDED_RRT_{code}.mp4' if sid==fixed['golden'] else sid+'_'+code+'_SOURCE_GUIDED_RRT_FULL.mp4';target=videos/name
        if not row['physics_executed']:
            from PIL import Image
            shot=videos/'contact_sheets'/(sid+'_'+method+'.png');shot.parent.mkdir(exist_ok=True)
            Image.fromarray(card((960,810),[sid,method,'NO_COMPLETE_COMMAND','TRACE TYPE: NONE',
                'FIRST FAILURE: '+failure_label(row['first_failure']),'No measured contact sheet exists'])).save(shot);sheets.append(shot)
            if sid==fixed['golden']:
                comparison([None],[[method,sid,'NO_COMPLETE_COMMAND',failure_label(row['first_failure'])]],target);outputs.append(target)
            continue
        atomic_json(out/'WORK_PROGRESS.json',dict(stage='full_video_validation',source_id=sid,method=method,
            substage='Full measured replay',completed_work=index,remaining_work=len(rows)-index))
        selected=selected_phases(plan_by[key]);phase_selection=phase_metadata(selected,sid)
        folder=Path(row['folder']);atomic_json(folder/'SOURCE_GUIDED_RENDER_METADATA.json',dict(phase_selection=phase_selection,
            selected_geometry=record(out/'SOURCE_GUIDED_PATH_EVIDENCE.json'),measured_trace=record(folder/'event_log.npz')))
        try:render(folder,target,resume=True)
        except ValueError as error:
            if 'Changed replay inputs' not in str(error):raise
            preserve_previous(target);render(folder,target,resume=True)
        mapping[key]=target;outputs.append(target);sheets.append(contact_sheet(row,target,videos))
    def label(sid,method,version='NEW'):
        row=(by if version=='NEW' else old_by)[(sid,method)]
        return [version+' | '+method,sid,'SOURCE-GUIDED RRT' if version=='NEW' else 'FROZEN ARCHITECTURE REPAIR',
            'FIRST FAILURE: '+failure_label(row.get('first_failure'))]
    pairs=[]
    for sid in fixed['source_ids']:
        pair=comparison([mapping[(sid,m)] for m in ('B_INDEPENDENT','C_COUPLED')],
            [label(sid,m) for m in ('B_INDEPENDENT','C_COUPLED')],videos/(sid+'_BC_SOURCE_GUIDED_PAIR.mp4'))
        pairs.append(pair)
    outputs+=pairs;outputs.append(concatenate(pairs,videos/'BC_COVERAGE8_SOURCE_GUIDED_RRT_FULL.mp4'))
    comparisons=[]
    for sid in [fixed['golden'],*fixed['source_ids']]:
        for method,code in [('B_INDEPENDENT','B'),('C_COUPLED','C')]:
            old_path=old/'videos'/(f'GOLDEN_{code}_FULL_REPAIRED.mp4' if sid==fixed['golden'] else sid+'_'+code+'_FULL_REPAIRED.mp4')
            if not old_by[(sid,method)]['physics_executed']:old_path=None
            elif not old_path.exists():raise FileNotFoundError('Frozen measured replay absent: '+str(old_path))
            comparisons.append(comparison([old_path,mapping[(sid,method)]],[label(sid,method,'OLD'),label(sid,method)],
                videos/(sid+'_'+code+'_BEFORE_AFTER.mp4')))
    outputs+=comparisons;outputs.append(concatenate(comparisons,videos/'SOURCE_GUIDED_RRT_BEFORE_AFTER.mp4'))
    outputs.append(path_demo(out,videos/'PLANNER_PATH_VISUALIZATION.mp4'))
    candidate=candidate_demo(out,videos/'INTERACTION_CANDIDATES_DIAGNOSTIC.mp4')
    path_selection=path_demo(out,videos/'PATH_QUALITY_SELECTION_DIAGNOSTIC.mp4',selection=True)
    outputs += [candidate,path_selection,concatenate([candidate,path_selection],videos/'CANDIDATE_AND_PATH_SELECTION.mp4')]
    inspection=[inspect_frames(p,videos) for p in outputs];report=out/'SOURCE_GUIDED_VIDEO_VERIFICATION.json'
    atomic_json(report,dict(status='AUTOMATED_PASS_VISUAL_PENDING',videos=inspection,contact_sheets=[record(p) for p in sheets],
        solid_measured_geometry=True,true_horizon_from_physical_receipts=True,diagnostics_excluded_from_results=True,
        old_traces=record(old/'PHYSICAL_VERIFICATION.json'),new_traces=record(out/'PHYSICAL_VERIFICATION.json')))
    markdown=out/'VIDEO_VERIFICATION.md';atomic_text(markdown,'# Source-guided RRT visual verification\n\n'
        'Golden and every feasible fixed Coverage8 B/C command are replayed from actual PhysX substep states, with solid robot geometry through the true command horizon. '
        'All eight sources appear in fixed-order paired videos. No-command cases have explicit cards; shorter recordings have end cards. No task failure freezes a measured video. '
        'The before/after comparison uses unchanged frozen old measured videos on the left and current measured videos on the right, the same camera and 30 Hz physical timing. '
        'Geometric candidate and path diagnostics are continuously labelled and excluded from measured scores. '
        'Every MP4 has ffprobe and beginning/middle/end decode evidence in SOURCE_GUIDED_VIDEO_VERIFICATION.json.\n')
    review=out/'VISUAL_INSPECTION.json'
    if not review.exists():raise RuntimeError('VISUAL_INSPECTION_REQUIRED: agent must inspect saved frames')
    accepted=read(review)
    if not accepted.get('pass') or sorted(accepted.get('videos',[]),key=lambda r:r['path'])!=sorted([record(p) for p in outputs],key=lambda r:r['path']):
        raise RuntimeError('VISUAL_INSPECTION_REQUIRED: inspect all current hashes')
    value=read(report);value.update(status='PASS',visual_inspection=record(review));atomic_json(report,value)
    return [report,markdown,review,*outputs,*sheets]
