"""Measured full-horizon videos, explicit no-command tiles and diagnostics."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from .io import read,record,atomic_json,atomic_text


def preserve_previous(target):
    """Changed render inputs invalidate a result; retain every previous video."""
    import time
    if not target.exists():return
    archive=target.parent/'INVALIDATED_VIDEOS'/(target.stem+'_'+str(time.time_ns()));archive.mkdir(parents=True)
    paths=[target,target.with_suffix('.json'),*target.parent.glob(target.stem+'.*.json'),
        *(target.parent/'inspection_frames').glob(target.stem+'_*.png')]
    for path in set(paths):
        if path.exists():path.rename(archive/path.name)


def probe(path):
    result=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,check=True)
    value=json.loads(result.stdout);stream=next(s for s in value['streams'] if s['codec_type']=='video')
    return dict(codec=stream['codec_name'],width=int(stream['width']),height=int(stream['height']),
        frames=int(stream['nb_frames']),duration_s=float(value['format']['duration']),fps=stream['avg_frame_rate'])


def writer(path,size):
    return subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','30','-i','-',
        '-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)


def inspect_frames(path,out):
    import cv2
    info=probe(path);cap=cv2.VideoCapture(str(path));shots=[]
    for label,index in [('begin',0),('middle',info['frames']//2),('end',info['frames']-1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
        if not ok:raise RuntimeError('Undecodable '+label+' frame: '+str(path))
        image=out/'inspection_frames'/(path.stem+'_'+label+'.png');image.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(image),frame);shots.append(record(image))
    cap.release();return dict(video=record(path),ffprobe=info,snapshots=shots)


def card(size,lines,color=(30,35,45)):
    from PIL import Image,ImageDraw,ImageFont
    canvas=Image.new('RGB',size,color);draw=ImageDraw.Draw(canvas)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,max(12,size[0]//35))
    for index,line in enumerate(lines):
        text=str(line);f=font;width=draw.textlength(text,font=f)
        if width>size[0]-28:f=ImageFont.truetype(font_path,max(10,int(font.size*(size[0]-28)/width)))
        draw.text((14,20+index*max(24,size[0]//22)),text,font=f,fill=(238,238,238))
    return np.asarray(canvas)


def composite(paths,labels,target,columns,rows,tile=(640,540)):
    import cv2
    contract=dict(inputs=[record(p) if p else None for p in paths],labels=labels,columns=columns,rows=rows,tile=list(tile),
        implementation=record(__file__))
    receipt=target.with_suffix('.composition.json')
    if receipt.exists() and target.exists() and read(receipt)['contract']==contract and read(receipt)['video']==record(target):return target
    preserve_previous(target)
    caps=[cv2.VideoCapture(str(p)) if p else None for p in paths]
    lengths=[probe(p)['frames'] if p else 0 for p in paths];frames=max(lengths,default=0) or 90
    temp=target.with_name(target.stem+'.incomplete.mp4');proc=writer(temp,(tile[0]*columns,tile[1]*rows))
    try:
        for frame in range(frames):
            board=np.zeros((tile[1]*rows,tile[0]*columns,3),np.uint8)
            for index,(cap,label,n) in enumerate(zip(caps,labels,lengths)):
                if cap is not None and frame<n:
                    ok,picture=cap.read()
                    if not ok:raise RuntimeError('Premature physical video end')
                    picture=cv2.cvtColor(cv2.resize(picture,tile),cv2.COLOR_BGR2RGB)
                else:
                    picture=card(tile,[*label,'RECORDED HORIZON ENDED' if cap else 'NO_COMPLETE_COMMAND',
                        'No physical states after this horizon' if cap else 'No official physical rollout exists',
                        'TRACE: END CARD' if cap else 'TRACE: NONE'])
                y=(index//columns)*tile[1];x=(index%columns)*tile[0];board[y:y+tile[1],x:x+tile[0]]=picture
            proc.stdin.write(board.tobytes())
        proc.stdin.close();assert proc.wait(timeout=60)==0
    finally:
        for cap in caps:
            if cap:cap.release()
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    temp.replace(target);assert probe(target)['frames']==frames
    atomic_json(receipt,dict(contract=contract,video=record(target),frames=frames,
        same_camera=True,same_physical_timing=True,shorter_trace_end_cards=True,no_failure_freeze=True))
    return target


def plot_movie(target,frames,draw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    preserve_previous(target)
    temp=target.with_name(target.stem+'.incomplete.mp4');fig=plt.figure(figsize=(12.8,7.2),dpi=100)
    proc=writer(temp,(1280,720))
    try:
        for index in range(frames):
            fig.clear();draw(fig,index);fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].tobytes())
        proc.stdin.close();assert proc.wait(timeout=60)==0
    finally:
        plt.close(fig)
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    temp.replace(target);return target


def planner_fixture_demo(out,target):
    import matplotlib.patches as patches
    cases=read(out/'PLANNER_SEARCH_REGRESSION.json')['cases']
    if target.exists():return target
    def draw(fig,frame):
        case=cases[min(frame//120,len(cases)-1)];local=frame%120
        ax=fig.add_subplot(111);ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_aspect('equal')
        cx,cy,hx,hy=case['obstacle'];ax.add_patch(patches.Rectangle((cx-hx,cy-hy),2*hx,2*hy,color='#626b78',alpha=.8))
        a=np.asarray(case['q_start']);b=np.asarray(case['q_goal']);path=np.asarray(case['result']['path'])
        ax.plot(*np.array([a,b]).T,'--',color='#bf363c',lw=3,label='DIRECT PATH: REJECTED')
        ax.plot(*path.T,color='#00796b',lw=3,label='PLANNER PATH: ACCEPTED')
        fraction=min(1.,local/100);lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);cumulative=np.r_[0,np.cumsum(lengths)]
        distance=fraction*cumulative[-1];segment=min(np.searchsorted(cumulative,distance,side='right')-1,len(path)-2)
        u=(distance-cumulative[segment])/max(lengths[segment],1e-12);q=(1-u)*path[segment]+u*path[segment+1]
        ax.scatter(*q,s=160,color='#00796b');ax.scatter(*a,s=100,color='black');ax.scatter(*b,s=100,color='black',marker='x')
        ax.legend(loc='lower right');ax.set_xlabel('Joint coordinate q0 (rad)');ax.set_ylabel('Joint coordinate q1 (rad)')
        ax.set_title(f'Planner regression {case["case"]+1}/3 — deterministic RRT-Connect\n'
            '2-DOF obstacle fixture for the exact common planner — diagnostic, not PhysX')
        fig.text(.12,.015,f"State checks: {case['result']['state_checks']} | Edge checks: {case['result']['edge_checks']} | Fixed seed:1729",fontsize=12)
        fig.tight_layout(rect=[0,.04,1,1])
    return plot_movie(target,360,draw)


def planner_demo(out,target):
    """Three actual G1 connections whose direct joint chords were rejected."""
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    from .paper_replays import renderer_class
    from .morphology_repair import world_wrist
    cases=[];seen_paths=set()
    planned=[*read(out/'GOLDEN_PLANNING.json')['rows'],*read(out/'COVERAGE8_PLANNING.json')['rows']]
    for row in planned:
        if not row['full_task_plan']:continue
        context=Path(row['context']);sid=row['source_id']
        early=context/'prototype'/sid/'morphology_acquisition_v4/PLAN_RESULT.json'
        connection=Path(row['plan']).parent;summary=read(connection/'RESULT.json')
        chosen=next(r for r in summary['results'] if r['all_admissible'])
        late=connection/chosen['subdirectory']/'PHASE_IK.json'
        for path in (early,late):
            for phase in read(path)['phases']:
                valid=next((p for p in phase.get('planner_attempts',[]) if p['status']=='PATH_FOUND' and p['search_used'] and not p['direct_path_valid']),None)
                if valid and phase['phase'] in ('PREGRASP','LEFT_CARRY','RECEIVER_APPROACH'):
                    key=np.asarray(valid['path']).round(8).tobytes()
                    if key in seen_paths:continue
                    seen_paths.add(key)
                    cases.append(dict(source_id=sid,method=row['method_key'],context=str(context),phase=phase['phase'],
                        planner=valid,evidence=record(path),plan=row['plan']))
        if len(cases)>=3:break
    cases=cases[:3]
    if len(cases)<3:raise RuntimeError('Need three actual G1 rejected-direct / accepted-planner cases')
    contract=dict(cases=cases,code=record(__file__))
    receipt=target.with_suffix('.cases.json')
    if target.exists() and receipt.exists() and read(receipt)['contract']==contract:return target
    preserve_previous(target)
    renderer=renderer_class()();renderer.model.geom_rgba[:,3]=1.
    import mujoco
    from tools.render_final_episode_registered_physical_evidence import camera,_add_geom
    renderer.cameras['overview']=camera([1.1,.9,1.7],[.42,.1,1.0])
    old_scene=renderer.add_scene;marker=[None]
    def add_scene():
        old_scene()
        if marker[0] is not None:
            _add_geom(renderer.renderer.scene,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.013,0.,0.]),
                np.asarray(marker[0]),np.array([.9,.08,.1,1.],np.float32))
    renderer.add_scene=add_scene
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    temp=target.with_name(target.stem+'.incomplete.mp4');proc=writer(temp,(1280,720))
    try:
        for index,case in enumerate(cases):
            context=Path(case['context']);cal=read(context/'target_repair/CONTACT_CALIBRATION.json')
            selected=read(Path(case['plan'])/'CONTACT_SELECTION.json')
            original=np.asarray(read(context/'source_phase'/case['source_id']/'PHASE_RECORD.json')['initial_object_pose_world'])
            path=np.asarray(case['planner']['path']);length=np.linalg.norm(np.diff(path,axis=0),axis=1);cum=np.r_[0,np.cumsum(length)]
            collision=case['planner']['first_invalid']['forbidden_contacts'][0]
            for frame in range(150):
                progress=np.clip((frame-15)/120.,0.,1.)
                direct=path[0]+progress*(path[-1]-path[0]);distance=progress*cum[-1]
                segment=min(max(np.searchsorted(cum,distance,side='right')-1,0),len(path)-2)
                u=(distance-cum[segment])/max(length[segment],1e-12);planned=(1-u)*path[segment]+u*path[segment+1]
                board=Image.new('RGB',(1280,720),(244,246,249));draw=ImageDraw.Draw(board)
                draw.text((20,10),'G1 collision-aware path repair: '+case['phase'],font=font,fill='black')
                draw.text((20,43),case['source_id']+' | '+case['method']+' | Case '+str(index+1)+'/3',font=small,fill='black')
                for side,q in enumerate((direct,planned)):
                    marker[0]=collision['point_world'] if side==0 else None
                    fingers=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q']);obj=original
                    if case['phase']!='PREGRASP':
                        fingers=fingers.copy();fingers[:7]=selected['contacts']['left']['measured_finger_q'][:7]
                        relation=selected['carry_contacts']['left'];renderer.g1.assign(q)
                        obj=world_wrist(renderer.g1,'left')@np.asarray(relation['T_wrist_H'])@np.asarray(relation['T_HO'])
                    from scipy.spatial.transform import Rotation
                    renderer.set_state(np.r_[q,fingers],obj[:3,3],Rotation.from_matrix(obj[:3,:3]).as_quat())
                    picture=cv2.cvtColor(renderer.view('overview'),cv2.COLOR_BGR2RGB)
                    board.paste(Image.fromarray(picture).resize((620,465)),(10+side*640,105))
                    draw.text((20+side*640,78),'DIRECT PATH: REJECTED' if side==0 else 'PLANNER PATH: ACCEPTED',font=font,fill='#bc3038' if side==0 else '#00776b')
                draw.text((20,590),'TRACE TYPE: GEOMETRIC PLANNER DIAGNOSTIC — NOT MEASURED PHYSICS',font=small,fill='black')
                bodies=' / '.join(collision['bodies']).replace('_link','').replace('_hand_',' hand ')
                draw.text((20,625),'Red marker: direct-chord collision location — '+bodies,font=small,fill='black')
                draw.text((20,660),f"Search states: {case['planner']['state_checks']} | Edges: {case['planner']['edge_checks']} | Fixed seed1729",font=small,fill='black')
                proc.stdin.write(np.asarray(board).tobytes())
        proc.stdin.close();assert proc.wait(timeout=60)==0
    finally:
        renderer.close()
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    temp.replace(target);atomic_json(receipt,dict(contract=contract,video=record(target),actual_G1_cases=3,
        distinct_numerical_paths=True,physics_result=False))
    return target


def candidate_demo(out,target):
    # Predeclare selection from planning evidence, never physical outcomes:
    # most complete-chain branches attempted, then lowest source identifier.
    planned=[*read(out/'GOLDEN_PLANNING.json')['rows'],*read(out/'COVERAGE8_PLANNING.json')['rows']]
    eligible=[r for r in planned if r['method_key']=='C_COUPLED' and r['full_task_plan']]
    chosen=sorted(eligible,key=lambda r:(-len(r['attempts']),r['source_id']))[0]
    context=Path(chosen['context']);sid=chosen['source_id']
    pointer=read(context/'target_repair/CURRENT_CONTACT_REGION_FIT.json');folder=Path(pointer['path'])
    bank=read(folder/'ENDPOINT_BANK.json')['bank'];ranking=read(folder/'RANKING_True.json')
    contact=read(Path(chosen['plan'])/'CONTACT_SELECTION.json');selected=contact['score']
    grasp_path=Path(chosen['receipt']).parent/'GRASP_CANDIDATES.json';grasp=read(grasp_path)
    contract=dict(grasp=record(grasp_path),bank=record(folder/'ENDPOINT_BANK.json'),
        contact=record(Path(chosen['plan'])/'CONTACT_SELECTION.json'),code=record(__file__))
    receipt=target.with_suffix('.candidates.json')
    if target.exists() and receipt.exists() and read(receipt)['contract']==contract:return target
    reasons={}
    for candidate in bank['right']:
        ik=candidate['IK_result']['phases'][0]
        reasons[candidate['candidate_id']]='NO_IK' if not ik['goal_satisfied'] else 'COLLISION' if not candidate['valid'] else 'GEOMETRY_VALID'
    for candidate in grasp:
        attempt=next((a for a in chosen['attempts'] if a['candidate_id']==candidate['candidate_id']),None)
        reasons[candidate['candidate_id']]=('SELECTED_COMPLETE_CHAIN' if candidate['candidate_id']==chosen['selected_candidate_id'] else
            'COMPLETE_CHAIN_VALID_NOT_SELECTED' if attempt and attempt['complete'] else
            attempt['first_failure']['cause']+' @ '+attempt['first_failure']['phase'] if attempt else 'NOT_EVALUATED_AFTER_COMPLETE_CHAIN')
    def draw(fig,frame):
        ax=fig.add_subplot(111,projection='3d');handoff=frame>=150
        candidates=bank['right'] if handoff else grasp
        for index,candidate in enumerate(candidates):
            xyz=np.asarray(candidate['task_space_target'])[:3,3]
            reason=reasons[candidate['candidate_id']]
            color='#187e72' if reason in ('GEOMETRY_VALID','SELECTED_COMPLETE_CHAIN','COMPLETE_CHAIN_VALID_NOT_SELECTED') else '#486bb4' if reason.startswith('NOT_EVALUATED') else '#c44143'
            ax.scatter(*xyz,color=color,s=40,alpha=.7)
            if len(candidates)<10:ax.text(*xyz,candidate['candidate_id'].rsplit(':',1)[1])
        if handoff:
            shared=np.asarray(selected['shared_object_pose'])[:3,3]
            ax.scatter(*shared,s=150,color='#f3af22',marker='*',label='C shared object pose X')
            l=np.asarray(bank['left'][selected['left']]['object_pose'])[:3,3];r=np.asarray(bank['right'][selected['right']]['object_pose'])[:3,3]
            ax.plot(*np.array([l,shared,r]).T,color='#f3af22',lw=3);ax.legend()
            for side,values in [('left',bank['left']),('right',bank['right'])]:
                target_pose=np.asarray(values[selected[side]]['task_space_target'])[:3,3]
                ax.scatter(*target_pose,s=150,color='#f3af22',marker='s')
                ax.plot(*np.array([target_pose,shared]).T,color='#d99a16',lw=1.5)
        else:
            picked=next(c for c in grasp if c['candidate_id']==chosen['selected_candidate_id'])
            ax.scatter(*np.asarray(picked['task_space_target'])[:3,3],s=180,color='#f3af22',marker='*',label='Selected complete chain');ax.legend()
        ax.view_init(elev=25,azim=-50+(frame%150)*.25)
        ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_zlabel('World Z (m)')
        ax.set_title(('HANDOFF' if handoff else 'GRASP')+' candidate selection — C_COUPLED — '+sid+'\nTask-space candidates, diagnostic visualization; no physics substituted')
        from collections import Counter
        counts=Counter(reasons[c['candidate_id']] for c in candidates)
        if handoff:counts.update({'CROSS_HAND_INCOMPATIBLE_PAIRS':sum(r['common_physical_validation']=='CROSS_HAND_INCOMPATIBLE' for r in ranking)})
        fig.text(.03,.06,' | '.join(k+': '+str(v) for k,v in counts.items()),fontsize=9)
        fig.text(.03,.035,'Gold: actual selected candidate / shared X | Green: valid | Red: rejected | Blue: unevaluated',fontsize=10)
        fig.text(.03,.01,'Finite hierarchical selection; rejected and unevaluated alternatives are distinguished. No physical outcome input.',fontsize=9)
    result=plot_movie(target,300,draw)
    atomic_json(receipt,dict(contract=contract,video=record(target),rejection_reasons=reasons,
        selected_grasp=chosen['selected_candidate_id'],selected_handoff=contact['selected_candidate_ids'],
        selection_rule='C complete plan with most attempted grasp branches, then lowest source ID; no physics outcomes used'))
    return result


def contact_sheet(row,video,out):
    from PIL import Image,ImageDraw
    from .full_attempt_replay import recorded_failures
    import cv2
    folder=Path(row['folder']);trace=np.load(folder/'event_log.npz');labels=trace['NOMINAL_PHASE'].astype(str)
    frames=trace['control_frame'];rec=read(folder/'FULL_ATTEMPT_RECORDING.json')
    failure,_=recorded_failures(folder,rec,read(folder/'ABC_NOMINAL_SCORE.json'),probe(video)['frames'])
    requested=[('START',None),('GRASP','POWER_GRASP'),('LIFT','LIFT_5CM'),('HANDOFF','DUAL_SUPPORT'),
               ('OWNERSHIP','RIGHT_OWNERSHIP_VERIFY'),('PLACE','PLACE'),('FIRST FAILURE',None)]
    cap=cv2.VideoCapture(str(video));board=Image.new('RGB',(4*480,2*430),'white');draw=ImageDraw.Draw(board)
    for index,(label,phase) in enumerate(requested):
        if label=='START':frame=0
        elif label=='FIRST FAILURE':frame=failure['control_frame'] if failure else None
        else:
            matches=np.flatnonzero(labels==phase);frame=int(frames[matches[len(matches)//2]]) if len(matches) else None
        if frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,picture=cap.read()
            if not ok:raise RuntimeError('Missing contact-sheet measured frame')
            picture=Image.fromarray(cv2.cvtColor(cv2.resize(picture,(480,406)),cv2.COLOR_BGR2RGB))
        else:picture=Image.fromarray(card((480,406),[row['source_id'],row['method_key'],label,'NONE / NOT RECORDED']))
        x=(index%4)*480;y=(index//4)*430;board.paste(picture,(x,y+22));draw.text((x+5,y+3),label,fill='black')
    cap.release();path=out/'contact_sheets'/(row['source_id']+'_'+row['method_key']+'.png');path.parent.mkdir(exist_ok=True);board.save(path)
    return path


def run(out):
    from .full_attempt_replay import render
    fixed=read(out/'COVERAGE8.json');rows=read(out/'PHYSICAL_VERIFICATION.json')['rows']
    by_key={(r['source_id'],r['method_key']):r for r in rows}
    def labels_for(sid,method):
        row=by_key[(sid,method)];lines=[sid,method]
        if not row['physics_executed']:
            fail=row['first_failure']
            lines+=['CURRENT PHASE: '+fail['phase'],'FIRST FAILURE: '+fail['cause'],
                    'SELECTED CANDIDATE: NONE','PLANNER: NO_COMPLETE_CHAIN']
        return lines
    videos=out/'videos';videos.mkdir(exist_ok=True);mapping={};outputs=[];sheets=[]
    for index,row in enumerate(rows):
        key=(row['source_id'],row['method_key']);mapping[key]=None
        if not row['physics_executed']:
            from PIL import Image
            path=videos/'contact_sheets'/(key[0]+'_'+key[1]+'.png');path.parent.mkdir(exist_ok=True)
            Image.fromarray(card((960,810),[key[0],key[1],'NO_COMPLETE_COMMAND',
                'START / GRASP / LIFT / HANDOFF / OWNERSHIP / PLACE: NOT RECORDED',
                'FIRST CAUSE: '+row['first_failure']['cause']+' / '+row['first_failure']['phase'],'TRACE TYPE: NONE','No official physical contact sheet exists'])).save(path)
            sheets.append(path);continue
        atomic_json(out/'WORK_PROGRESS.json',dict(stage='full_video_validation',source_id=key[0],method=key[1],
            substage='Render solid measured full trace',completed_work=index,remaining_work=len(rows)-index))
        code='B' if key[1]=='B_INDEPENDENT' else 'C'
        name=f'GOLDEN_{code}_FULL_REPAIRED.mp4' if key[0]==fixed['golden'] else key[0]+'_'+code+'_FULL_REPAIRED.mp4'
        path=videos/name
        try:render(Path(row['folder']),path,resume=True)
        except ValueError as error:
            if 'Changed replay inputs' not in str(error):raise
            preserve_previous(path);render(Path(row['folder']),path,resume=True)
        mapping[key]=path;outputs.append(path)
        sheets.append(contact_sheet(row,path,videos))
    for method,code in [('B_INDEPENDENT','B'),('C_COUPLED','C')]:
        paths=[mapping[(sid,method)] for sid in fixed['source_ids']]
        labels=[labels_for(sid,method) for sid in fixed['source_ids']]
        outputs.append(composite(paths,labels,videos/f'{code}_COVERAGE8_FULL_REPAIRED.mp4',4,2))
    pairs=[]
    for sid in fixed['source_ids']:
        pairs.append(composite([mapping[(sid,m)] for m in ('B_INDEPENDENT','C_COUPLED')],
            [labels_for(sid,'B_INDEPENDENT'),labels_for(sid,'C_COUPLED')],videos/(sid+'_BC_PAIR.mp4'),2,1,tile=(960,810)))
    listing=videos/'BC_PAIRS.concat.txt';atomic_text(listing,''.join("file '"+str(p)+"'\n" for p in pairs))
    target=videos/'BC_COVERAGE8_SIDE_BY_SIDE.mp4'
    concat_receipt=target.with_suffix('.concat.json');concat_inputs=[record(p) for p in pairs]
    if not (target.exists() and concat_receipt.exists() and read(concat_receipt)['inputs']==concat_inputs and read(concat_receipt)['video']==record(target)):
        subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(target)],check=True)
        atomic_json(concat_receipt,dict(inputs=concat_inputs,video=record(target)))
    outputs.append(target);outputs.extend(pairs)
    outputs.append(planner_demo(out,videos/'PLANNER_REPAIR_DEMONSTRATION.mp4'))
    outputs.append(candidate_demo(out,videos/'CANDIDATE_SELECTION_DEMONSTRATION.mp4'))
    inspection=[inspect_frames(path,videos) for path in outputs]
    report=out/'VIDEO_VERIFICATION.json'
    atomic_json(report,dict(status='AUTOMATED_PASS_VISUAL_PENDING',videos=inspection,
        contact_sheets=[record(p) for p in sheets],solid_measured_geometry=True,
        true_horizon_from_physical_receipts=True,diagnostic_videos_excluded_from_physical_results=True))
    atomic_text(out/'VIDEO_VERIFICATION.md','# Repaired converter video verification\n\n'
        'Every available complete command has a solid measured-PhysX replay through its actual recorded horizon. '
        'No-plan tiles are labelled NO_COMPLETE_COMMAND; shorter completed recordings end in explicit end cards. '
        'Planner and candidate videos are diagnostic fixtures, excluded from measured results. '
        'ffprobe and beginning/middle/end decode evidence are in VIDEO_VERIFICATION.json. '
        'Visual inspection is required in VISUAL_INSPECTION.json before readiness.\n')
    review=out/'VISUAL_INSPECTION.json'
    if not review.exists():raise RuntimeError('VISUAL_INSPECTION_REQUIRED: inspect saved beginning/middle/end frames before final gate')
    accepted=read(review)
    if not accepted.get('pass') or any(record(v['path'])!=v for v in accepted.get('videos',[])) or len(accepted.get('videos',[]))!=len(outputs):
        raise RuntimeError('VISUAL_INSPECTION_REQUIRED: current video hashes not all inspected')
    value=read(report);value['status']='PASS';value['visual_inspection']=record(review);atomic_json(report,value)
    return [report,out/'VIDEO_VERIFICATION.md',review,*outputs,*sheets]
