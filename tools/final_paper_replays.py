#!/usr/bin/env python3
"""Measured PhysX replay or explicit static failure card, never commanded motion."""
from pathlib import Path
import json,os,subprocess,sys,textwrap
os.environ.setdefault('MUJOCO_GL','egl')
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_reference_physics import DEST,OUT,RESET,read,atomic_json,atomic_text,file_record
from tools import render_final_episode_registered_physical_evidence as legacy
legacy.COMMON=RESET/'config/common_config.json'
PAPER=OUT/'07_paper_artifacts/final';VIDEOS=PAPER/'replays'
W,H=3840,2160;CW,CH=540,405;GX,GY=30,67

def card(row,letter,index):
    image=np.full((CH,CW,3),(234,235,238),np.uint8)
    cv2.rectangle(image,(0,0),(CW,42),(42,45,52),-1)
    cv2.putText(image,f'REFERENCE {letter} | DEV {index+1:02d}/35',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.67,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image,'RETARGETING FAILURE',(20,120),cv2.FONT_HERSHEY_SIMPLEX,.75,(45,55,165),2,cv2.LINE_AA)
    reason=row.get('failure_class',row['first_failure_stage'])
    for j,line in enumerate(textwrap.wrap(reason.replace('_',' '),31)):
        cv2.putText(image,line,(20,170+30*j),cv2.FONT_HERSHEY_SIMPLEX,.55,(35,35,40),1,cv2.LINE_AA)
    cv2.putText(image,'NO PHYSICAL ROLLOUT FABRICATED',(20,300),cv2.FONT_HERSHEY_SIMPLEX,.48,(60,65,75),1,cv2.LINE_AA)
    cv2.putText(image,row['case']['source_recording_id'][:40],(20,365),cv2.FONT_HERSHEY_SIMPLEX,.40,(70,70,75),1,cv2.LINE_AA)
    return image

def canvas(letter,view):
    v=np.full((H,W,3),248,np.uint8)
    cv2.putText(v,f'REFERENCE {letter} | DEV35 DEVELOPMENT EVALUATION | {view.upper()}',(35,43),cv2.FONT_HERSHEY_SIMPLEX,1.,(35,35,35),2,cv2.LINE_AA)
    cv2.putText(v,'Measured robot + PhysX object states where executed; otherwise explicit retargeting-failure cards',(35,2135),cv2.FONT_HERSHEY_SIMPLEX,.7,(55,55,55),1,cv2.LINE_AA)
    return v

def main():
    PAPER.mkdir(parents=True,exist_ok=True);VIDEOS.mkdir(parents=True,exist_ok=True)
    result=read(PAPER/'FINAL_NUMERICAL_RESULTS.json');methods=result['reference_episodes'];products={};renderer=None;representatives=[]
    actual=sum(bool(r.get('physical_trace')) for rows in methods.values() for r in rows)
    global_frames=0
    for rows in methods.values():
        for row in rows:
            with np.load(row['case']['source']['path']) as z:global_frames=max(global_frames,len(z['source_timestamp'])+21)
            if row.get('physical_trace'):
                trace=legacy.control_trace(Path(row['physical_trace']['path']).parent)
                global_frames=max(global_frames,len(trace['frame']))
    if actual:renderer=legacy.PhysicalRenderer()
    try:
        for mode,letter in [('WRIST','A'),('INTERACTION','B')]:
            rows=methods[mode];traces=[];cards=[];max_frames=global_frames
            for i,row in enumerate(rows):
                if row.get('physical_trace'):traces.append(legacy.control_trace(Path(row['physical_trace']['path']).parent));cards.append(None)
                else:traces.append(None);cards.append(card(row,letter,i))
                with np.load(row['case']['source']['path']) as z:max_frames=max(max_frames,len(z['source_timestamp'])+21)
            for view in ('top','overview'):
                path=VIDEOS/f'REFERENCE_{letter}_DEV35_{view.upper()}_35SPLIT.mp4'
                if path.exists():
                    products[f'{letter}_{view}']=dict(artifact=file_record(path),probe=legacy.probe(path),actual_trace_count=sum(t is not None for t in traces),failure_card_count=sum(t is None for t in traces),measured_states_only=True)
                    continue
                if not any(t is not None for t in traces):
                    board=canvas(letter,view)
                    for i,img in enumerate(cards):
                        y,x=divmod(i,7);board[GY+y*CH:GY+(y+1)*CH,GX+x*CW:GX+(x+1)*CW]=img
                    png=VIDEOS/f'{letter}_{view}_RETARGETING_FAILURE_CARDS.png';legacy.atomic_image(png,board)
                    tmp=path.with_name(path.stem+'.incomplete.mp4')
                    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-i',str(png),'-t',str(max_frames/30),'-r','30','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',str(tmp)],check=True)
                    os.replace(tmp,path)
                else:
                    writer=legacy.Writer(path)
                    for frame in range(max_frames):
                        board=canvas(letter,view)
                        for i,(row,trace) in enumerate(zip(rows,traces)):
                            if trace is None:img=cards[i]
                            else:
                                j=min(frame,len(trace['frame'])-1);renderer.set_state(trace['q'][j],trace['position'][j],trace['quaternion'][j]);img=renderer.view(view)
                                cv2.rectangle(img,(0,0),(CW,38),(35,35,40),-1)
                                label=f'{letter} DEV{i+1:02d}: '+('SUCCESS' if row['outcomes']['FULL_TASK_SUCCESS'] else row['first_failure_stage'])
                                cv2.putText(img,label[:56],(8,25),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
                                if frame>=len(trace['frame']):cv2.putText(img,'TRACE ENDED',(12,CH-16),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
                            y,x=divmod(i,7);board[GY+y*CH:GY+(y+1)*CH,GX+x*CW:GX+(x+1)*CW]=img
                        writer.write(board)
                        if frame%100==0:print('PHYSICS_REPLAY',letter,view,frame,max_frames,flush=True)
                    writer.finish()
                probe=legacy.probe(path);assert probe['width']==W and probe['height']==H and probe['codec_name']=='h264' and probe['avg_frame_rate']=='30/1'
                products[f'{letter}_{view}']=dict(artifact=file_record(path),probe=probe,actual_trace_count=sum(t is not None for t in traces),failure_card_count=sum(t is None for t in traces),measured_states_only=True)
        selected=read(PAPER/'OBJECTIVE_REPRESENTATIVE_SELECTION.json');indices=set(selected['categories'].values());gaps=[]
        for i in range(35):
            means=[]
            for mode in ('WRIST','INTERACTION'):
                p=read(DEST/'position'/methods[mode][i]['case']['key']/'RESULT.json')
                with np.load(p['selected']['trajectory']['path']) as z:means.append(float(z['CERTIFIED_LOWER_BOUND_MM'].max(axis=1).mean()))
            gaps.append(means[0]-means[1])
        largest=min(range(35),key=lambda i:(-abs(gaps[i]),i));indices.add(largest);selected.update(largest_raw_morphology_gap_index=largest,largest_gap_rule='Maximum absolute difference of per-episode mean certified raw residual lower bounds; lowest-index tie break',gap_mm=gaps[largest]);atomic_json(PAPER/'OBJECTIVE_REPRESENTATIVE_SELECTION.json',selected)
        images=[];labels=[]
        for i in sorted(indices):
            for mode,letter in [('WRIST','A'),('INTERACTION','B')]:
                row=methods[mode][i]
                if row.get('physical_trace'):
                    trace=legacy.control_trace(Path(row['physical_trace']['path']).parent)
                    # Fixed normalized midpoint, not a visually chosen frame.
                    j=len(trace['frame'])//2;renderer.set_state(trace['q'][j],trace['position'][j],trace['quaternion'][j]);image=renderer.view('overview')
                else:image=card(row,letter,i)
                images.append(image);labels.append(f'DEV{i+1:02d} {letter} | '+(row['first_failure_stage'] if row.get('physical_trace') else 'RETARGETING FAILURE'))
        contact=legacy.composite(images,2,labels);legacy.atomic_image(PAPER/'Fig_Representative_Physical_Rollouts.png',contact)
        atomic_json(PAPER/'REPLAY_MANIFEST.json',dict(products=products,implementation=file_record(Path(__file__)),renderer=file_record(Path(legacy.__file__)),physical_traces=actual,common_video_frames=global_frames,representative_selection=selected,
            interpretation='Failure cards are not simulations. Actual trajectories are rendered only from measured robot and PhysX object states. No command-only object animation.'))
    finally:
        if renderer:renderer.close()
    print('REFERENCE_REPLAYS_VERIFIED',list(products),flush=True)

if __name__=='__main__':main()
