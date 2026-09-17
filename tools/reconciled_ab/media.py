"""Actual measured-state mosaics; nonexecution summaries are explicitly named."""
import os,sys,time,subprocess,textwrap
os.environ.setdefault('MUJOCO_GL','egl')
import cv2,numpy as np
from .common import *
from tools import render_final_episode_registered_physical_evidence as legacy
legacy.COMMON=ROOT/'outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4/config/common_config.json'
W,H=3840,2160;CW,CH=540,405;GX,GY=30,67

def card(row,index,letter):
    im=np.full((CH,CW,3),230,np.uint8);cv2.rectangle(im,(0,0),(CW,38),(45,45,48),-1);cv2.putText(im,f'{letter} DEV {index+1:02d}/35',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(im,'NO PHYSICAL EXECUTION',(20,110),cv2.FONT_HERSHEY_SIMPLEX,.7,(40,60,120),2,cv2.LINE_AA)
    reason=row.get('outcome','NOT_AVAILABLE')
    for k,line in enumerate(textwrap.wrap(reason.replace('_',' '),35)):cv2.putText(im,line,(20,160+26*k),cv2.FONT_HERSHEY_SIMPLEX,.5,(35,35,40),1,cv2.LINE_AA)
    cp=RUN/'construction'/row['case']['key']/'RESULT.json'
    if cp.exists():
        c=read(cp);best=min(c.get('candidates',[]),key=lambda x:x['tracking_cost'],default=None)
        if best:
            for k,line in enumerate(textwrap.wrap('Gates: '+', '.join(best['layer_b_gates']),40)):cv2.putText(im,line,(20,250+24*k),cv2.FONT_HERSHEY_SIMPLEX,.43,(55,55,60),1,cv2.LINE_AA)
    cv2.putText(im,'Observed task outcome: NOT MEASURED',(20,345),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,45),1,cv2.LINE_AA)
    cv2.putText(im,row['case']['source_recording_id'][:42],(20,378),cv2.FONT_HERSHEY_SIMPLEX,.4,(60,60,65),1,cv2.LINE_AA);return im

def main():
    while not (RUN/'REFERENCE_BATCH_COMPLETE.json').exists():time.sleep(15)
    rows=[read(p) for p in sorted((RUN/'reference_physics').glob('*/RESULT.json'))];assert len(rows)==70
    traces={r['case']['key']:legacy.control_trace(Path(r['physical_trace']['path']).parent) for r in rows if r.get('physical_trace')}
    frames=max(len(np.load(r['case']['source']['path'])['source_timestamp'])+21 for r in rows)
    renderer=legacy.PhysicalRenderer() if traces else None
    if renderer:
        renderer.model.vis.headlight.ambient[:]=(.25,.25,.25);renderer.model.vis.headlight.diffuse[:]=(.5,.5,.5)
    products={};representatives=[]
    try:
        for mode,letter in [('WRIST','A'),('INTERACTION','B')]:
            method=sorted([r for r in rows if r['case']['representation_mode']==mode],key=lambda r:r['case']['index']);assert len(method)==35
            actual=sum(r['case']['key'] in traces for r in method);cards={r['case']['key']:card(r,r['case']['index'],letter) for r in method if r['case']['key'] not in traces}
            for view in ('top','overview'):
                kind='PHYSICAL_ROLLOUT_VIDEO' if actual else 'NONEXECUTION_SUMMARY';path=RUN/f'replays/REFERENCE_{letter}_DEV35_{view.upper()}_35SPLIT_{kind}.mp4'
                if path.exists():products[f'{letter}_{view}']=dict(artifact=record(path),probe=legacy.probe(path),kind=kind,actual_trace_count=actual,card_count=35-actual);continue
                def canvas(f):
                    board=np.full((H,W,3),245,np.uint8);cv2.putText(board,f'REFERENCE {letter} | DEV35 DEVELOPMENT EVALUATION | {view.upper()} | {kind}',(30,42),cv2.FONT_HERSHEY_SIMPLEX,.9,(30,30,30),2,cv2.LINE_AA)
                    cv2.putText(board,'Saved measured robot + PhysX doll states only. Nonexecuted cases are cards; invalid physics remains labeled unknown.',(30,2135),cv2.FONT_HERSHEY_SIMPLEX,.7,(40,40,40),1,cv2.LINE_AA)
                    for i,r in enumerate(method):
                        key=r['case']['key'];tr=traces.get(key)
                        if tr is None:im=cards[key]
                        else:
                            j=min(f,len(tr['frame'])-1);renderer.set_state(tr['q'][j],tr['position'][j],tr['quaternion'][j]);im=renderer.view(view)
                            cv2.rectangle(im,(0,0),(CW,40),(35,35,40),-1);label=f'{letter} DEV{i+1:02d} '+r['first_failure_stage'];cv2.putText(im,label[:55],(8,25),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
                            cv2.putText(im,f'MEASURED | object z={tr["position"][j,2]:.3f}m',(8,CH-12),cv2.FONT_HERSHEY_SIMPLEX,.44,(25,25,25),1,cv2.LINE_AA)
                            if not r.get('physical_valid'):cv2.putText(im,'INVALID PHYSICS / OUTCOME UNKNOWN',(8,62),cv2.FONT_HERSHEY_SIMPLEX,.4,(40,40,180),1,cv2.LINE_AA)
                            if f>=len(tr['frame']):cv2.putText(im,'TRACE ENDED',(8,82),cv2.FONT_HERSHEY_SIMPLEX,.4,(25,25,25),1,cv2.LINE_AA)
                        y,x=divmod(i,7);board[GY+y*CH:GY+(y+1)*CH,GX+x*CW:GX+(x+1)*CW]=im
                    return board
                if not actual:
                    png=path.with_suffix('.png');legacy.atomic_image(png,canvas(0));tmp=path.with_name(path.stem+'.incomplete.mp4')
                    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-i',str(png),'-t',str(frames/30),'-r','30','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',str(tmp)],check=True);os.replace(tmp,path)
                else:
                    writer=legacy.Writer(path)
                    for f in range(frames):
                        board=canvas(f);writer.write(board)
                        if f in (0,frames//2,frames-1):legacy.atomic_image(path.with_name(path.stem+f'_frame{f:04d}.png'),board)
                        if f%150==0:print('MEASURED_REPLAY_RENDER',letter,view,f,frames,flush=True)
                    writer.finish()
                products[f'{letter}_{view}']=dict(artifact=record(path),probe=legacy.probe(path),kind=kind,actual_trace_count=actual,card_count=35-actual)
    finally:
        if renderer:renderer.close()
    for r in products.values():
        p=r['probe'];assert p['codec_name']=='h264' and p['width']==3840 and p['height']==2160 and p['avg_frame_rate']=='30/1'
    save(RUN/'REFERENCE_MEDIA_COMPLETE.json',dict(products=products,physical_fields=['MEASURED_Q','object_position_world_m','object_quaternion_xyzw'],camera_and_order_identical=True,visualization_only='MuJoCo named-model rendering of saved PhysX states, no new dynamics or object command following',physics_protocol=record(RUN/'freeze/PHYSICS_PROTOCOL.json')))
    print('REFERENCE_MEDIA_COMPLETE',flush=True)

if __name__=='__main__':main()
