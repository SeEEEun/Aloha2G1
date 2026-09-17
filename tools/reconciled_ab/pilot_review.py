"""Measured TRAIN pilot contact sheet, clearly distinct from reference targets."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import cv2,numpy as np
from .common import *
from tools import render_final_episode_registered_physical_evidence as render
render.COMMON=ROOT/'outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4/config/common_config.json'

def main():
    folder=RUN/'train_pilot/TRAIN40_INTERACTION_010';inv=read(folder/'INVOCATION_MANIFEST.json');trace=render.control_trace(folder)
    with np.load(inv['commands']['path']) as z:clock=dict(zip(z['source_event_names'].astype(str),z['execution_event_frames'].astype(int)))
    selected=[0,clock['LEFT_CLOSE_COMPLETE'],clock['LEFT_LIFT_BEGIN'],clock['RIGHT_ACQUIRE_SOURCE']]
    renderer=render.PhysicalRenderer();board=np.full((850,2160,3),245,np.uint8)
    try:
        for col,f in enumerate(selected):
            j=min(f,len(trace['frame'])-1);renderer.set_state(trace['q'][j],trace['position'][j],trace['quaternion'][j])
            for row,view in enumerate(('top','overview')):
                im=renderer.view(view);cv2.putText(im,f'MEASURED TRAIN B10 / frame {f}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA);board[30+405*row:30+405*(row+1),540*col:540*(col+1)]=im
    finally:renderer.close()
    cv2.putText(board,'TRAIN COMPONENT PILOT: actual measured robot + PhysX doll state; no acquisition observed',(20,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
    path=RUN/'figures/TRAIN_PHYSICAL_PILOT_MEASURED_CONTACT_SHEET.png';render.atomic_image(path,board)
    save(RUN/'audit/PILOT_REPLAY_PROVENANCE.json',dict(image=record(path),trace=record(folder/'event_log.npz'),frames=[int(x) for x in selected],model=record(renderer.g1.path),frame_selection='Natural frame0, source close-complete, lift-begin, receiver-acquire; not outcome selected',measured_fields=['MEASURED_Q','object_position_world_m','object_quaternion_xyzw']))
    print(path,flush=True)

if __name__=='__main__':main()
