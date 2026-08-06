#!/usr/bin/env python3
from __future__ import annotations
import json
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
import numpy as np
from aloha_source_validation_common import *
def main():
 q,fps,_=load_action();timeline=json.loads((ROOT/'configs/episode49_task_timeline.approved.json').read_text());ev={x['event']:x['frame'] for x in timeline['events']}
 specs=[('A_PHONE_APPROACH',0,ev['left_phone_grasp_start'],'left hand approaches phone'),('B_PHONE_ROTATION',ev['phone_rotation_to_portrait_start'],ev['phone_portrait_reached'],'phone landscape to portrait rotation'),('C_ACCESSORY_REMOVE',max(0,ev['right_accessory_grasp_start']-30),ev['accessory_removed'],'right hand approaches and removes accessory'),('D_CHARGER_MOVE',ev['phone_move_to_charger_start'],ev['phone_charger_attachment_complete'],'left hand moves phone toward charger'),('E_ACCESSORY_PLACE',ev['accessory_removed'],ev['right_accessory_release_complete'],'right hand places accessory')]
 seg=[{'segment_id':i,'start_frame':int(s),'end_frame':int(e),'meaning':m,'evidence':[str(ROOT/'configs/episode49_task_timeline.approved.json'),'USER_CONFIRMED manual video event bounds'],'review_status':'REQUIRES_HUMAN_REVIEW'} for i,s,e,m in specs]
 out=BASE/'segments';dump(out/'segment_candidates.json',{'status':'REQUIRES_HUMAN_REVIEW','segments':seg});dump(out/'segment_manifest.reviewed.json',{'status':'NOT_REVIEWED','segments':seg,'hardware_execution_allowed':False})
 t=np.arange(len(q))/fps
 for fn,overlay in [('trajectory_overview.png',False),('gripper_event_overlay.png',True)]:
  fig,ax=plt.subplots(figsize=(13,5));ax.plot(t,q);[ax.axvspan(s/fps,e/fps,alpha=.08) for _,s,e,_ in specs]
  if overlay:ax.plot(t,q[:,6],lw=2,label='left gripper');ax.plot(t,q[:,13],lw=2,label='right gripper');ax.legend()
  fig.savefig(out/fn,dpi=140);plt.close(fig)
 print(json.dumps({'status':'REQUIRES_HUMAN_REVIEW','segments':seg},indent=2))
if __name__=='__main__':main()
