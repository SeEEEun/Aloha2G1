#!/usr/bin/env python3
"""Gate-controlled offline scene-registered retargeting runner."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
from task_frame_registration import build,REG,SCENE_OUT,ACTION,ARM,sha
APPROVALS={'registration':(REG/'g1_scene_registration.approved.json','G1_SCENE_REGISTRATION_APPROVED'),'semantic_frames':(REG/'semantic_frames.approved.json','OBJECT_SEMANTIC_FRAMES_APPROVED'),'aloha_tool_axes':(REG/'aloha_tool_axes.approved.json','ALOHA_TOOL_AXES_APPROVED'),'timeline':(ROOT/'configs/episode49_task_timeline.approved.json','EPISODE49_TIMELINE_APPROVED')}
WEIGHTS=[0,.00025,.0005,.001,.002,.004,.008]
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def graph_png(graph):
 import matplotlib.pyplot as plt
 frames=graph['frames'];levels={None:0,'fixed_scene_world':1,'g1_base':2,'g1_left_wrist_yaw':3,'g1_right_wrist_yaw':3,'aloha_stationary_world':1}
 fig,ax=plt.subplots(figsize=(15,9));bucket={}
 for x in frames:bucket.setdefault(levels.get(x['parent_frame'],2),[]).append(x)
 pos={}
 for level,xs in bucket.items():
  for i,x in enumerate(xs):pos[x['frame_name']]=(level,(i-(len(xs)-1)/2)*1.1)
 for x in frames:
  a=pos[x['frame_name']];ax.scatter(*a,c='tab:green' if 'verified' in x['status'] else 'tab:orange');ax.text(a[0],a[1],x['frame_name'],fontsize=8)
  if x['parent_frame'] in pos:
   b=pos[x['parent_frame']];ax.plot([b[0],a[0]],[b[1],a[1]],'k-',lw=.6)
 ax.set_title('Coordinate frame graph (orange = inferred/dynamic/unknown)');ax.axis('off');REG.mkdir(parents=True,exist_ok=True);fig.savefig(REG/'frame_graph.png',dpi=170,bbox_inches='tight');plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--inspect',action='store_true');p.add_argument('--diagnostic',action='store_true',help='Generate a visibly reviewable SIMULATION-ONLY non-final candidate while preserving unresolved gates.');p.add_argument('--iterations',type=int,default=8);p.add_argument('--force-unapproved',action='store_true',help='Not permitted; retained to make refusal explicit.');a=p.parse_args()
 if a.force_unapproved:raise SystemExit('REFUSED: unapproved registration/semantic frames/tool axes/timeline cannot be bypassed.')
 graph,registration,semantic,tool=build();graph_png(graph)
 gates={k:{'path':str(v[0]),'required_flag':v[1],'approved':v[0].exists() and json.load(open(v[0])).get('status')==v[1]} for k,v in APPROVALS.items()};all_ok=all(x['approved'] for x in gates.values())
 freeze={'source_action':str(ACTION),'source_action_sha256':sha(ACTION),'authoritative_arm':str(ARM),'authoritative_arm_sha256':sha(ARM)};dump(REG/'input_hashes.json',freeze)
 approval={'status':'WAITING_FOR_USER_APPROVAL' if not all_ok else 'ALL_MANUAL_APPROVALS_PRESENT','flags':gates,'downstream_generation_blocked':not all_ok};dump(REG/'approval_status.json',approval)
 if a.diagnostic:
  cmd=[sys.executable,str(ROOT/'tools/generate_scene_registered_ep49_diagnostic.py'),'--iterations',str(a.iterations)]
  return subprocess.run(cmd,check=False).returncode
 if not all_ok:
  verdict={'selected_candidate':None,'verdict':'WAITING_FOR_USER_APPROVAL','failure_classification':[k.upper()+'_NOT_APPROVED' for k,v in gates.items() if not v['approved']],'manual_review':'INCOMPLETE','downstream_generation':'BLOCKED','real_g1_safety':'NOT_PERFORMED'};dump(SCENE_OUT/'selection.json',verdict)
  html=f'''<!doctype html><meta charset="utf-8"><title>Manual approval</title><style>body{{font:16px sans-serif;max-width:1100px;margin:2em auto}}pre{{background:#eee;padding:1em}}.bad{{color:#b00;font-weight:bold}}</style><h1>WAITING FOR USER APPROVAL</h1><p class="bad">NO TRAJECTORY CANDIDATE WAS GENERATED</p><img src="../../../task_frame_registration/approval_views/g1_scene_front.png" style="max-width:100%"><pre>{json.dumps(gates,indent=2)}</pre><pre>{json.dumps(verdict,indent=2)}</pre>''';(SCENE_OUT/'report').mkdir(parents=True,exist_ok=True);(SCENE_OUT/'report/index.html').write_text(html);print('REFUSED: four manual approvals are required before downstream generation.');print(json.dumps(verdict,indent=2));return 3
 anchor={'candidate':'scene_registered_relative_position','status':'READY_TO_COMPUTE','formula':'p_G(t)=global_scene_anchor + 0.42 R delta_p_A(t)','global_anchor_invariant_over_frames':True,'stage_specific_offsets':False,'source_action_hash':freeze['source_action_sha256']};dump(SCENE_OUT/'scene_aware_anchor.json',anchor)
 cont={'weights':WEIGHTS,'strictly_monotonic':bool(np.all(np.diff(WEIGHTS)>0)),'warm_start':'previous feasible weight result','start':'position_only_frozen','stop_rule':'stop at first weight failing position/structural gate','results':[],'status':'READY'};dump(SCENE_OUT/'continuation_sweep.json',cont)
 names=('position_only_frozen','scene_registered_position_only','scene_registered_neutral_wrist','scene_registered_phase_partial_orientation','scene_registered_phase_partial_orientation_with_nominal_elbow')
 candidates=[]
 for n in names:
  status='REFERENCE_EXACT_PRESERVED' if n=='position_only_frozen' else ('READY' if all_ok else 'NOT_GENERATED_UNAPPROVED_CALIBRATION')
  x={'candidate':n,'status':status,'source_action_hash':freeze['source_action_sha256'],'scale':.42,'frame_count':990,'fps':30.0,'registration_shared':True,'object_poses_unchanged':True};candidates.append(x);dump(SCENE_OUT/'candidates'/n/'status.json',x)
 automatic={'status':'NOT_RUN' if not all_ok else 'PENDING_GENERATION','approval_gates':gates,'structural':'NOT_AVAILABLE','position':'NOT_AVAILABLE','orientation':'NOT_AVAILABLE','visual':'NOT_AVAILABLE','collision':'NOT_AVAILABLE','real_g1_safety':'NOT_PERFORMED'};dump(SCENE_OUT/'automatic_gate.json',automatic)
 verdict={'selected_candidate':None,'verdict':'NO SCENE-REGISTERED TASK-VALID CANDIDATE','failure_classification':[k.upper()+'_NOT_APPROVED' for k,v in gates.items() if not v['approved']],'manual_review':'INCOMPLETE','real_g1_safety':'NOT_PERFORMED'};dump(SCENE_OUT/'selection.json',verdict)
 dump(SCENE_OUT/'report/candidate_metrics.json',{'status':'NOT_AVAILABLE_BEFORE_APPROVAL','candidates':candidates})
 visual={'front_video':'NOT_AVAILABLE','top_video':'NOT_AVAILABLE','phase_montage':'NOT_AVAILABLE','reason':'No candidate may be generated/rendered before all explicit approvals.'};dump(SCENE_OUT/'report/visual_outputs.json',visual)
 html=f'''<!doctype html><meta charset="utf-8"><title>Scene-registered retargeting</title><style>body{{font:16px sans-serif;max-width:1100px;margin:2em auto}}pre{{background:#eee;padding:1em;white-space:pre-wrap}}.bad{{color:#b00;font-weight:bold}}</style><h1>Scene-registered task retargeting</h1><p class="bad">NO REAL G1 / NO REAL ALOHA — REAL G1 SAFETY NOT_PERFORMED</p><h2>Frame graph</h2><img src="../../../task_frame_registration/frame_graph.png" style="max-width:100%"><h2>Registration</h2><pre>{json.dumps(registration,indent=2)}</pre><h2>Approval gates</h2><pre>{json.dumps(gates,indent=2)}</pre><h2>Objects</h2><pre>{json.dumps(semantic,indent=2)}</pre><h2>ALOHA axes</h2><pre>{json.dumps(tool,indent=2)}</pre><h2>Continuation</h2><pre>{json.dumps(cont,indent=2)}</pre><h2>Decision</h2><pre>{json.dumps(verdict,indent=2)}</pre>'''
 (SCENE_OUT/'report').mkdir(parents=True,exist_ok=True);(SCENE_OUT/'report/index.html').write_text(html)
 print('SCENE-REGISTERED TASK RETARGETING\nNO REAL G1 WAS USED\nNO REAL ALOHA WAS USED\nREAL G1 EXECUTION NOT APPROVED UNTIL MANUAL REVIEW');print(json.dumps(verdict,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
