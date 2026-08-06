#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));from g1_behavior_schema import load_behavior,now
REQ=['task_start','left_pregrasp_start','left_grasp_start','left_hold_start','left_release_start','right_pregrasp_start','right_grasp_start','right_hold_start','right_release_start','task_end']
def main():
 p=argparse.ArgumentParser();p.add_argument('--behavior',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--manual',type=Path);a=p.parse_args();d,m=load_behavior(a.behavior);events=[]
 for name in REQ:
  if name=='task_start':f=0
  elif name=='task_end':f=len(d['timestamps'])-1
  else:
   side,phase,_=name.split('_',2);idx=np.flatnonzero(d[f'{side}_phase']==phase.upper());f=int(idx[0]) if len(idx) else None
  if f is not None:events.append({'name':name,'frame':f,'timestamp':float(d['timestamps'][f]),'source':'automatic','confidence':1.0 if m['behavior_role']=='generated_target' else .5,'note':'semantic phase' if m['behavior_role']=='generated_target' else 'Dex3/phase-derived candidate; manual confirmation required'})
 if a.manual:
  manual=json.loads(a.manual.read_text())['events'];by={e['name']:e for e in events};by.update({e['name']:dict(e,source='manual') for e in manual});events=sorted(by.values(),key=lambda x:x['frame'])
 out=a.output or a.behavior.with_suffix('.events.json');out.write_text(json.dumps({'behavior':str(a.behavior.resolve()),'created_at':now(),'automatic_is_ground_truth':m['behavior_role']=='generated_target','missing_events':[x for x in REQ if x not in {e['name'] for e in events}],'events':events},indent=2)+'\n');print(out);return 0
if __name__=='__main__':raise SystemExit(main())
