#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));from g1_behavior_schema import load_behavior
def interp_indices(n,m):return np.linspace(0,n-1,m)
def sample(a,x):
 a=np.asarray(a);lo=np.floor(x).astype(int);hi=np.minimum(lo+1,len(a)-1);w=x-lo
 return a[lo]*(1-w)[:,None]+a[hi]*w[:,None] if a.ndim>1 else a[lo]*(1-w)+a[hi]*w
def dtw_path(x,y):
 n,m=len(x),len(y);cost=np.full((n+1,m+1),np.inf);cost[0,0]=0;back=np.zeros((n,m),np.uint8)
 for i in range(n):
  for j in range(m):
   opts=(cost[i,j+1],cost[i+1,j],cost[i,j]);k=int(np.argmin(opts));cost[i+1,j+1]=np.linalg.norm(x[i]-y[j])+opts[k];back[i,j]=k
 i,j=n-1,m-1;path=[]
 while i>=0 and j>=0:
  path.append((i,j));k=back[i,j]
  if k==0:i-=1
  elif k==1:j-=1
  else:i-=1;j-=1
 return np.asarray(path[::-1]),float(cost[n,m])
def phase_pairs(a,b):
 pa=np.char.add(np.char.add(a['left_phase'].astype(str),'|'),a['right_phase'].astype(str));pb=np.char.add(np.char.add(b['left_phase'].astype(str),'|'),b['right_phase'].astype(str));pairs=[];warnings=[]
 order=[]
 for p in pa:
  if p not in order:order.append(p)
 for p in order:
  ia=np.flatnonzero(pa==p);ib=np.flatnonzero(pb==p)
  if not len(ia) or not len(ib):warnings.append(f'missing phase {p}');continue
  count=max(len(ia),len(ib));pairs.extend(zip(np.linspace(ia[0],ia[-1],count),np.linspace(ib[0],ib[-1],count)))
 return np.asarray(pairs),warnings
def align(a,b,method='normalized_time',samples=300,position_weight=1,orientation_weight=.2):
 if method=='raw_time':
  ta=a['timestamps']-a['timestamps'][0];tb=b['timestamps']-b['timestamps'][0];end=min(ta[-1],tb[-1]);t=np.linspace(0,end,max(2,int(end*min(float(a['fps']),float(b['fps'])))));pairs=np.c_[np.interp(t,ta,np.arange(len(ta))),np.interp(t,tb,np.arange(len(tb)))];warn=[]
 elif method=='normalized_time':pairs=np.c_[interp_indices(len(a['timestamps']),samples),interp_indices(len(b['timestamps']),samples)];warn=[]
 elif method=='dtw_hand':
  keys=('left_hand_position','right_hand_position','bimanual_midpoint','bimanual_relative_position');xa=np.concatenate([a[k] for k in keys],1)*position_weight;xb=np.concatenate([b[k] for k in keys],1)*position_weight;pairs,cost=dtw_path(xa,xb);warn=[]
 elif method=='phase_aligned':
  if 'event_names' not in a or 'event_names' not in b:raise ValueError('phase alignment unavailable: missing event annotation')
  pairs,warn=phase_pairs(a,b)
  if not len(pairs):raise ValueError('phase alignment unavailable: no shared phase segments')
 else:raise ValueError(method)
 return {'method':method,'index_pairs':pairs,'warnings':warn,'raw_duration_difference_sec':float(abs(a['timestamps'][-1]-b['timestamps'][-1]))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--generated',type=Path,required=True);p.add_argument('--expert',type=Path,required=True);p.add_argument('--method',choices=('raw_time','normalized_time','dtw_hand','phase_aligned'),default='normalized_time');p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=300);p.add_argument('--position-weight',type=float,default=1);p.add_argument('--orientation-weight',type=float,default=.2);a=p.parse_args();ga,gm=load_behavior(a.generated);ea,em=load_behavior(a.expert)
 if gm.get('left_right_swapped') or em.get('left_right_swapped'):raise RuntimeError('LEFT_RIGHT_SWAP_DETECTED')
 r=align(ga,ea,a.method,a.samples,a.position_weight,a.orientation_weight);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,index_pairs=r['index_pairs'],method=np.array(a.method));a.output.with_suffix('.json').write_text(json.dumps({k:v for k,v in r.items() if k!='index_pairs'},indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='index_pairs'},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
