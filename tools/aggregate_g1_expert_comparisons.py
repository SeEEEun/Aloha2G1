#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
def flatten(d,p=''):
 out={}
 for k,v in d.items():
  key=f'{p}.{k}' if p else k
  if isinstance(v,dict):out.update(flatten(v,key))
  elif isinstance(v,(int,float)) and v is not None:out[key]=float(v)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--comparisons',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=20260803);p.add_argument('--bootstrap-samples',type=int,default=2000);a=p.parse_args();rows=[];excluded=[]
 for path in a.comparisons:
  try:
   d=json.loads((path/'comparison_metrics.json').read_text());expert=Path(d['expert'])
   side=expert.with_suffix('.metadata.json');em=json.loads(side.read_text()) if side.exists() else {}
   group=em.get('success_label','synthetic' if em.get('is_synthetic') else 'unlabeled')
   r={'trial':expert.stem,'alignment':path.name,'group':group,'paper_valid':d['paper_valid']};r.update(flatten(d['metrics']));rows.append(r)
  except Exception as e:excluded.append({'trial':path.name,'reason':str(e)})
 keys=sorted(set().union(*(r.keys() for r in rows))-{'trial','alignment','group','paper_valid'});rng=np.random.default_rng(a.seed);agg=[]
 for group in sorted({r['group'] for r in rows}):
  for k in keys:
   x=np.array([r[k] for r in rows if r['group']==group and k in r]);
   if not len(x):continue
   bs=np.mean(rng.choice(x,(a.bootstrap_samples,len(x)),replace=True),axis=1);agg.append({'group':group,'metric':k,'mean':x.mean(),'std':x.std(),'median':np.median(x),'iqr':np.percentile(x,75)-np.percentile(x,25),'min':x.min(),'max':x.max(),'ci95_low':np.percentile(bs,2.5),'ci95_high':np.percentile(bs,97.5),'n':len(x)})
 a.output_dir.mkdir(parents=True,exist_ok=True)
 for name,data in (('per_trial_metrics.csv',rows),('aggregate_metrics.csv',agg),('exclusion_log.csv',excluded)):
  with (a.output_dir/name).open('w',newline='') as f:
   fields=sorted(set().union(*(x.keys() for x in data))) if data else ['trial','reason'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
 (a.output_dir/'aggregate_summary.json').write_text(json.dumps({'seed':a.seed,'nearest_expert':'DIAGNOSTIC_ONLY_NOT_SELECTED_AS_PRIMARY','trial_count':len(rows),'excluded':excluded,'aggregate':agg},indent=2)+'\n');return 0
if __name__=='__main__':raise SystemExit(main())
