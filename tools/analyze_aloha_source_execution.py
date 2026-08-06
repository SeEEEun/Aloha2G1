#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
import numpy as np
from aloha_source_validation_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--trial-dir',type=Path,required=True);a=p.parse_args();c=load_record(a.trial_dir/'command_action.npz');act=load_record(a.trial_dir/'actual_state.npz');cmd=c['command_action_14d'];actual=act['mapped_observation_state_14d'];m=tracking_metrics(cmd,actual);tm=np.asarray(act['timestamp_monotonic']);dt=np.diff(tm);m.update({'control_interval_mean_s':float(dt.mean()),'control_interval_std_s':float(dt.std()),'control_interval_max_s':float(dt.max()),'dropped_or_stale_packets':int(np.sum(dt>.1)),'stop_reason':json.loads((a.trial_dir/'trial_metadata.json').read_text()).get('stop_reason'),'fk_metrics':'NOT_COMPUTED_NO_VERIFIED_COMMAND_ACTUAL_FK_ALIGNMENT'})
 out=a.trial_dir/'analysis';dump(out/'metrics.json',m)
 with (out/'per_joint_metrics.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['joint','rmse']);w.writerows(zip(NAMES,m['per_joint_rmse']))
 t=np.arange(min(len(cmd),len(actual)))/30
 for fn,data,title in [('tracking_plot.png',(cmd,actual),'Command and actual'),('tracking_error_plot.png',(actual[:len(t)]-cmd[:len(t)],),'Tracking error'),('gripper_timing_plot.png',(cmd[:len(t),[6,13]],actual[:len(t),[6,13]]),'Gripper timing'),('control_timing_plot.png',(dt,),'Control timing')]:
  fig,ax=plt.subplots(figsize=(12,5));[ax.plot(x) for x in data];ax.set_title(title);fig.savefig(out/fn,dpi=140);plt.close(fig)
 (out/'report.md').write_text('# ALOHA source execution analysis\n\n'+json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
if __name__=='__main__':main()
