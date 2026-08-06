#!/usr/bin/env python3
"""Render front/side/top frame-sequence projections for the diagnostic target."""
from pathlib import Path
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R=Path('/home/jbnu/aloha_g1_dataset'); O=R/'outputs/scene_registered_retargeting/current_layout_ep49'
with np.load(O/'scene_registered_targets.npz',allow_pickle=False) as z:
 l=z['g1_target_left_position_scene']; r=z['g1_target_right_position_scene']
events=json.loads((O/'task_event_metrics.json').read_text())['events']
frames=[v['frame'] for v in events.values() if 'frame' in v]
layout=json.loads((R/'isaaclab_magsafe_fixed_scene/scene_layout.json').read_text())
objects={'phone':(.525,.07,.83075),'accessory':(.525,.076425,.83075),'charger':(.42,.21,.807)}
views={'front':(0,2,'scene X (m)','scene Z (m)'),'side':(1,2,'scene Y (m)','scene Z (m)'),'top':(0,1,'scene X (m)','scene Y (m)')}
out=O/'report/frame_sequences';out.mkdir(parents=True,exist_ok=True)
for name,(a,b,xlab,ylab) in views.items():
 fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
 for ax,f in zip(axes.flat,frames):
  ax.plot(l[:,a],l[:,b],color='tab:blue',alpha=.25);ax.plot(r[:,a],r[:,b],color='tab:orange',alpha=.25)
  ax.scatter(l[f,a],l[f,b],c='tab:blue',label='left wrist');ax.scatter(r[f,a],r[f,b],c='tab:orange',label='right wrist')
  for obj,p in objects.items():ax.scatter(p[a],p[b],marker='x',s=45,label=obj)
  ax.set_title(f'frame {f}');ax.grid(alpha=.25)
 for ax in axes[-1]:ax.set_xlabel(xlab)
 for ax in axes[:,0]:ax.set_ylabel(ylab)
 axes.flat[0].legend(fontsize=7,ncol=2);fig.suptitle(f'{name} diagnostic target projection — not an Isaac camera render')
 fig.tight_layout();fig.savefig(out/f'{name}_event_frame_sequence.png',dpi=160);plt.close(fig)
print(out)
