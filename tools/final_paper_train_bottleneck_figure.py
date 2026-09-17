#!/usr/bin/env python3
"""A TRAIN-only diagnostic of distinct raw, position and6D failure layers."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_io import *

def main():
    ids=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')['selection']['qualification_ids'];eligible=[]
    for index in sorted(ids):
        a=read(DEST/'position'/f'TRAIN40_WRIST_{index:03d}'/'RESULT.json');b=read(DEST/'position'/f'TRAIN40_INTERACTION_{index:03d}'/'RESULT.json');six=read(DEST/'full6d'/f'TRAIN40_INTERACTION_{index:03d}'/'RESULT.json')
        if a['outcome']!='POSITION_EXECUTABLE' and b['outcome']=='POSITION_EXECUTABLE' and six['outcome'].startswith('FULL6D_') and six['outcome']!='FULL6D_EXECUTABLE':eligible.append((index,a,b,six))
    assert eligible,'No eligible predeclared TRAIN diagnostic; do not fabricate a case'
    index,a,b,six=eligible[0];data=[];geometry=[]
    for r in (a,b):
        with np.load(r['selected']['trajectory']['path']) as z:
            data.append(dict(time=z['source_timestamp']-z['source_timestamp'][0],res=z['POSITION_CORRECTION_MM'].max(axis=1),bound=z['CERTIFIED_LOWER_BOUND_MM'].max(axis=1)))
        rows=read(r['selected']['geometry']['path'])['source'];classes=[]
        for row in rows:
            c={v['classification'] for v in row['records']};classes.append(2 if 'HARD_SELF_COLLISION' in c else 3 if 'UNRESOLVED_GEOMETRY' in c else 1 if 'PROXY_ONLY_OVERLAP' in c else 0)
        geometry.append(classes)
    assert np.array_equal(data[0]['time'],data[1]['time'])
    with np.load(six['trajectory']['path']) as z:post=z['POSITION_CORRECTION_MM'].max(axis=1);angle=z['ORIENTATION_RESIDUAL_RAD'].max(axis=1)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'axes.spines.top':False})
    fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained',gridspec_kw={'height_ratios':[2,1,2]})
    for d,label,color in zip(data,('A — Wrist','B — Interaction'),('#cc6b42','#277f9d')):
        axes[0].plot(d['time'],d['res'],label=label+' position attempt',color=color)
        axes[0].plot(d['time'],d['bound'],ls=':',lw=1.5,color=color,label=label+' certified raw lower bound')
    axes[0].axhline(10,color='black',ls='--',lw=1,label='10 mm raw-fidelity gate');axes[0].legend(fontsize=8,ncol=2)
    axes[0].set(ylabel='Worse-wrist residual (mm)',title='(a) Raw morphology gap and achieved position residual');axes[0].grid(alpha=.2)
    cmap=ListedColormap(['#e7ebed','#b9cce2','#b64c4c','#888888'])
    axes[1].imshow(geometry,aspect='auto',interpolation='nearest',extent=[0,data[0]['time'][-1],1.5,-.5],cmap=cmap,vmin=0,vmax=3)
    axes[1].set_yticks([0,1],['A','B']);axes[1].set_title('(b) Detailed geometry on position candidates (not physical execution)')
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor=cmap(k),label=v) for k,v in enumerate(['Clear','Proxy only','Hard collision','Unresolved'])],ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15))
    t=data[1]['time'];axes[2].plot(t,data[1]['res'],color='#277f9d',label='B qualified position');axes[2].plot(t,post,color='#9a5987',label='B after frozen 6D solve');axes[2].axhline(10,color='black',ls='--',lw=1)
    axes[2].set(xlabel='Source-task time (s); common preparation excluded',ylabel='Cartesian residual (mm)',title='(c) 6D final acceptance is separate from orientation fitting');axes[2].legend(fontsize=9,loc='upper left');axes[2].grid(alpha=.2)
    twin=axes[2].twinx();twin.plot(t,angle,color='#777777',lw=.7,alpha=.6);twin.set_ylabel('Orientation residual (rad)',color='#777777');twin.set_ylim(0,max(.75,float(angle.max())*1.1))
    fig.suptitle(f'Frozen common solver diagnostic — TRAIN source index {index}\nNot a physical rollout; no individual repair',fontsize=14)
    folder=OUT/'07_paper_artifacts/final';folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    for ext in ('png','pdf'):
        p=folder/f'Fig_TRAIN_Frozen_Solver_Bottleneck.{ext}';fig.savefig(p,dpi=250,bbox_inches='tight');paths.append(file_record(p))
    plt.close(fig)
    atomic_json(folder/'TRAIN_BOTTLENECK_DIAGNOSTIC_MANIFEST.json',dict(source_index=index,scope='TRAIN_DIAGNOSTIC_NOT_PHYSICS',selection_rule='Lowest source index in the already predeclared TRAIN11 subset with A position failure, B position pass and B6D failure; no visual choice or DEV outcome selection',inputs=[a['selected']['trajectory'],b['selected']['trajectory'],six['trajectory']],outcomes=[a['outcome'],b['outcome'],six['outcome']],artifacts=paths,interpretation='These are saved candidate/qualified kinematics. A lower bound is not a constrained optimum, and a failed search is not a global impossibility proof.'))
    print('TRAIN_ONLY_DIAGNOSTIC_FIGURE_SAVED',index)

if __name__=='__main__':main()
