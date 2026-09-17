"""Compatibility cards and deterministic source/reference/measured comparisons."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,cv2
from .common import *
from tools import render_final_episode_registered_physical_evidence as legacy
from tools.final_paper_position_run import model,RESET
legacy.COMMON=RESET/'config/common_config.json'

def main():
    fig,axs=plt.subplots(1,4,figsize=(15,4.4),layout='constrained')
    cards=[('OLD_TRAINING','40 source episodes each\nActual checkpoint supervision\n\nOwn raw/projected target contract\nNo current natural-start prefix\n\nSaved FK independently reproduced\nCurrent-target verdict: NOT COMPARABLE'),('INTERMEDIATE','Selected TRAIN11 portfolio\nA7/11 · B10/11\n\nSame raw targets as current\nMultiple common recovery stages\nSelected q differs from new batch\n\nDiagnostic history; not frozen outputs'),('CURRENT_FROZEN','Uniform original position batch\nTRAIN A0/40 · B19/40\nDEV A0/35 · B18/35\n\nA6D NOT RUN UPSTREAM\nB37 complete6D returns rejected\nNo DEV35 physical rollouts'),('RECONCILED','Accuracy ≠ execution validity\nUnchanged weighted6D numeric core\nKnown-answer kinematic tests PASS\n\nBest physically valid common candidate\nActual PhysX only for valid commands\nUnknown ≠ observed failure')]
    for ax,(title,body) in zip(axs,cards):
        ax.axis('off');ax.set_title(title,fontsize=12,color='#285374');ax.text(.02,.92,body,va='top',fontsize=10,linespacing=1.7)
    fig.suptitle('Artifact-family reconciliation — separate contracts, not one performance curve',fontsize=14)
    fig.savefig(RUN/'figures/Fig_Artifact_Family_Reconciliation.png',dpi=200);plt.close(fig)
    directories=[RUN/'train_pilot/TRAIN40_INTERACTION_010']
    for mode in ('WRIST','INTERACTION'):
        rows=[read(p) for p in sorted((RUN/'reference_physics').glob('*/RESULT.json')) if read(p)['case']['representation_mode']==mode]
        candidates=[r for r in rows if r.get('physical_trace')]
        if candidates:directories.append(Path(min(candidates,key=lambda r:r['case']['index'])['physical_trace']['path']).parent)
    renderer=legacy.PhysicalRenderer();renderer.model.vis.headlight.ambient[:]=(.25,.25,.25);renderer.model.vis.headlight.diffuse[:]=(.5,.5,.5);g,c,n=model();records=[]
    try:
        for folder in directories:
            inv=read(folder/'INVOCATION_MANIFEST.json');row=inv['case'];tr=legacy.control_trace(folder)
            with np.load(inv['commands']['path']) as z:
                q=z['q'] if 'q' in z else z['commanded_q_rad'][21:,:14];raw=z['RAW_TARGET'] if 'RAW_TARGET' in z else z['RAW_REPRESENTATION_TARGET'];planned=z['FK_OF_SOLVER_Q'] if 'FK_OF_SOLVER_Q' in z else z['EXECUTABLE_FK_POSITION'];events=dict(zip(z['source_event_names'].astype(str),z['execution_event_frames'].astype(int)))
            source=ROOT/'raw_recordings'/row['source_recording_id']/'images/observation.images.cam_high/episode_000000';images=sorted(source.glob('frame_*.png'));assert len(images)==len(q)
            frames=[events['LEFT_CLOSE_COMPLETE'],events['RIGHT_ACQUIRE_SOURCE'],len(q)+20]
            fig,axs=plt.subplots(3,3,figsize=(12,9),layout='constrained');source_records=[]
            for i,f in enumerate(frames):
                sf=min(max(f-21,0),len(q)-1);j=min(f,len(tr['frame'])-1)
                im=cv2.cvtColor(cv2.imread(str(images[sf])),cv2.COLOR_BGR2RGB);axs[i,0].imshow(im);axs[i,0].set_title(f'SOURCE observation / task frame{sf}');axs[i,0].axis('off');source_records.append(record(images[sf]))
                # Position plots have explicit target/reference/measured labels;
                # no command-only doll motion is presented as physics evidence.
                tm=np.arange(len(q))/30;measured=[]
                for v in tr['q'][21:]:measured.append(g.wrist_state(v[:14])['left_position'][2])
                ax=axs[i,1];ax.plot(tm,raw[:,0,2],label='Raw target z',lw=1);ax.plot(tm,planned[:,0,2],label='Reference FK z',lw=1);ax.plot(np.arange(len(measured))/30,measured,label='Measured FK z',lw=1);ax.axvline(sf/30,color='gray',ls=':');ax.set_xlabel('Source task time (s)');ax.set_ylabel('Model wrist z (m)');ax.legend(fontsize=7)
                renderer.set_state(tr['q'][j],tr['position'][j],tr['quaternion'][j]);axs[i,2].imshow(cv2.cvtColor(renderer.view('overview'),cv2.COLOR_BGR2RGB));axs[i,2].set_title(f'MEASURED PhysX / global frame{f}');axs[i,2].axis('off')
            fig.suptitle(row['key']+' — real source, reference fidelity, and measured motion\nDeterministic events; no outcome-selected target or physical-state edits',fontsize=12)
            p=RUN/f"figures/SOURCE_REFERENCE_MEASURED_{row['key']}.png";fig.savefig(p,dpi=160);plt.close(fig)
            records.append(dict(case=row,figure=record(p),physical_trace=record(folder/'event_log.npz'),source_images=source_records,frames=[int(x) for x in frames],selection='Predeclared TRAIN pilot plus lowest executed DEV index for each method, independent of success'))
    finally:renderer.close()
    save(RUN/'analysis/SOURCE_REFERENCE_MEASURED_MANIFEST.json',records)
    print('SOURCE_REFERENCE_MEASURED_COMPARISONS',len(records),flush=True)

if __name__=='__main__':main()
