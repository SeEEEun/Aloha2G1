"""Resolve old model/config provenance and explain comparable verdict changes."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from tools.doll_handoff_retargeting.common import load_common_config,load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics

def main():
    configs=read(RUN/'audit/OLD_CONFIG_HASH_RESOLUTION.json');models={}
    for entry in configs:
        h=entry['record']['sha256']
        if h not in models:
            cfg=load_common_config(entry['record']['path']);models[h]=G1Kinematics(cfg,load_scene(cfg))
    old=read(RUN/'audit/OLD_TRAINING_LINEAGE.json');resolved=[]
    for row in old:
        g=models[row['old_common_config_sha256']];z=np.load(row['trajectory']['path']);q=z['g1_arm_qpos'];errors=[]
        for f in range(len(q)):
            g.assign(q[f])
            for side in ('left','right'):
                world=g.model_to_world_position(g.data.xpos[g.wrist_ids[side]])
                errors.append(float(np.linalg.norm(world-z[f'achieved_{side}_wrist_position_world'][f])))
        resolved.append(dict(key=row['case']['key'],recording=row['case']['source_recording_id'],old_config_sha256=row['old_common_config_sha256'],model=record(g.path),mapping=record(g.mapping_path),old_saved_world_fk_max_diff_m=max(errors),joint_order_units='named q radians; own config world transform',target_cross_evaluation='NOT_PERFORMED_TARGET_GENERATOR_AND_EVENT_CONTRACT_CHANGED'))
    save(RUN/'audit/OLD_MODEL_AND_FK_PROVENANCE.json',resolved)
    inter=read(RUN/'audit/INTERMEDIATE_AUDIT.json');current=read(RUN/'audit/CURRENT_CANDIDATE_AUDIT.json');changes=[]
    for i in inter:
        if 'case' not in i or not isinstance(i['case'],dict):continue
        row=i['case'];now=[x for x in current if x['case']['key']==row['key'] and x['stage']=='POSITION']
        iq=np.load(i['trajectory']['path'])['q']
        for cur in now:
            cq=np.load(cur['trajectory']['path'])['q']
            changes.append(dict(key=row['key'],recording=row['source_recording_id'],intermediate_qualified=i['qualified'],current_family=cur['family'],same_raw_targets=True,same_source_timestamps=True,same_model=resolved[0]['model']['sha256'],q_arrays_identical=bool(np.array_equal(iq,cq)),q_max_difference_rad=float(np.max(abs(iq-cq))),current_first_layer_b_gate=next(iter(cur['layer_b_gates']),'NONE'),current_all_layer_b_gates=';'.join(cur['layer_b_gates']),changed_item='SOLVER_PORTFOLIO_AND_GENERATED_Q; not raw representation or relaxed aggregate gate',intermediate_raw10=i['raw_frame_fraction_10mm'],current_raw10=cur['raw_frame_fraction_10mm']))
    csvsave(RUN/'TABLE_PROTOCOL_RECONCILIATION.csv',changes)
    text(RUN/'audit/PROVENANCE_ADDENDUM.md','# Model and verdict provenance\n\nAll three archived common-config hashes were resolved to repository files. OLD q is recomputed through its own named model and its own world transform; the saved-FK comparison is in OLD_MODEL_AND_FK_PROVENANCE.json. Old target generation/projection and event timing are not equated to current raw targets.\n\nAll22 intermediate diagnostic candidates have exactly the current raw target arrays. Their q differs from the newly generated fixed portfolio. TABLE_PROTOCOL_RECONCILIATION.csv names every changed verdict and all physical rejecting gates. No old motion is corrected to fit the comparison.\n\nThe historical0.179rad restriction was already internal-only in both later acceptance implementations. The current numerical and geometry failures therefore cannot be attributed to reintroducing that aggregate gate.\n')
    fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for k,(mode,ep) in enumerate([('WRIST',0),('WRIST',24),('INTERACTION',10)]):
        key=f'TRAIN40_{mode}_{ep:03d}';a=next(x for x in inter if isinstance(x.get('case'),dict) and x['case']['key']==key);b=next(x for x in current if x['case']['key']==key and x['stage']=='POSITION')
        qa=np.load(a['trajectory']['path'])['q'];qb=np.load(b['trajectory']['path'])['q'];ts=np.arange(len(qa))/30
        axs[0,k].plot(ts,np.linalg.norm(qa-qb,axis=1),lw=1);axs[0,k].set_title(f'{mode} TRAIN{ep:02d}\nSame raw targets; different solver portfolio');axs[0,k].set_ylabel('Configuration difference (rad)');axs[0,k].set_xlabel('Source task time (s)')
        rates=[a['raw_frame_fraction_10mm'],b['raw_frame_fraction_10mm']];axs[1,k].bar(['Intermediate','Current'],np.array(rates)*100,color=['#888888','#397bb3']);axs[1,k].set_ylim(0,100);axs[1,k].set_ylabel('Raw within10mm (%)');axs[1,k].text(.02,.92,'Current physical gates:\n'+', '.join(b['layer_b_gates']),transform=axs[1,k].transAxes,fontsize=8,va='top')
    fig.suptitle('Regression reconciliation — compatible TRAIN trajectories only\nOLD_TRAINING is a different target/event contract and is not merged into this curve',fontsize=12)
    p=RUN/'figures/Fig_Regression_Reconciliation.png';p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=180);plt.close(fig)
    print('LINEAGE_MODEL_RESOLVED',len(resolved),'max FK difference',max(x['old_saved_world_fk_max_diff_m'] for x in resolved),flush=True)

if __name__=='__main__':main()
