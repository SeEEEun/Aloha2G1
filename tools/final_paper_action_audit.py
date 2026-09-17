#!/usr/bin/env python3
"""Exact old supervision audit and paired executable membership, never impute failures."""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_reference_physics import *
AUDIT=DEST/'action_dataset_audit'

def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def statistics(x):return dict(mean=float(x.mean()),p95=float(np.quantile(x,.95)),max=float(x.max())) if x.size else None

def main():
    # Do not train or characterize absent outputs as valid supervision.
    rows=[r for r in cases() if r['group']=='TRAIN40'];by={(r['representation_mode'],r['index']):r for r in rows}
    while not all((DEST/'complete_action'/r['key']/'RESULT.json').exists() for r in rows):time.sleep(15)
    ids=sorted({r['index'] for r in rows});membership=[];paired=[]
    for index in ids:
        outcomes={mode:read(DEST/'complete_action'/by[mode,index]['key']/'RESULT.json')['outcome'] for mode in ('WRIST','INTERACTION')}
        if any(v=='INFRASTRUCTURE_INVALID' for v in outcomes.values()):raise RuntimeError('Repair infrastructure before deriving paired TRAIN membership')
        include=all(v=='COMPLETE_ACTION_EXECUTABLE' for v in outcomes.values())
        if include:paired.append(index)
        stage_outcomes={mode:{stage:read(DEST/stage/by[mode,index]['key']/'RESULT.json')['outcome'] for stage in ('position','full6d','complete_action')} for mode in ('WRIST','INTERACTION')}
        membership.append(dict(final_dataset_index=index,source_recording_id=by['WRIST',index]['source_recording_id'],included=include,reason='BOTH_COMPLETE_EXECUTABLE' if include else stage_outcomes))
    manifest=dict(status='PAIRED_EXECUTABLE_TRAIN_SET_FROZEN',source_split=file_record(SPLIT),included_ids=paired,entries=membership,denominator=40,selection='Intersection of complete A/B executable position+6D targets; no physical-success selection')
    atomic_json(AUDIT/'PAIRED_TRAIN_SET_MANIFEST.json',manifest)
    results={}
    protocol=read(ROOT/'outputs/paper_core_ab/act_a_b_training_contract.json')
    for mode,letter in [('WRIST','a'),('INTERACTION','b')]:
        old=Path(protocol['records'][letter]['dataset_root']);pack=read(old/'meta/g1_packaging_manifest.json');mapping={r['final_dataset_index']:r for r in pack['episode_mapping']}
        parquet=sorted((old/'data').rglob('*.parquet'));table=pq.read_table(parquet,columns=['episode_index','frame_index','timestamp','observation.state','action']);episode=np.array(table['episode_index']);actions=np.array(table['action'].to_pylist(),dtype=np.float32);states=np.array(table['observation.state'].to_pylist(),dtype=np.float32);times=np.array(table['timestamp']);reports=[]
        for index in ids:
            oldrow=mapping[index];mask=episode==oldrow['output_episode_index'];oa=actions[mask];os=states[mask];ot=times[mask]
            record=dict(final_dataset_index=index,old_frames=len(oa),old_action_sha256=array_hash(oa),old_state_sha256=array_hash(os),old_timestamp_sha256=array_hash(ot),included_in_paired_set=index in paired)
            if index not in paired:
                record.update(status='EXCLUDED_FROM_BOTH_CORRECTED_DATASETS',removed_frames=len(oa),removed_action_scalars=int(oa.size),overlap_numerical_diff=None,reason='No paired complete executable target; invalid candidate never becomes supervision')
            else:
                row=by[mode,index];six=read(DEST/'full6d'/row['key']/'RESULT.json');cp,_=command(row,six['trajectory']['path'])
                with np.load(cp) as z:ca=z['commanded_q_rad'].astype(np.float32)
                assert len(ca)==len(oa)+21
                # Align by preserved source timestamps, report prefix separately.
                delta=abs(ca[21:].astype(float)-oa.astype(float));changed=delta!=0
                with np.load(row['source']['path']) as z:source_state=z['source_state'].astype(np.float32)
                # Same source state for both policies, padded only to retain the
                # authoritative ACT 28-input architecture. Never call this G1 q.
                cs=np.pad(np.vstack((np.tile(source_state[0],(21,1)),source_state)),((0,0),(0,14)))
                record.update(status='COMPLETE_NEW_TARGET',new_frames=len(ca),added_preparation_frames=21,
                    changed_source_frames=int(changed.any(axis=1).sum()),changed_source_action_scalars=int(changed.sum()),
                    maximum_absolute_difference=float(delta.max()),mean_absolute_difference=float(delta.mean()),
                    new_action_sha256=array_hash(ca),new_state_sha256=array_hash(cs),
                    source_state_change='COMMON_SOURCE_ALOHA14_PLUS_ZERO_PADDING14 replaces method-derived previous-action G1 surrogate',
                    state_source_frames_changed=int(np.any(cs[21:]!=os,axis=1).sum()),target=file_record(cp),
                    action_chunk_change='21-frame prefix changes 50-frame chunks; same causal padding/chunk rule for both policies')
                atomic_npz(AUDIT/'corrected_targets'/f'{mode}_{index:03d}.npz',action=ca,observation_state=cs)
            reports.append(record)
        results[mode]=dict(old_dataset=str(old),old_dataset_records=[file_record(p) for p in parquet]+[file_record(old/'meta/stats.json'),file_record(old/'meta/g1_training_contract.json')],episodes=reports,
            removed_episodes=sum(not r['included_in_paired_set'] for r in reports),paired_episodes=len(paired),
            old_state_contract=read(old/'meta/g1_training_contract.json'),
            actions_changed=True,retraining_required=True,
            reason='Paired membership differs and/or explicit21-frame preparation changes exact supervision, timing/chunks/state; no metadata-only reuse')
    result=dict(retraining_required=True,old_checkpoints_reusable=False,methods=results,paired_train_episodes=len(paired),
        paired_manifest=file_record(AUDIT/'PAIRED_TRAIN_SET_MANIFEST.json'),
        normalization='Must refit by identical official procedure on paired data; old stats cannot be reused',
        common_observation_contract='Original source RGB and identical source14 state zero-padded to28; padding retains ACT architecture, does not invent measured G1 state',
        old_state_confounds='Old observation.state equals each representation\'s previous action, so it is not identical source-state conditioning. New pair uses identical source conditioning; this common schema change requires both policies to be retrained.',
        checkpoint_selection='Fixed final100000-step checkpoint, identical seed1000, no DEV35 validation or outcome selection; unchanged100000-step training budget',
        training_protocol=file_record(ROOT/'outputs/paper_core_ab/act_a_b_training_contract.json'))
    atomic_json(AUDIT/'EXACT_ACTION_DATASET_AUDIT.json',result)
    atomic_text(AUDIT/'EXACT_ACTION_DATASET_AUDIT.md','# Exact old/current supervision audit\n\n'+json.dumps(result,indent=2)+'\n')
    if not paired:
        atomic_json(DEST/'ACT_BRANCH_RESULT.json',dict(status='UNAVAILABLE_EMPTY_PAIRED_EXECUTABLE_TRAIN_SET',retraining_required=True,datasets_regenerated=False,act_retrained=False,paired_train_episodes=0,
            reason='No source episode has complete executable supervision for both methods under the frozen pipeline. No invalid actions or unequal training membership are substituted.',audit=file_record(AUDIT/'EXACT_ACTION_DATASET_AUDIT.json'),act_physical_results_available=False))
    print('ACTION_AUDIT_COMPLETE','paired',len(paired),'retraining required YES',flush=True)

if __name__=='__main__':main()
