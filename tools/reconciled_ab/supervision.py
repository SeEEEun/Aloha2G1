"""Compare effective ACT examples, not archive or report names."""
import hashlib,time,sys
import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file
from .common import *

def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def chunks(a):
    indices=np.arange(len(a))[:,None]+np.arange(50)[None];mask=indices>=len(a)
    return a[np.minimum(indices,len(a)-1)],mask
def prepare_old():
    out=[];protocol=read(ROOT/'outputs/paper_core_ab/act_a_b_training_contract.json');lineage=read(ROOT/'outputs/paper_core_ab/train40_manifest.json')['entries']
    for mode,letter in [('WRIST','a'),('INTERACTION','b')]:
        dataset=Path(protocol['records'][letter]['dataset_root']);selection=read(ROOT/f'outputs/paper_core_ab/offline_heldout8/{letter}/selected_checkpoint_result.json')['checkpoint_selection'];checkpoint=Path(selection['selected_checkpoint'])
        norm=load_file(checkpoint/'policy_preprocessor_step_3_normalizer_processor.safetensors')
        pack=read(dataset/'meta/g1_packaging_manifest.json');table=pq.read_table(sorted((dataset/'data').rglob('*.parquet')))
        ep=np.array(table['episode_index']);a=np.asarray(table['action'].to_pylist(),np.float32);s=np.asarray(table['observation.state'].to_pylist(),np.float32);ts=np.array(table['timestamp']);rows=[]
        for entry in pack['episode_mapping']:
            e=entry['output_episode_index'];mask=ep==e;oa=a[mask];os=s[mask];ot=ts[mask];meta=next(x for x in lineage if x['original_source_recording_id']==entry['original_source_recording_id']);z=np.load(meta[f'{letter}_trajectory_path'])
            names=read(dataset/'meta/g1_training_contract.json')['joint_order'];oldnames=z['replay_joint_names'].astype(str).tolist();q=z['replay_named_joint_qpos'][:,[oldnames.index(n) for n in names]]
            assert np.array_equal(q,oa),'Packaged action differs from archived named trajectory'
            expected=np.vstack((oa[:1],oa[:-1]));assert np.array_equal(os,expected)
            na=(oa-norm['action.mean'])/(norm['action.std']+1e-8);ns=(os-norm['observation.state.mean'])/(norm['observation.state.std']+1e-8);chunk,pad=chunks(na)
            rows.append(dict(final_dataset_index=entry['final_dataset_index'],source_recording_id=entry['original_source_recording_id'],original_episode_index=meta['original_episode_index'],stable_episode_id=entry['stable_episode_id'],frames=len(oa),action_hash=ah(oa),state_hash=ah(os),timestamps_hash=ah(ot),normalized_action_hash=ah(na),normalized_state_hash=ah(ns),effective_action_chunk_hash=ah(chunk),padding_mask_hash=ah(pad),masked_chunk_scalars=int(pad.sum()*28),archived_actions_exact=True,own_previous_action_state_exact=True,source_rgb=meta['source_rgb_identity']))
        out.append(dict(representation=mode,dataset=str(dataset),checkpoint=selection,model=record(checkpoint/'model.safetensors'),processor=record(checkpoint/'policy_preprocessor.json'),normalizer=record(checkpoint/'policy_preprocessor_step_3_normalizer_processor.safetensors'),training_config=record(checkpoint/'train_config.json'),episodes=rows,consumed_inputs=['observation.images.cam_high','observation.state'],unused_task_metadata=True))
    save(RUN/'act/OLD_EFFECTIVE_EXAMPLES.json',out)
    save(RUN/'act/SOURCE_TASK_ONLY_LEARNING_CONVENTION.json',dict(status='PREDECLARED_BEFORE_TRAINING',state='same own-trajectory G1 previous-action surrogate definition for both, not equal numeric values',preparation='0.700s deterministic execution prefix outside learned source task examples; no fabricated RGB frames; common first predicted task q defines preparation endpoint and must pass physical validity',observations='exact original source RGB at original source timestamps',inference='source-observation-conditioned open-loop action generation, not autonomous target-camera control',normalization='same official per-dataset MEAN_STD procedure; values may differ legitimately',chunk='50 source-task action frames, clamped end frames plus action_is_pad; no synthetic prefix examples'))
    print('OLD_EFFECTIVE_EXAMPLES',sum(len(x['episodes']) for x in out),flush=True)

def finalize():
    if not (RUN/'act/OLD_EFFECTIVE_EXAMPLES.json').exists():prepare_old()
    cases=[x for x in read(PRIOR/'CASE_MANIFEST.json') if x['group']=='TRAIN40']
    while not all((RUN/'construction'/x['key']/'RESULT.json').exists() for x in cases):time.sleep(15)
    results={x['key']:read(RUN/'construction'/x['key']/'RESULT.json') for x in cases}
    if any(x['outcome']=='INFRASTRUCTURE_INVALID' for x in results.values()):raise RuntimeError('Do not turn unresolved construction errors into training exclusions')
    ids=sorted({x['index'] for x in cases});membership=[];paired=[]
    for index in ids:
        a=results[f'TRAIN40_WRIST_{index:03d}'];b=results[f'TRAIN40_INTERACTION_{index:03d}'];include=a['selected'] is not None and b['selected'] is not None
        if include:paired.append(index)
        membership.append(dict(index=index,recording=a['case']['source_recording_id'],included=include,A=a['outcome'],B=b['outcome'],reason='PAIRED_EXECUTABLE' if include else 'MISSING_PAIRED_EXECUTABLE_SUPERVISION; no physical-success selection'))
    save(RUN/'act/PAIRED_TRAIN_SET_MANIFEST.json',dict(denominator=40,included_ids=paired,entries=membership,construction_protocol=record(RUN/'freeze/CONSTRUCTION_PROTOCOL.json')))
    diffs=[]
    for old in read(RUN/'act/OLD_EFFECTIVE_EXAMPLES.json'):
        mode=old['representation'];dataset=Path(old['dataset']);table=pq.read_table(sorted((dataset/'data').rglob('*.parquet')));ep=np.array(table['episode_index']);actions=np.asarray(table['action'].to_pylist(),np.float32);states=np.asarray(table['observation.state'].to_pylist(),np.float32);mapping=read(dataset/'meta/g1_packaging_manifest.json')['episode_mapping'];norm=load_file(old['normalizer']['path'])
        for entry in mapping:
            idx=entry['final_dataset_index'];result=results[f'TRAIN40_{mode}_{idx:03d}'];oa=actions[ep==entry['output_episode_index']];os=states[ep==entry['output_episode_index']]
            d=dict(method=mode,index=idx,recording=entry['original_source_recording_id'],paired=idx in paired,old_frames=len(oa))
            if result['selected']:
                with np.load(result['selected']['trajectory']['path']) as z:ca=z['commanded_q_rad'][21:].astype(np.float32)
                assert ca.shape==oa.shape;cs=np.vstack((ca[:1],ca[:-1]));delta=abs(ca.astype(float)-oa);cd=ca!=oa
                oc,mask=chunks((oa-norm['action.mean'])/(norm['action.std']+1e-8));nc,nmask=chunks((ca-norm['action.mean'])/(norm['action.std']+1e-8));assert np.array_equal(mask,nmask)
                d.update(status='NUMERICALLY_COMPARABLE_SOURCE_TASK_EXAMPLES',changed_frames=int(cd.any(axis=1).sum()),changed_scalars=int(cd.sum()),max_abs_diff=float(delta.max()),mean_abs_diff=float(delta.mean()),state_changed_scalars=int((cs!=os).sum()),normalized_chunk_changed_scalars=int((nc!=oc).sum()),chunk_mask_changed=False,old_normalizer_used_for_diff_only=True,source_observation_timestamps_changed=False,added_fake_observations=0,execution_prefix_seconds=.7,new_action_hash=ah(ca),new_state_hash=ah(cs))
            else:d.update(status='NO_VALID_NEW_SUPERVISION',changed_frames=None,changed_scalars=None,max_abs_diff=None,mean_abs_diff=None,reason='Missing valid target is not a zero-valued action or an invented numerical diff')
            diffs.append(d)
    save(RUN/'act/EFFECTIVE_SUPERVISION_DIFF.json',dict(paired_ids=paired,rows=diffs,retraining_required=bool(len(paired)!=40 or any(x.get('changed_scalars',0) for x in diffs)),old_state_difference_not_automatically_confound=True,normalization_refit_required=True,checkpoint_reuse_compatible=False))
    csvsave(RUN/'act/EFFECTIVE_SUPERVISION_DIFF.csv',diffs)
    text(RUN/'EFFECTIVE_ACT_SUPERVISION_DIFF.md','# Effective ACT supervision diff\n\nActual archived actions, own previous-action state, checkpoint normalizers, 50-frame action chunks and masks were inspected. The checkpoint selection is A100000 andB20000 under the same historical heldout8 rule. Both used source RGB and a method-consistent generated-G1 state surrogate. Different state values alone are not an unintended confound; the previous paper-run report\'s contrary interpretation is not retained.\n\nThe new common learning convention retains original source RGB/timestamps and uses the0.700s preparation outside learned examples; no fabricated observation interval is created. Actual action/state/chunk numerical differences and missing supervision are recorded separately. Serialization-only changes do not trigger retraining.\n\nPaired executable TRAIN membership: '+str(len(paired))+'/40. Every exclusion is disclosed. Old checkpoints are not compatible with the changed effective supervision/membership. '+('ACT is unavailable because the paired executable set is empty; no target or policy result is fabricated.' if not paired else 'Both policies require the same retraining protocol on the paired subset.')+'\n')
    save(RUN/'act/BRANCH_STATUS.json',dict(status='UNAVAILABLE_EMPTY_PAIRED_TRAIN_SET' if not paired else 'RETRAIN_REQUIRED',paired_ids=paired,retraining_required=True,checkpoints_reused=False,training_started=False,physical_rollouts=0))
    print('EFFECTIVE_ACT_DECISION','paired',len(paired),flush=True)

if __name__=='__main__':prepare_old() if len(sys.argv)>1 and sys.argv[1]=='old' else finalize()
