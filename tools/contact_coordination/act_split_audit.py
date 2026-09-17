"""Source identity/content audit for the requested disjoint ACT training split."""
from pathlib import Path
import hashlib,csv,io,json
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text

MANIFEST=ROOT/'outputs/final_single_variable_ab/00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json'


def audit(out):
    import pyarrow.parquet as pq
    manifest=read(MANIFEST);entries=manifest['entries'];rows=[];group={}
    previous=ROOT/'outputs/contact_coordination_hybrid/20260907T083735Z'
    selection=read(previous/'bootstrap/SELECTION.json')
    for entry in entries:
        source=entry['source_recording_id'];file=Path(entry['source_parquet']);actual=record(file)
        assert actual['sha256']==entry['source_parquet_sha256'],source+' source content changed'
        table=pq.read_table(file);digest=hashlib.sha256();fields=[]
        for key in ['timestamp','observation.state','action']:
            if key not in table.column_names:continue
            a=np.asarray(table[key].to_pylist(),dtype='<f8');digest.update(key.encode());digest.update(str(a.shape).encode());digest.update(a.tobytes());fields.append(key)
        canonical=digest.hexdigest();group.setdefault(canonical,[]).append(source)
        images=sorted((ROOT/'raw_recordings'/source/'images/observation.images.cam_high/episode_000000').glob('frame_*.png'))
        probes=[record(images[i]) for i in sorted(set([0,len(images)//2,len(images)-1]))] if images else []
        row=dict(source_id=source,original50=entry['original50'],TRAIN40=entry['TRAIN40'],DEV35=entry['DEV35_DIAGNOSTIC35'],original_HELDOUT8=entry['original_HELDOUT8'],
            authorized_training=entry['TRAIN40'],evaluation_index=entry['dev_index'],original_episode_index=entry['original_episode_index'],final_dataset_index=entry['final_dataset_index'],
            source_parquet=str(file),source_sha256=actual['sha256'],canonical_signal_sha256=canonical,frames=len(table),source_image_frames=len(images),
            development_use='PRIOR_DEV35_INSPECTION' if entry['DEV35_DIAGNOSTIC35'] else 'CURRENT_FIXED_TRAIN_PROTOTYPE_OR_PILOT' if source in selection['train_source_ids'] else entry.get('previous_checkpoint_usage','LEGACY_TRAIN_DEVELOPMENT' if entry['TRAIN40'] else 'PREVIOUSLY_EXCLUDED'),
            identity_mapping='RECORDING_ID_FROM_CONTENT_BOUND_MANIFEST; original/final indices retained separately',duplicate_signal_group='')
        rows.append(row);atomic_json(out/'split_evidence'/f'{source}.json',dict(manifest_entry=entry,actual_source=actual,canonical_fields=fields,canonical_signal_sha256=canonical,image_probes=probes))
    duplicates=[v for v in group.values() if len(v)>1]
    for row in rows:row['duplicate_signal_group']=';'.join(group[row['canonical_signal_sha256']]) if len(group[row['canonical_signal_sha256']])>1 else ''
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);atomic_text(out/'SPLIT_AUDIT.csv',stream.getvalue())
    train=sorted(r['source_id'] for r in rows if r['authorized_training']);dev=[r['source_id'] for r in sorted([r for r in rows if r['DEV35']],key=lambda r:r['evaluation_index'])];original={r['source_id'] for r in rows if r['original50']};outside=[r['source_id'] for r in rows if not r['TRAIN40'] and not r['DEV35']]
    assert len(train)==40 and len(dev)==35 and not set(train)&set(dev);assert dev==selection['dev_source_ids']
    raw_ids={p.name for p in (ROOT/'raw_recordings').glob('GoPark_*') if p.is_dir()};assert raw_ids=={r['source_id'] for r in rows}
    result=dict(status='RESEARCHER_SPLIT_DECISION_REQUIRED',requested_training_per_method=50,source_pool_count=len(rows),authorized_training_count=len(train),authorized_training_source_ids=train,evaluation_source_ids=dev,evaluation_count=len(dev),
        train_eval_source_overlap=sorted(set(train)&set(dev)),original50_eval_overlap=sorted(original&set(dev)),original50_eval_overlap_count=len(original&set(dev)),previously_excluded_non_eval_sources=outside,
        maximum_non_eval_recording_count=len(rows)-len(dev),authorized_training_shortfall=50-len(train),absolute_pool_shortfall=50+len(dev)-len(rows),matched_valid_G1_demonstrations_A=0,matched_valid_G1_demonstrations_B=0,
        exact_signal_duplicate_groups=duplicates,duplicate_limit='Canonical timestamp/state/action equality and sampled image hashes are checked; no near-duplicate visual identity claim is made.',
        raw_recording_inventory_matches_manifest=True,content_hashes_verified=True,manifest=record(MANIFEST),prior_selection=record(previous/'bootstrap/SELECTION.json'),fixed_train_prototype_and_pilot=selection['train_source_ids'],
        checkpoint_selection='UNFROZEN; existing heldout8 are all in DEV35 and cannot select new checkpoints for this study',
        ablation_positions=selection['ablation_dev_positions'],ablation_source_ids=selection['ablation_source_ids'],evaluation_status='DEV35_DEVELOPMENT_NOT_UNTOUCHED_TEST',
        smaller_training_study_authorized=False,researcher_decision='Explicitly authorize matched TRAIN40/DEV35, or retain50 pending additional eligible data. Previously excluded4 are not silently promoted; collecting new sources is outside this run.')
    atomic_json(out/'SPLIT_CONTRACT.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r=audit(a.run_dir);print({k:r[k] for k in ['source_pool_count','authorized_training_count','original50_eval_overlap_count','absolute_pool_shortfall','exact_signal_duplicate_groups']})
