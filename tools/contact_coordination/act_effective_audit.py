"""Audit actual consumed ACT40 tensors, videos, and checkpoint contracts."""
import hashlib
from pathlib import Path
import numpy as np
from .io import ROOT, read, record, atomic_json, atomic_text


def canonical(a):
    a = np.asarray(a, dtype='<f4')
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def run(out):
    import pyarrow.parquet as pq
    methods = {}
    arrays = {}
    for label, suffix in [('A', 'fair_a'), ('B', 'proposed_b')]:
        root = ROOT / f'datasets/doll_handoff_{suffix}_train40'
        packing = read(root / 'meta/g1_packaging_manifest.json')
        contract = read(root / 'meta/g1_training_contract.json')
        pieces = [pq.read_table(p).to_pandas() for p in sorted((root/'data').rglob('*.parquet'))]
        import pandas as pd
        df = pd.concat(pieces).sort_values(['episode_index', 'frame_index'])
        action = np.stack(df['action']).astype(np.float32)
        state = np.stack(df['observation.state']).astype(np.float32)
        lag = np.empty_like(action)
        episode = df['episode_index'].to_numpy()
        for e in np.unique(episode):
            ids = np.flatnonzero(episode == e)
            lag[ids[0]] = action[ids[0]]
            lag[ids[1:]] = action[ids[:-1]]
        videos = []
        for row in packing['source_rgb_assets']:
            actual = record(root / row['output_relative_path'])
            source = record(row['source_path'])
            assert actual['sha256'] == source['sha256'] == row['sha256']
            videos.append(dict(episode=row['output_episode_index'], actual=actual,
                               source=source, same_inode=Path(actual['path']).stat().st_ino == Path(source['path']).stat().st_ino))
        ckpt = ROOT / f'outputs/paper_core_ab/act_{label.lower()}40/train/checkpoints/100000/pretrained_model'
        training = read(ckpt / 'train_config.json')
        assert Path(training['dataset']['root']) == root
        dependencies = [record(p) for p in sorted(ckpt.iterdir()) if p.is_file()]
        sources = [e['original_source_recording_id'] for e in packing['episode_mapping']]
        arrays[label] = dict(action=action, state=state, time=df['timestamp'].to_numpy())
        methods[label] = dict(dataset_root=root, checkpoint=ckpt, checkpoint_files=dependencies,
            training_config=training, dataset_contract=contract,
            episode_count=int(len(np.unique(episode))), frames=len(df), sources=sources,
            canonical_action_sha256=canonical(action), canonical_state_sha256=canonical(state),
            lagged_command_state_exact=np.array_equal(lag, state),
            lagged_command_state_max_error_rad=float(np.max(np.abs(lag-state))),
            videos=videos, timestamps_sha256=canonical(df['timestamp'].to_numpy()),
            compatible_with_requested_study=False,
            incompatibilities=['40 source membership versus requested 50',
                'ALOHA RGB instead of G1 simulation RGB',
                'Lagged target-command surrogate instead of measured pre-action G1 state',
                'Legacy dense targets rather than repaired physically realized commands',
                'No dynamic contact-driven object/state supervision lineage'])
    a, b = methods['A'], methods['B']
    shared = ['seed', 'batch_size', 'steps', 'optimizer', 'scheduler']
    parity = dict(source_membership_equal=a['sources'] == b['sources'],
        row_timestamps_equal=np.array_equal(arrays['A']['time'], arrays['B']['time']),
        canonical_actions_equal=np.array_equal(arrays['A']['action'], arrays['B']['action']),
        canonical_states_equal=np.array_equal(arrays['A']['state'], arrays['B']['state']),
        source_videos_equal=all(x['actual']['sha256'] == y['actual']['sha256'] for x, y in zip(a['videos'], b['videos'], strict=True)),
        training_protocol_fields_equal={k: a['training_config'].get(k) == b['training_config'].get(k) for k in shared})
    result = dict(schema='effective_supervision_audit_v1', methods=methods, old_pair_parity=parity,
        current_matched_dynamic_G1_datasets_available=False, reuse_authorized_for_main_study=False,
        normalization_rule='Each checkpoint owns its saved processors/statistics; values are not exchanged.',
        other_assets=dict(matched51='Different MagSafe task and three-camera interface; excluded.',
            g1visual_B='KINEMATIC_OWNERSHIP_OBJECT_RECONSTRUCTION and surrogate state; no matched A checkpoint.',
            final_paper_builder='Separate newer builder pads ALOHA14 state to28 and repeats source images in startup; not the actual ACT40 dataset.'))
    atomic_json(out/'effective_data_audit/EFFECTIVE_SUPERVISION_AUDIT.json', result)
    atomic_text(out/'EFFECTIVE_SUPERVISION_DIFF.md', '''# Effective supervision compatibility

Both existing ACT40 checkpoints are incompatible with this study. This conclusion uses all saved numerical rows and actual video hashes, not archive byte differences alone.

| Field | Actual existing ACT40 pair | Required experiment |
|---|---|---|
| Source membership | Same 40 sources, 27,577 frames each | Requested 50 eligible disjoint sources each; authorization unresolved |
| RGB | Frozen ALOHA cam_high assets, verified against all 40 video hashes | G1 rendering of actual dynamic measured demonstrations |
| State | Exactly own q_target[t-1], with q_target[0] at row0 | Measured G1 state before the current command |
| Action | Legacy dense absolute 28-joint targets | Executed repaired converter commands with actual timing |
| Object evidence | No dynamic realization in these samples | Contact-driven object traces aligned to observations |
| ACT | Same official architecture, 50-step chunks, 100,000 steps, seed1000 | Same protocol between newly realized datasets, selection fixed before DEV35 |

The canonical state/action tensors differ between A and B, as their actions differ. Shared source-video bytes do not establish compatibility with G1-image deployment. Serialization or unused metadata is not the rejection reason.

The separate newer final_paper_build_act_datasets.py uses padded ALOHA state and repeated source startup images. That builder is also unsuitable, but it did not create the inspected ACT40 tensors. The paired51 policies concern MagSafe, another task. The B-only G1-visual adaptation used kinematic ownership reconstruction and surrogate state, so it does not supply a compatible dynamic A/B pair.

Exact checkpoint/processor hashes, actual tensor hashes, membership, protocol and video evidence: effective_data_audit/EFFECTIVE_SUPERVISION_AUDIT.json. Both policies must be retrained after authorized matched physical datasets exist. No replacement dataset has yet been realized; no new-versus-old numerical equality is claimed for unavailable tensors.
''')
    atomic_text(out/'DATASET_PARITY.md', '# Dataset parity\n\n'+
        'The requested realization contract shares source membership, scenes, cameras, pre-action sampling, named state/action order and training rules. Actual A/B RGB, measured states, commands and durations may differ. Privileged object/source annotations remain metadata.\n\n'+
        'Existing ACT40 pair checks:\n\n'+ '\n'.join('- '+k+': '+str(v) for k,v in parity.items()) +
        '\n\nNew matched dynamic datasets: unavailable. This is an audit of legacy supervision, not a completed conversion.\n')
    return result


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();r=run(a.run_dir);print(r['old_pair_parity'])
