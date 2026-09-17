"""Real single-trace serialization diagnostic, explicitly excluded from training."""
from pathlib import Path
import hashlib
import numpy as np
from .io import read, record, atomic_json
from .target_dataset import CAM, checked_episode
from .target_dataset_audit import numeric_alignment


def run(out):
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig

    area = out / 'dataset_interface_diagnostic'
    milestone = read(out / 'FIRST_SOURCE_CONDITIONED_FULL_TASK.json')
    row = {k: milestone[k] for k in ('source_id', 'alignment', 'score')}
    alignment, observed, success = checked_episode(row)
    contract = dict(purpose='DATASET_INTERFACE_DIAGNOSTIC_NOT_TRAINING',
                    source=row, implementation=record(__file__),
                    matched_membership=False, policy_training_permitted=False)
    path = area / 'CONTRACT.json'
    if path.exists() and read(path) != contract:
        raise ValueError('Diagnostic contract changed; preserve the old realization')
    atomic_json(path, contract)
    receipt = area / 'RESULT.json'
    if receipt.exists():
        result = read(receipt)
        for dep in result['dataset_files']:
            if record(dep['path']) != dep:
                raise ValueError('Diagnostic dataset changed')
        return result
    names = observed['joint_names'].tolist()
    features = {k: dict(dtype='float32', shape=(28,), names=names)
                for k in ('action', 'observation.state')}
    features[CAM] = dict(dtype='video', shape=(480, 640, 3), names=['height', 'width', 'channels'])
    root = area / 'single_development_trace'
    kwargs = dict(repo_id='local/hybrid_interface_diagnostic', root=root,
                  video_backend='torchcodec', image_writer_threads=2, encoder_threads=1,
                  rgb_encoder=RGBEncoderConfig(vcodec='h264', crf=18, preset='veryfast'))
    dataset = (LeRobotDataset.resume(**kwargs) if (root / 'meta/info.json').exists()
               else LeRobotDataset.create(fps=30, features=features, robot_type='g1_dex3_dynamic',
                                          metadata_buffer_size=1, **kwargs))
    try:
        if dataset.meta.total_episodes == 0:
            for state, action, image in zip(observed['measured_state'], observed['executed_command'],
                                           alignment['images'], strict=True):
                if record(image['path']) != image:
                    raise ValueError('Changed real target observation')
                with Image.open(image['path']) as im:
                    rgb = np.asarray(im.convert('RGB'))
                dataset.add_frame({'task': 'Pick up the doll with the left hand, pass it to the right hand, and place it in the bin.',
                                   'action': action.astype(np.float32),
                                   'observation.state': state.astype(np.float32), CAM: rgb})
            dataset.save_episode(parallel_encoding=False)
        elif dataset.meta.total_episodes != 1:
            raise ValueError('Unexpected diagnostic episode count')
    finally:
        dataset.finalize()
    dataset = LeRobotDataset('local/hybrid_interface_diagnostic', root=root, video_backend='torchcodec',
                             delta_timestamps={'action': [i / 30 for i in range(50)]})
    raw = dataset.select_columns(['action', 'observation.state', 'timestamp', 'frame_index'])[:]
    n = len(observed['measured_state'])
    np.testing.assert_array_equal(raw['frame_index'], np.arange(n))
    numeric_hashes = numeric_alignment(raw, observed)
    decoded = []
    for frame in (0, n // 2, n - 1):
        item = dataset[frame]
        rgb = item[CAM].numpy()
        if rgb.shape != (3, 480, 640) or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('Invalid decoded ACT image input')
        indices = np.minimum(frame + np.arange(50), n - 1)
        np.testing.assert_array_equal(item['action'].numpy(), observed['executed_command'].astype(np.float32)[indices])
        np.testing.assert_array_equal(item['action_is_pad'].numpy(), frame + np.arange(50) >= n)
        np.testing.assert_array_equal(item['observation.state'].numpy(), observed['measured_state'][frame].astype(np.float32))
        decoded.append(dict(frame=frame, rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            padded_actions=int(item['action_is_pad'].sum())))
    result = dict(status='REAL_G1_TRACE_DATASET_INTERFACE_VERIFIED', contract=record(path),
                  frames=n, source_id=row['source_id'], development_full_task=success,
                  training_episodes=0, paired_training_sources=0, ACT_policy_rollouts=0,
                  numeric_hashes=numeric_hashes, decoded_examples=decoded,
                  dataset_files=[record(p) for p in sorted(root.rglob('*')) if p.is_file()],
                  interpretation='Only serialization, target RGB decoding, float32 state/action alignment and installed 50-action chunk/mask loading. This single development B trace is excluded from frozen TRAIN40 supervision and cannot form a matched A/B training dataset.')
    atomic_json(receipt, result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    print(run(parser.parse_args().run_dir)['status'])
