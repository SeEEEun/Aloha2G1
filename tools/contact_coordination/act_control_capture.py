"""Two bounded, immutable recaptures of the existing standalone controls."""
import os
import subprocess
import time
from .io import ROOT, ISAAC, read, record, atomic_json
from .control_audit import OLD


def run(out, resume=False):
    freeze = read(OLD / 'CURRENT_DEPENDENCIES_d42c796d7e59.json')
    for row in freeze['files']:
        assert record(row['path'])['sha256'] == row['sha256'], row['path']
    freeze['files'] += [record(ROOT / p) for p in (
        'tools/contact_coordination/calibration_capture.py',
        'tools/contact_coordination/physics_capture.py',
        'tools/contact_coordination/io.py', 'tools/reconciled_ab/runtime_geometry.py')]
    results = []
    for side in ('left', 'right'):
        folder = out / 'common_control' / f'{side}_full_state'
        if (folder / 'PROCESS.json').exists():
            if not resume:
                raise FileExistsError(folder)
            for dep in read(folder / 'DEPENDENCIES.json')['files']:
                assert record(dep['path'])['sha256'] == dep['sha256']
            results.append(read(folder / 'PROCESS.json'))
            continue
        if (folder / 'INVOCATION.json').exists():
            raise RuntimeError('Interrupted attempt requires diagnosis, not overwrite: ' + str(folder))
        args = list(read(OLD / f'{side}_d42c796d7e59/INVOCATION.json')['command'])
        args[:2] = [ISAAC, str(ROOT / 'tools/contact_coordination/calibration_capture.py')]
        args[args.index('--output-dir') + 1] = str(folder)
        args[args.index('--direct-freeze-manifest') + 1] = str(folder / 'DEPENDENCIES.json')
        # Use the authoritative current 150 mm bin; standalone controls do not touch it.
        args += ['--bin-height-m', '0.150', '--bin-rim-bevel-m', '0.003']
        atomic_json(folder / 'DEPENDENCIES.json', freeze)
        atomic_json(folder / 'INVOCATION.json', dict(command=args, method_result=False,
            purpose='Full-state contact calibration and PhysX cooked-geometry capture',
            runtime_controller_unchanged=True, timeout_s=360))
        start = time.monotonic()
        with (folder / 'engine.log').open('w') as log:
            p = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                         PYTHONDONTWRITEBYTECODE='1'))
            atomic_json(folder / 'ACTIVE_PROCESS.json', dict(pid=p.pid, command=args))
            try:
                rc = p.wait(timeout=360)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
                rc = 'INFRASTRUCTURE_TIMEOUT'
        result = dict(side=side, returncode=rc, wall_seconds=time.monotonic()-start)
        if rc == 0 and (folder / 'trial_result.json').exists():
            from tools.finalize_common_dex3_grasp_qualification import standalone
            result['mechanical_score'] = standalone(folder, side)
            atomic_json(folder / 'QUALIFICATION.json', result['mechanical_score'])
        atomic_json(folder / 'PROCESS.json', result)
        print(side, result, flush=True)
        results.append(result)
        if rc != 0:
            break
    atomic_json(out / 'common_control/RECAPTURE_RESULTS.json', results)
    return results


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    run(args.run_dir.resolve(), args.resume)
