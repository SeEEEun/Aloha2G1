#!/usr/bin/env python3
"""Explicit read-only instrumentation of the existing physical control engine.

No module attributes or scientific settings are monkey-patched. This adapter
adds runtime collider provenance, progress and recoverable measured checkpoints.
"""
import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import run_direct_physical_execution_isaac as engine


def instrument(source):
    result, counts = engine.instrument(source)
    anchor = '    dt = float(config["timing"]["physics_dt_s"])\n'
    assert result.count(anchor) == 1
    result = result.replace(anchor, '    from tools.reconciled_ab.runtime_geometry import capture\n    capture(stage, output_dir)\n' + anchor)
    anchor = '        common_snapshot = common_execution.snapshot(\n'
    assert result.count(anchor) == 1
    addition = '''        if control_frame % 120 == 0:
            print("HYBRID_MEASURED_PROGRESS", control_frame, len(commands), flush=True)
            if records["measured_q_rad"]:
                from tools.contact_coordination.io import atomic_npz
                atomic_npz(output_dir / "MEASURED_CHECKPOINT.npz", **{k: np.asarray(v) for k, v in records.items()})
'''
    result = result.replace(anchor, addition + anchor)
    result = result.replace('"policy_or_checkpoint_used": True,', '"policy_or_checkpoint_used": False,')
    compile(result, str(engine.ENGINE), 'exec')
    counts.update(read_only_geometry_capture=1, measured_checkpoint=1, provenance_policy_flag=1)
    return result, counts


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--direct-freeze-manifest', type=Path, required=True)
    parser.add_argument('--qualification-mode', action='store_true')
    parser.add_argument('--validate-patch-only', action='store_true')
    known, remaining = parser.parse_known_args()
    result, counts = instrument(engine.ENGINE.read_text())
    if known.validate_patch_only:
        print(counts); return 0
    os.environ['DIRECT_EVAL35_FREEZE_MANIFEST'] = str(known.direct_freeze_manifest.resolve())
    if known.qualification_mode:
        os.environ['DIRECT_EXECUTION_QUALIFICATION_MODE'] = '1'
    sys.argv = [str(engine.ENGINE), *remaining, '--dex3-hard-limit-contract',
                str(engine.AUTHORITATIVE_JOINT_CONTRACT), '--dex3-hard-limit-inset-rad', engine.DEX3_JOINT_STOP_INSET_RAD]
    namespace = dict(__name__='__main__', __file__=str(engine.ENGINE), __package__=None)
    exec(compile(result, str(engine.ENGINE), 'exec'), namespace, namespace)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
