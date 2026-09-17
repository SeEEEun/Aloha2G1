#!/usr/bin/env python3
"""Run the frozen 150 mm scripted command through PhysX articulation targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ISAAC_PYTHON = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
ENGINE = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
SCORER = ROOT / "tools/score_contact_constrained_full_task.py"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
COMMAND = (
    ROOT
    / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm"
    / "bin_calibrated_full_command.npz"
)
DEFAULT_ROOT = ROOT / "outputs/final_contact_constrained_eval/02_scripted_validation"
EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    COMMAND: "fc8b81b001f031c141843c55e78e606ae478c534b12ec86bfe9728a4a93bf724",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen dependency changed: {path}: {actual}")
    output = args.output_root.resolve() / f"run_{args.run_index:02d}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scripted physics evidence: {output}")
    output.mkdir(parents=True)
    engine_command = [
        str(ISAAC_PYTHON),
        str(ENGINE),
        "--config",
        str(CONFIG),
        "--side",
        "right",
        "--geometry",
        "FROZEN_COMPRESSED_SHORT_55",
        "--profile",
        "P14",
        "--output-dir",
        str(output),
        "--scripted-command-path",
        str(COMMAND),
        "--object-spawn-side",
        "left",
        "--audit-robot-bin",
        "--full-task-audit",
        "--bin-height-m",
        "0.150",
        "--bin-rim-bevel-m",
        "0.003",
        "--headless",
    ]
    manifest = {
        "schema_version": "contact_constrained_scripted_invocation_v1",
        "execution_mode": "CONTACT_CONSTRAINED_PHYSICS",
        "run_index": args.run_index,
        "engine_command": engine_command,
        "engine": str(ENGINE),
        "engine_sha256": sha256(ENGINE),
        "scorer": str(SCORER),
        "scorer_sha256": sha256(SCORER),
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "command": str(COMMAND),
        "command_sha256": sha256(COMMAND),
        "bin_height_m": 0.150,
        "bin_bottom_world_z_m": 0.795,
        "bin_rim_bevel_m": 0.003,
        "direct_state_writes_after_initialization": False,
        "state_restoration": False,
        "object_pose_writes_after_initialization": False,
        "attachment_or_magnet": False,
    }
    dump(output / "INVOCATION_MANIFEST.json", manifest)
    log = output / "engine.log"
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            engine_command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if process.returncode not in (0, 2):
        raise RuntimeError(
            f"physics engine infrastructure failure {process.returncode}; see {log}"
        )
    required = [output / name for name in ("event_log.npz", "robot_bin_contacts.npz", "trial_result.json")]
    if any(not path.is_file() for path in required):
        raise RuntimeError("physics engine did not persist the required complete trace")
    score_process = subprocess.run(
        [str(ISAAC_PYTHON), str(SCORER), "--run-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "scorer.log").write_text(
        score_process.stdout + score_process.stderr, encoding="utf-8"
    )
    result_path = output / "CONTACT_CONSTRAINED_TASK_RESULT.json"
    if not result_path.is_file():
        raise RuntimeError(f"scorer did not produce a result; see {output / 'scorer.log'}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    completion = {
        **manifest,
        "wall_seconds": time.monotonic() - started,
        "legacy_engine_exit_code": process.returncode,
        "contact_constrained_score_exit_code": score_process.returncode,
        "contact_constrained_status": result["status"],
        "artifacts": {
            str(path.name): {"path": str(path), "sha256": sha256(path)}
            for path in [*required, result_path, output / "CONTACT_CONSTRAINED_TASK_RESULT.md"]
        },
    }
    dump(output / "RUN_MANIFEST.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
