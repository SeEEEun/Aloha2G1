#!/usr/bin/env python3
"""Build immutable Fair-A/Proposed-B reference commands for alignment preflight."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from prepare_contact_eval10_physical_commands import semantic_stages
from evaluation.contracts import authoritative_joint_ranges


ROOT = Path("/home/jbnu/aloha_g1_dataset")
AUDIT = ROOT / "outputs/final_contact_constrained_eval/08_task_frame_alignment_audit/TASK_FRAME_ALIGNMENT_AUDIT.json"
OUT = ROOT / "outputs/final_contact_constrained_eval/09_reference_alignment_preflight/commands"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    authoritative_names, _ = authoritative_joint_ranges()
    records: list[dict[str, object]] = []
    for row in audit["per_episode"]:
        method = str(row["method"])
        reference = Path(row["reference_path"]).resolve()
        if sha256(reference) != row["reference_sha256"]:
            raise RuntimeError(f"reference hash drift: {reference}")
        with np.load(reference, allow_pickle=False) as source:
            commands = np.asarray(source["replay_named_joint_qpos"], dtype=np.float32)
            source_names = source["replay_joint_names"].astype(str).tolist()
        if set(source_names) != set(authoritative_names):
            raise RuntimeError(f"reference named-joint set mismatch: {reference}")
        # Pure named-column canonicalization: values remain attached to their
        # authoritative joint names; no trajectory value is recomputed.
        reorder = [source_names.index(name) for name in authoritative_names]
        commands = commands[:, reorder]
        names = np.asarray(authoritative_names)
        stages = semantic_stages(reference, len(commands))
        destination = OUT / method.lower().replace("-", "_") / f"eval_{int(row['eval_index']):02d}_{row['stable_episode_id']}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                commanded_q_rad=commands,
                stage=stages,
                joint_names=names,
                control_fps_hz=np.asarray(30.0),
                source_reference=np.asarray(str(reference)),
                source_reference_sha256=np.asarray(row["reference_sha256"]),
                method=np.asarray(method),
                eval_index=np.asarray(int(row["eval_index"])),
                stable_episode_id=np.asarray(row["stable_episode_id"]),
                reference_level_preflight=np.asarray(True),
                commands_modified=np.asarray(False),
                common_controller_override_mask=np.zeros_like(commands, dtype=bool),
                runtime_right_three_digit_gate_required=np.asarray(False),
            )
        os.replace(temporary, destination)
        records.append(
            {
                "method": method,
                "eval_index": int(row["eval_index"]),
                "stable_episode_id": row["stable_episode_id"],
                "source_reference": str(reference),
                "source_reference_sha256": row["reference_sha256"],
                "command": str(destination.resolve()),
                "command_sha256": sha256(destination),
                "frames": len(commands),
                "commands_modified": False,
                "named_columns_canonicalized": source_names != authoritative_names,
            }
        )
    manifest = {
        "schema_version": "reference_alignment_preflight_commands_v1",
        "status": "PASS",
        "records": records,
        "method_or_episode_specific_correction": False,
        "reference_commands_modified": False,
    }
    manifest_path = OUT.parent / "REFERENCE_COMMAND_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "commands": len(records), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
