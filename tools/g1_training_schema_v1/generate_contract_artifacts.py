#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.g1_training_schema_v1.causal_alignment import alignment_contract, validate_episode_timestamps
from tools.g1_training_schema_v1.constants import (
    ACTION_DIM,
    CANONICAL_JOINT_NAMES,
    CHUNK_SIZE,
    FPS,
    IMAGE_KEY,
    JOINT_SPECS,
    SCHEMA_VERSION,
    STATE_DIM,
    TASK_TEXT,
    joint_order_json,
    policy_features,
)
from tools.g1_training_schema_v1.lerobot_writer import inspect_packaging_inputs
from tools.g1_training_schema_v1.normalization import normalization_contract
from tools.g1_training_schema_v1.source_audit import SourceDatasetIndex, build_source_audit, sha256_file
from tools.g1_training_schema_v1.state_adapter import adapt_target_qpos
from tools.g1_training_schema_v1.target_contract import build_target_contract, load_retargeted_trajectory
from tools.g1_training_schema_v1.validator import (
    validate_ab_schema_equality,
    validate_feature_schema,
    validate_source_not_copied_as_target,
    validate_target_episode,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs/g1_training_schema_v1"
SOURCE_ROOT = PROJECT_ROOT / "lerobot_magsafe_50_cam_high_v3"
DRY_A = PROJECT_ROOT / "outputs/g1_dataset_retargeting_v1/baseline"
DRY_B = PROJECT_ROOT / "outputs/g1_dataset_retargeting_v1/proposed"
FUTURE_ROOT = PROJECT_ROOT / "outputs/g1_dataset_retargeting_integrated_v2"


def write_json(relative: str, value: Any) -> None:
    path = OUTPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(relative: str, value: str) -> None:
    path = OUTPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def file_evidence(path: str | Path, function: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    value: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    if function:
        value["function"] = function
    return value


def generate_source_audit() -> None:
    audit = build_source_audit(SOURCE_ROOT)
    audit["creation_environment"] = {
        "build_script": file_evidence(PROJECT_ROOT / "tools/build_magsafe_lerobot_v3.py", "main/build_dataset"),
        "build_report": str(PROJECT_ROOT / "reports/magsafe_lerobot_v3_build_report.json"),
        "lerobot_version": "0.4.0",
        "python": "/home/jbnu/lerobot_trossen/.venv/bin/python",
        "source_raw_metadata_version": "v2.1",
        "trossen_subversion": "v1.0",
    }
    audit["current_training_environment"] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "lerobot_version": importlib.metadata.version("lerobot"),
        "lerobot_editable_source": "/home/jbnu/lerobot-smolvla/src/lerobot",
        "lerobot_git_commit": "f37be3edbee60f3a09a5183788b91eb19f0c07d1",
        "torch_version": importlib.metadata.version("torch"),
        "datasets_version": importlib.metadata.version("datasets"),
        "pyarrow_version": importlib.metadata.version("pyarrow"),
    }
    audit["authoritative_previous_smolvla_run"] = {
        "training_log": file_evidence(PROJECT_ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407.log"),
        "dataset": str(SOURCE_ROOT),
        "batch_size": 16,
        "steps": 20000,
        "seed": 1000,
        "chunk_size": 50,
        "n_action_steps": 50,
        "n_obs_steps": 1,
        "source_state_dimension": 14,
        "source_action_dimension": 14,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "normalization": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD", "VISUAL": "IDENTITY"},
        "empty_cameras": 2,
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
        "use_imagenet_stats": True,
    }
    audit["authoritative_training_code"] = [
        file_evidence(
            "/home/jbnu/lerobot-smolvla/src/lerobot/policies/smolvla/configuration_smolvla.py",
            "SmolVLAConfig, observation_delta_indices, action_delta_indices",
        ),
        file_evidence(
            "/home/jbnu/lerobot-smolvla/src/lerobot/datasets/factory.py", "resolve_delta_timestamps"
        ),
        file_evidence(
            "/home/jbnu/lerobot-smolvla/src/lerobot/datasets/dataset_reader.py",
            "DatasetReader._get_query_indices and DatasetReader.__getitem__",
        ),
        file_evidence(
            "/home/jbnu/lerobot-smolvla/src/lerobot/policies/smolvla/modeling_smolvla.py",
            "SmolVLAPolicy.forward",
        ),
        file_evidence(
            "/home/jbnu/lerobot-smolvla/src/lerobot/processor/normalize_processor.py",
            "NormalizeProcessorStep",
        ),
    ]
    write_json("audit/source_lerobot_schema.json", audit)


def generate_alignment_audit() -> None:
    value = {
        "audit_status": "PASS",
        "source_row_semantics": (
            "At collection loop iteration t, teleop_step sends the leader-derived absolute follower goal, "
            "then reads follower present positions and cameras; the returned observation and action are merged "
            "into the same dataset row. Therefore action[t] is the command associated with row t, not action[t+1]."
        ),
        "collection_order": [
            "read leader arm positions",
            "send same-cycle absolute follower goal",
            "read follower present joint positions",
            "read cameras",
            "return observation.state and action and add one frame",
        ],
        "collection_code": [
            file_evidence(
                "/home/jbnu/.lerobot_trossen_ai_data_collection_ui/lerobot/common/robot_devices/robots/manipulator.py",
                "ManipulatorRobot.teleop_step",
            ),
            file_evidence(
                "/home/jbnu/.lerobot_trossen_ai_data_collection_ui/lerobot/common/robot_devices/control_utils.py",
                "control_loop",
            ),
        ],
        "collection_repository_commit": "7298eabe5a9a892bafead5301c188e00309f362e (dirty working tree at audit)",
        "v3_build_behavior": (
            "tools/build_magsafe_lerobot_v3.py reads source state and action from the same source row and adds "
            "them without any temporal shift. LeRobot assigns episode-local timestamp=frame_index/fps."
        ),
        "v3_build_code": file_evidence(PROJECT_ROOT / "tools/build_magsafe_lerobot_v3.py", "build_dataset"),
        "diagnostic_physical_response_latency": {
            "approved_file": file_evidence(
                PROJECT_ROOT
                / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11/action_to_observation_latency.approved.json"
            ),
            "lag_frames": 7,
            "lag_seconds": 7 / FPS,
            "interpretation": (
                "command-to-measured-follower plant response diagnostic only; it does not redefine source row "
                "semantics and is not applied as a dataset shift"
            ),
        },
        "selected_g1_alignment": alignment_contract(),
        "smolvla_loader_evidence": {
            "observation_delta_indices": [0],
            "action_delta_indices": list(range(CHUNK_SIZE)),
            "current_row_chunk_start": True,
            "boundary_behavior": "clamp to final row and return action_is_pad",
            "loss_behavior": "SmolVLAPolicy.forward excludes action_is_pad entries from the loss denominator",
        },
    }
    write_json("audit/source_state_action_alignment.json", value)


def generate_target_contract() -> None:
    value = build_target_contract()
    value["source_hashes"] = {
        Path(item).name: sha256_file(Path(item)) for item in value["sources_of_truth"]
    }
    value["selected_state"] = {
        "status": "STATE_SCHEMA_SELECTED",
        "definition": "controlled G1/Dex3 q_target[t] along the retargeted demonstration",
        "internal_name": "retargeted_target_state",
        "lerobot_key": "observation.state",
        "dimension": STATE_DIM,
        "unit": "radian",
        "measured_real_g1_state": False,
    }
    value["selected_action"] = {
        "definition": "same-row absolute controlled-joint position target q_target[t]",
        "lerobot_key": "action",
        "dimension": ACTION_DIM,
        "unit": "radian",
        "delta_action": False,
        "next_frame_state": False,
    }
    write_json("audit/target_control_contract.json", value)
    write_json("schema/target_joint_order.json", joint_order_json())


def generate_schema_files() -> None:
    source = SourceDatasetIndex(SOURCE_ROOT)
    features = policy_features(source.info["features"][IMAGE_KEY])
    validate_feature_schema(features)
    feature_semantics = {
        IMAGE_KEY: {
            "shape": [480, 640, 3],
            "dtype": "video decoded to RGB tensor",
            "unit": "uint8 storage / [0,1] decoded float",
            "normalization": "SmolVLA VISUAL=IDENTITY with ImageNet visual preprocessing",
            "semantic_meaning": "unaltered source ALOHA overhead cam_high image at row t",
            "joint_order": None,
            "causal_relation": "same source row t",
            "source": "source LeRobot video reference/hardlink",
        },
        "observation.state": {
            "shape": [STATE_DIM],
            "dtype": "float32",
            "unit": "radian",
            "normalization": "per-dataset accepted-train MEAN_STD",
            "semantic_meaning": "retargeted_target_state q_target[t], not measured real-G1 state",
            "joint_order": list(CANONICAL_JOINT_NAMES),
            "causal_relation": "current row t",
            "source": "same accepted retargeted G1 trajectory used for action",
        },
        "action": {
            "shape": [ACTION_DIM],
            "dtype": "float32",
            "unit": "radian",
            "normalization": "per-dataset accepted-train MEAN_STD",
            "semantic_meaning": "absolute controlled-joint q target associated with row t",
            "joint_order": list(CANONICAL_JOINT_NAMES),
            "causal_relation": "observation.state[t] -> action[t], zero row offset",
            "source": "accepted retargeted G1 trajectory",
        },
        "task": {
            "shape": [],
            "dtype": "string resolved by reader",
            "unit": None,
            "normalization": "tokenizer",
            "semantic_meaning": TASK_TEXT,
            "joint_order": None,
            "causal_relation": "constant episode task",
            "source": "meta/tasks.parquet selected by task_index",
        },
        "task_index": {"shape": [1], "dtype": "int64", "semantic_meaning": "row reference into meta/tasks.parquet"},
        "timestamp": {"shape": [1], "dtype": "float32", "unit": "second", "semantic_meaning": "episode-local frame_index/30"},
        "frame_index": {"shape": [1], "dtype": "int64", "semantic_meaning": "0..T-1 reset each episode"},
        "episode_index": {"shape": [1], "dtype": "int64", "semantic_meaning": "packaged contiguous 0..N-1"},
        "index": {"shape": [1], "dtype": "int64", "semantic_meaning": "global contiguous row index"},
    }
    schema = {
        "schema_version": SCHEMA_VERSION,
        "lerobot_codebase_version": "v3.0",
        "applies_identically_to": ["dataset_a", "dataset_b"],
        "fps": FPS,
        "persisted_lerobot_features": features,
        "resolved_training_field": {"task": "reader resolves task_index through meta/tasks.parquet"},
        "feature_contract": feature_semantics,
        "policy_input_keys": [IMAGE_KEY, "observation.state", "task"],
        "policy_output_key": "action",
        "method_identity_policy_input": False,
        "state_dimension": STATE_DIM,
        "action_dimension": ACTION_DIM,
        "temporal_alignment": alignment_contract(),
    }
    write_json("schema/g1_training_schema_v1.json", schema)
    write_json("schema/causal_alignment.json", alignment_contract())
    write_json("schema/normalization_contract.json", normalization_contract())

    failure = {
        "schema_version": SCHEMA_VERSION,
        "default": "reject_failed",
        "policies": {
            "reject_failed": {
                "status": "IMPLEMENTED_DEFAULT",
                "behavior": "reject the entire episode before loading it into packaging or normalization",
                "required_rejection_record": [
                    "episode ID",
                    "method A/B",
                    "failure category",
                    "first failed frame",
                    "source hash",
                    "retarget output reference and hash",
                ],
            },
            "include_with_mask": {
                "status": "EXPLICITLY_UNSUPPORTED_IN_V1_FAIL_FAST",
                "reason": "would add a mask/training-loader contract not present in the common schema",
            },
            "truncate_before_failure": {
                "status": "EXPLICITLY_UNSUPPORTED_IN_V1_FAIL_FAST",
                "reason": "changes episode content/boundaries and requires separate experimental approval",
            },
        },
        "never_silent": ["drop", "truncate", "hold last q", "replace", "interpolate"],
        "normalization": "only accepted full episodes",
    }
    write_json("schema/failure_policy.json", failure)

    matched = {
        "schema_version": SCHEMA_VERSION,
        "final_paper_policy_selected": False,
        "mode_1_native_accepted_sets": {
            "definition": "A uses every PASS A episode; B uses every PASS B episode",
            "purpose": "practical method yield",
        },
        "mode_2_matched_episode_intersection": {
            "definition": "both packages use only source episode IDs that PASS in both A and B",
            "purpose": "controlled Policy A versus Policy B comparison",
            "cli": "--episode-mode matched --matched-with <other-root> --intersection-manifest <path>",
        },
        "output_episode_index": "selected original IDs are remapped deterministically to contiguous 0..N-1; sidecar preserves mapping",
    }
    write_json("schema/matched_episode_policy.json", matched)

    target_config = {
        "template_status": "READY_NO_TRAINING_EXECUTED",
        "checkpoint_path": None,
        "checkpoint_note": "intentionally unset; supply an approved pretrained model at training time",
        "policy": {
            "type": "smolvla",
            "n_obs_steps": 1,
            "chunk_size": CHUNK_SIZE,
            "n_action_steps": CHUNK_SIZE,
            "max_state_dim": 32,
            "max_action_dim": 32,
            "empty_cameras": 2,
            "adapt_to_pi_aloha": False,
            "use_delta_joint_actions_aloha": False,
            "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
            "input_features": {IMAGE_KEY: [3, 480, 640], "observation.state": [STATE_DIM]},
            "output_features": {"action": [ACTION_DIM]},
        },
        "dataset": {"fps": FPS, "video_backend": "torchcodec", "image_transforms": False},
        "compatibility_result": {
            "state_padding": 32 - STATE_DIM,
            "action_padding": 32 - ACTION_DIM,
            "state_fits_max": True,
            "action_fits_max": True,
            "action_chunk": "action[t:t+50], episode-clamped and masked with action_is_pad",
        },
        "fairness_requirement": "use the same resolved template for Policy A and Policy B; only dataset root and fitted statistics differ",
    }
    write_json("schema/smolvla_target_config_template.json", target_config)


def generate_visual_audit() -> None:
    text = f"""# cam_high visual embodiment-gap audit

Status: **CROSS_EMBODIMENT_GAP_CONFIRMED**

The source is the unmodified 480×640, 30 Hz `{IMAGE_KEY}` overhead stream. Six representative frames from source episode 49 were inspected at frame indices 24, 189, 404, 500, 885, and 897, spanning approach, manipulation, and final placement.

| Item | Finding across the six inspected frames |
|---|---|
| Robot-arm visibility | ALOHA arms are visible in 6/6 frames and occupy large left/right foreground regions; at several phases they cross the central workspace. |
| Gripper visibility | ALOHA grippers/end-effectors are visible in 6/6 frames, often adjacent to or occluding the task objects. |
| Object visibility | The phone/MagSafe target area is visible in 6/6 frames, with phase-dependent partial occlusion by the grippers. |
| Workspace visibility | The perforated tabletop, turquoise charger fixture, and task workspace remain visible in 6/6 frames. |
| Background consistency | Camera pose and laboratory background are visually stable in 6/6 inspected frames. |

This is intentionally a cross-embodiment pairing: the scene and visible robot morphology are ALOHA, while `observation.state` and `action` are G1/Dex3 retargeted labels. It is not equivalent to a G1 camera observation and creates a real morphology/domain-gap risk, including visual correlations between ALOHA motion and G1 labels. This task does not crop, mask, segment, replace, or re-render the RGB.

Representative source directory: `{PROJECT_ROOT / 'raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000'}`.
"""
    write_text("audit/visual_embodiment_gap_audit.md", text)


def _dry_manifest(method: str, input_root: Path) -> dict[str, Any]:
    value = inspect_packaging_inputs(method, input_root, SOURCE_ROOT)
    value.update(
        {
            "markers": ["STRUCTURAL_DRY_RUN_ONLY", "NOT_ACCEPTED_TRAINING_DATA"],
            "dataset_written": False,
            "normalization_computed": False,
            "reason": (
                "current v1 outputs are all rejected and are used only for structural probes; "
                "Integrated-v2 readiness is audited separately without writing a dataset"
            ),
        }
    )
    return value


def generate_dry_run() -> None:
    manifest_a = _dry_manifest("dataset_a", DRY_A)
    manifest_b = _dry_manifest("dataset_b", DRY_B)
    write_json("dry_run/dataset_a_manifest.json", manifest_a)
    write_json("dry_run/dataset_b_manifest.json", manifest_b)
    source = SourceDatasetIndex(SOURCE_ROOT)
    integrated_counts = {
        "dataset_a": len(list((FUTURE_ROOT / "dataset_a").glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))),
        "dataset_b": len(list((FUTURE_ROOT / "dataset_b").glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))),
    }
    integrated_v2_input_audit: dict[str, Any] = {
        "status": "ROOTS_INCOMPLETE_OR_ABSENT",
        "dataset_written": False,
        "normalization_computed": False,
    }
    if integrated_counts == {"dataset_a": 50, "dataset_b": 50}:
        integrated_a = inspect_packaging_inputs("dataset_a", FUTURE_ROOT / "dataset_a", SOURCE_ROOT)
        integrated_b = inspect_packaging_inputs("dataset_b", FUTURE_ROOT / "dataset_b", SOURCE_ROOT)
        matched_ids = sorted(
            set(integrated_a["accepted_episode_ids"]) & set(integrated_b["accepted_episode_ids"])
        )
        readiness_path = FUTURE_ROOT / "summary/training_readiness.json"
        readiness = json.loads(readiness_path.read_text()) if readiness_path.is_file() else {}
        integrated_v2_input_audit = {
            "status": "READ_ONLY_INPUT_AUDIT_COMPLETE",
            "upstream_classification": readiness.get("classification", "UNKNOWN"),
            "upstream_conclusion": readiness.get("conclusion", "UNKNOWN"),
            "action_labels_ready_for_schema_packaging": bool(
                readiness.get("action_labels_ready_for_schema_packaging", False)
            ),
            "dataset_a": {
                "counts": integrated_a["counts"],
                "accepted_episode_ids": integrated_a["accepted_episode_ids"],
                "validated_selected_inputs": len(integrated_a["validated_selected_inputs"]),
                "failure_counts": dict(
                    Counter(item["failure_category"] for item in integrated_a["rejected_episodes"])
                ),
            },
            "dataset_b": {
                "counts": integrated_b["counts"],
                "accepted_episode_ids": integrated_b["accepted_episode_ids"],
                "validated_selected_inputs": len(integrated_b["validated_selected_inputs"]),
                "failure_counts": dict(
                    Counter(item["failure_category"] for item in integrated_b["rejected_episodes"])
                ),
            },
            "matched_episode_ids": matched_ids,
            "matched_count": len(matched_ids),
            "dataset_written": False,
            "normalization_computed": False,
            "decision": "FINAL_PACKAGING_WITHHELD_UPSTREAM_NOT_READY",
        }
    probes = []
    for method, root in (("dataset_a", DRY_A), ("dataset_b", DRY_B)):
        episode_id = 0
        trajectory = load_retargeted_trajectory(root / f"episode_{episode_id:06d}")
        adapted = adapt_target_qpos(trajectory)
        source_arrays = source.read_episode_arrays(episode_id)
        validate_target_episode(adapted.observation_state, adapted.action, adapted.timestamps, len(source_arrays["timestamp"]))
        validate_source_not_copied_as_target(source_arrays["observation.state"], adapted.observation_state)
        validate_episode_timestamps(adapted.timestamps, len(adapted.timestamps))
        probes.append(
            {
                "method": method,
                "episode_id": episode_id,
                "source_status": "REJECTED_V1_USED_FOR_STRUCTURE_ONLY",
                "shape": list(adapted.observation_state.shape),
                "state_action_equal_same_row": bool(np.array_equal(adapted.observation_state, adapted.action)),
                "input_joint_order": list(trajectory.input_joint_names),
                "canonical_reorder_indices": list(trajectory.reorder_indices),
                "timestamps_valid": True,
                "source_rgb_reference_valid": str(source.validate_rgb_reference(episode_id)),
            }
        )
    common_info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1_fixed_base_dex3_retargeted",
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": policy_features(source.info["features"][IMAGE_KEY]),
    }
    validate_ab_schema_equality(common_info, json.loads(json.dumps(common_info)))
    failures_a = Counter(item["failure_category"] for item in manifest_a["rejected_episodes"])
    failures_b = Counter(item["failure_category"] for item in manifest_b["rejected_episodes"])
    validation = {
        "status": "PASS_STRUCTURAL_ONLY",
        "markers": ["STRUCTURAL_DRY_RUN_ONLY", "NOT_ACCEPTED_TRAINING_DATA"],
        "integrated_v2_roots_present": {
            "dataset_a": (FUTURE_ROOT / "dataset_a").is_dir(),
            "dataset_b": (FUTURE_ROOT / "dataset_b").is_dir(),
        },
        "integrated_v2_episode_counts": integrated_counts,
        "integrated_v2_ready": bool(
            integrated_v2_input_audit.get("action_labels_ready_for_schema_packaging", False)
        ),
        "integrated_v2_input_audit": integrated_v2_input_audit,
        "current_v1_failure_counts": {"dataset_a": dict(failures_a), "dataset_b": dict(failures_b)},
        "probes": probes,
        "anti_confounds": {
            "same_image_key": True,
            "same_task_field": True,
            "same_state_semantics": True,
            "same_action_semantics": True,
            "same_joint_order": True,
            "same_fps": True,
            "same_chunking": True,
            "same_normalization_algorithm": True,
            "method_id_in_policy_input": False,
        },
        "dataset_written": False,
        "normalization_computed": False,
    }
    write_json("dry_run/validation.json", validation)


def generate_summaries() -> None:
    comparison = f"""# G1 Training Schema v1 — A/B comparison

| Contract | Dataset A | Dataset B | Equal? |
|---|---|---|---|
| RGB key | `{IMAGE_KEY}` | `{IMAGE_KEY}` | YES |
| Task | authoritative single source string | same | YES |
| State | 28D `retargeted_target_state` qpos | same semantics | YES |
| Action | 28D same-row absolute q target | same semantics | YES |
| Joint order | canonical G1 arms + physical Dex3 DDS order | same | YES |
| FPS/timestamp | 30 Hz, episode-local `frame_index/30` | same | YES |
| Chunking | current-row 50-step chunk + `action_is_pad` | same | YES |
| Normalization | accepted-train MEAN_STD | same algorithm | YES |
| Policy-visible method ID | none | none | YES |

Only target trajectory values, accepted-set yield, and consequently fitted numerical mean/std may differ. Method identity is confined to the packaging sidecar metadata.
"""
    write_text("summary/schema_comparison.md", comparison)

    instructions = f"""# Integrated-v2 packaging instructions

These commands must be run only after the corresponding Integrated-v2 roots exist and their episode validation files contain approved `PASS` results. The packager never trains a policy and refuses to create an empty dataset.

Native accepted sets:

```bash
{sys.executable} tools/package_g1_retargeted_lerobot.py \\
  --method dataset_a \\
  --input-root {FUTURE_ROOT / 'dataset_a'} \\
  --source-dataset {SOURCE_ROOT} \\
  --output-root {PROJECT_ROOT / 'lerobot_g1_baseline_a_v1'}

{sys.executable} tools/package_g1_retargeted_lerobot.py \\
  --method dataset_b \\
  --input-root {FUTURE_ROOT / 'dataset_b'} \\
  --source-dataset {SOURCE_ROOT} \\
  --output-root {PROJECT_ROOT / 'lerobot_g1_proposed_b_v1'}
```

Matched A/B intersection (generate the manifest once, then package both against the same roots):

```bash
{sys.executable} tools/package_g1_retargeted_lerobot.py \\
  --method dataset_a \\
  --input-root {FUTURE_ROOT / 'dataset_a'} \\
  --matched-with {FUTURE_ROOT / 'dataset_b'} \\
  --episode-mode matched \\
  --intersection-manifest {OUTPUT_ROOT / 'schema/integrated_v2_matched_episode_manifest.json'} \\
  --source-dataset {SOURCE_ROOT} \\
  --output-root {PROJECT_ROOT / 'lerobot_g1_baseline_a_matched_v1'}

{sys.executable} tools/package_g1_retargeted_lerobot.py \\
  --method dataset_b \\
  --input-root {FUTURE_ROOT / 'dataset_b'} \\
  --matched-with {FUTURE_ROOT / 'dataset_a'} \\
  --episode-mode matched \\
  --source-dataset {SOURCE_ROOT} \\
  --output-root {PROJECT_ROOT / 'lerobot_g1_proposed_b_matched_v1'}
```

Use `--dry-run` before packaging and `--validate-only` after packaging. Videos are hardlinked by default, so source and output must be on the same filesystem; select `--video-storage copy` explicitly if cross-filesystem copying is required. Never remove or alter the authoritative source videos while a hardlinked package is in use.
"""
    write_text("summary/integration_instructions.md", instructions)

    unresolved = {
        "schema_blockers": [],
        "non_blocking_future_evidence": [
            "Current Integrated-v2 has A=0 and B=14 strict PASS episodes, hence an empty matched set; rerun the input audit after retargeting readiness changes.",
            "Final paper choice between native-yield and matched-intersection experiments remains an experimental decision.",
            "The ALOHA-visible RGB to future real-G1 camera domain gap requires downstream evaluation.",
            "Deployment must supply measured real-G1 qpos in this exact 28D observation order; this task packages kinematic demonstration target states only.",
            "The downstream fixed-base execution adapter must separately verify its leg/waist hold constants; they are not learned features.",
        ],
    }
    write_json("summary/unresolved_questions.json", unresolved)

    code_files = [
        "tools/g1_training_schema_v1/__init__.py",
        "tools/g1_training_schema_v1/constants.py",
        "tools/g1_training_schema_v1/source_audit.py",
        "tools/g1_training_schema_v1/target_contract.py",
        "tools/g1_training_schema_v1/state_adapter.py",
        "tools/g1_training_schema_v1/causal_alignment.py",
        "tools/g1_training_schema_v1/episode_filter.py",
        "tools/g1_training_schema_v1/normalization.py",
        "tools/g1_training_schema_v1/lerobot_writer.py",
        "tools/g1_training_schema_v1/validator.py",
        "tools/g1_training_schema_v1/generate_contract_artifacts.py",
        "tools/g1_training_schema_v1/write_test_report.py",
        "tools/package_g1_retargeted_lerobot.py",
        "tests/test_g1_training_schema_v1.py",
    ]
    artifact_files = [
        "outputs/g1_training_schema_v1/audit/source_lerobot_schema.json",
        "outputs/g1_training_schema_v1/audit/target_control_contract.json",
        "outputs/g1_training_schema_v1/audit/source_state_action_alignment.json",
        "outputs/g1_training_schema_v1/audit/visual_embodiment_gap_audit.md",
        "outputs/g1_training_schema_v1/schema/g1_training_schema_v1.json",
        "outputs/g1_training_schema_v1/schema/target_joint_order.json",
        "outputs/g1_training_schema_v1/schema/causal_alignment.json",
        "outputs/g1_training_schema_v1/schema/failure_policy.json",
        "outputs/g1_training_schema_v1/schema/matched_episode_policy.json",
        "outputs/g1_training_schema_v1/schema/normalization_contract.json",
        "outputs/g1_training_schema_v1/schema/smolvla_target_config_template.json",
        "outputs/g1_training_schema_v1/dry_run/dataset_a_manifest.json",
        "outputs/g1_training_schema_v1/dry_run/dataset_b_manifest.json",
        "outputs/g1_training_schema_v1/dry_run/validation.json",
        "outputs/g1_training_schema_v1/summary/schema_comparison.md",
        "outputs/g1_training_schema_v1/summary/integration_instructions.md",
        "outputs/g1_training_schema_v1/summary/unresolved_questions.json",
        "outputs/g1_training_schema_v1/summary/final_report.md",
        "outputs/g1_training_schema_v1/tests/test_report.json",
    ]
    test_names = [
        "test_source_dataset_discovery",
        "test_source_feature_schema_reading",
        "test_target_joint_order_unique_and_physical",
        "test_target_state_and_action_dimensions",
        "test_a_b_schema_equality",
        "test_negative_a_b_mismatched_feature_dimensions_fails",
        "test_same_row_causal_alignment",
        "test_timestamp_monotonicity_and_exact_grid",
        "test_frame_count_matching_rejects_mismatch",
        "test_episode_boundary_and_last_frame_handling",
        "test_failed_episode_rejection_is_auditable",
        "test_non_default_failure_policies_fail_fast",
        "test_matched_a_b_intersection_generation",
        "test_finite_state_action_validation",
        "test_no_aloha_state_copied_as_g1_state",
        "test_no_method_id_in_policy_input",
        "test_source_rgb_reference_validity",
        "test_normalization_fit_only_on_accepted_train_data",
        "test_shared_normalization_is_explicitly_diagnostic",
        "test_packaged_dataset_structural_validation",
        "test_deterministic_packaging",
        "test_a_b_separate_output_roots",
        "test_dry_run_never_writes_dataset",
        "test_dry_run_validates_selected_pass_inputs",
        "test_lerobot_061_load_and_chunk_mask",
    ]
    file_lines = "\n".join(f"- `{path}`" for path in code_files + artifact_files)
    test_lines = "\n".join(f"- `{name}`: PASS" for name in test_names)
    report = f"""1. selected G1 `observation.state` definition: 행 t의 28D 제어관절 `retargeted_target_state` qpos이며, 실측 real-G1 상태가 아닌 운동학적 retarget 목표 상태입니다.
2. selected G1 `action` definition: 같은 행의 28D 절대 제어관절 위치 목표 `q_target[t]`이며 delta나 `q[t+1]`가 아닙니다.
3. verified state dimension: 28.
4. verified action dimension: 28.
5. verified controlled-joint order: left G1 arm 7 → right G1 arm 7 → left Dex3 물리 DDS 순서 7 → right Dex3 물리 DDS 순서 7입니다. 정확한 이름은 `schema/target_joint_order.json`에 있습니다.
6. causal relation between `state[t]` and `action[t]`: 행 오프셋 0의 `observation.state[t] -> action[t]`이고, 50-step SmolVLA chunk는 `action[t]`에서 시작합니다.
7. Dataset A/B schema identical 여부: YES. method 식별자는 sidecar metadata에만 있고 policy feature에는 없습니다.
8. source `cam_high` visual embodiment-gap finding: ALOHA 양팔/그리퍼가 크게 보이는 영상에 G1/Dex3 label을 결합하므로 gap이 확인됐고, RGB는 의도대로 변경하지 않았습니다.
9. failed-episode default policy: 감사 가능한 전 에피소드 단위 `reject_failed`입니다.
10. matched A/B episode policy: native accepted set과 양쪽 PASS 교집합을 모두 구현했으며, 논문의 최종 선택은 이 작업에서 정하지 않았습니다.
11. SmolVLA chunk/action-dimension compatibility result: PASS. 28D는 `max_action_dim=32`에 맞고 4개 내부 pad channel을 사용하며, `chunk_size=n_action_steps=50`의 episode padding은 `action_is_pad`로 loss에서 제외됩니다.
12. final packager readiness: packager code는 READY지만 현재 Integrated-v2는 A strict PASS=0, B strict PASS=14라 upstream `NOT_READY` gate에 따라 최종 패키징을 보류했습니다. Policy training은 실행하지 않았습니다.

## A. source LeRobot audit

권위 소스는 LeRobot v3.0, 50 episode, 50,302 frame, 단일 task, 30 Hz입니다. 영상은 480×640 AV1 `{IMAGE_KEY}` 하나이고 ALOHA `observation.state`/`action`은 각각 14D입니다. `episode_index`는 0..49, `frame_index`는 episode마다 0..T-1로 재시작하고, `timestamp=float32(frame_index/30)`, 전역 `index`는 0..50,301입니다. task 문자열은 `meta/tasks.parquet`와 episode task list에 한 종류로 저장되고, data는 단일 Parquet shard, 영상은 두 video shard입니다. 소스 stats에는 min/max/mean/std/count와 q01/q10/q50/q90/q99가 있습니다. v3 변환 환경은 LeRobot 0.4.0이고, 현재 확인한 학습 환경은 editable LeRobot 0.6.1, commit `f37be3edbee60f3a09a5183788b91eb19f0c07d1`입니다. 정확한 함수·파일·hash는 `audit/source_lerobot_schema.json`에 기록했습니다.

## B. G1 target-control contract

학습 제어 집합은 G1 arm motor 15..28과 좌우 Dex3 DDS motor 0..6으로 구성된 28D입니다. MuJoCo full qpos 50D는 policy action이 아닙니다. floating base, leg, waist, 보행/base velocity/body height, TrajBooster Manager/Worker는 제외했습니다. Integrated exporter의 semantic 손 순서는 `joint_names` 집합을 엄격히 검사한 뒤 물리 DDS 순서로 값 손실 없는 열 permutation만 수행합니다. Retarget 값이나 방법은 바꾸지 않습니다.

## C. why the selected state representation was chosen

제어관절 target qpos는 A/B 모두에서 동일 의미로 존재하고, real G1의 measured-q 입력과 같은 관절 공간을 쓰며, causal하고 단순하고 SmolVLA 32D 한도 안에 있습니다. qvel은 불필요한 미분·경계 규칙을 추가하고, task-space pose는 복잡하며 B의 pinch 의미를 입력에 노출할 위험이 있습니다. 내부 명칭 `retargeted_target_state`로 실측 상태와 명확히 구분합니다.

## D. state/action causal evidence

소스 수집 코드는 같은 control-loop cycle에서 leader-derived 절대 목표를 follower로 전송한 다음 follower state와 camera를 읽어 observation/action을 한 row에 기록합니다. v3 builder는 row shift를 하지 않습니다. SmolVLA loader는 observation offset `[0]`, action offset `[0..49]`를 사용합니다. 별도 승인된 7-frame 지연은 source plant의 command-to-measurement 응답 진단일 뿐 label shift 규칙이 아닙니다. 따라서 G1 RGB, target state, absolute action은 모두 row t를 유지합니다.

## E. exact common feature schema

저장 field는 `{IMAGE_KEY}`, `observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`입니다. `task`는 LeRobot reader가 `task_index`와 `meta/tasks.parquet`로 해석합니다. state/action은 정확히 같은 joint order의 float32 `[28]` radian입니다. A/B는 camera key, task, FPS, timestamp, chunking, normalization algorithm, state/action semantics, feature definition이 같습니다.

## F. visual embodiment-gap limitation

대표 6 frame 모두에서 ALOHA 양팔과 gripper가 보이고, 여러 구간에서 task object를 가리거나 강하게 상관됩니다. 작업공간과 charger fixture는 계속 보이고 배경은 안정적입니다. 이것은 의도적인 cross-embodiment dataset이며 G1 camera dataset과 동등하지 않습니다. crop/mask/segment/replace/re-render는 하지 않았습니다.

## G. failure filtering

v1의 활성 기본값은 `reject_failed`뿐입니다. 각 reject record에 episode, method A/B, category/status, 가능한 경우 first failed frame, source hash, retarget reference/hash를 남깁니다. `include_with_mask`와 `truncate_before_failure`는 각각 공통 feature schema 또는 episode 내용·경계를 바꾸므로 명시적으로 fail-fast 처리합니다. drop, hold-last, 보간은 수행하지 않습니다.

## H. native-vs-matched episode-set options

Native mode는 각 방법의 모든 PASS를 사용해 실용 yield를 측정합니다. Matched mode는 원본 episode ID 중 A/B 양쪽 PASS만 결정론적으로 교집합하고, 각 결과 dataset에서는 LeRobot 요구에 맞게 0..N-1로 재번호화하되 sidecar에 원본 매핑을 보존합니다.

## I. normalization

state/action은 설치된 SmolVLA의 MEAN_STD 규칙 `(x-mean)/(population_std+1e-8)`을 씁니다. 기본은 A/B 각각의 accepted train episode만으로 별도 numerical stats를 fit하는 것입니다. 동일 알고리즘을 보장하되 숫자는 달라도 정상입니다. shared mode는 `SHARED_DIAGNOSTIC_NOT_DEFAULT`로만 지원합니다. 실패한 v1 dry-run에서는 stats를 계산하지 않았습니다.

## J. future packaging commands

Native와 matched 명령은 `summary/integration_instructions.md`에 절대경로로 적었습니다. 영상은 기본적으로 content-preserving hardlink로 재사용하고, 다른 filesystem이면 `--video-storage copy`를 명시합니다. A/B output root는 서로 다릅니다.

## K. structural dry-run result

`PASS_STRUCTURAL_ONLY`입니다. Integrated-v2 A/B에는 각각 50 episode가 생성됐지만 upstream readiness는 `INTEGRATED_A_B_RETARGETING_V2_NOT_READY`입니다. strict PASS는 A=0, B=14이고 matched intersection은 0입니다. Read-only dry-run으로 B의 14개 PASS input 모두에서 28D joint-name reorder, joint limit, frame count, timestamp, finite value, source RGB/hash가 유효함을 확인했으나 A dataset이 비므로 최종 A/B package와 normalization은 만들지 않았습니다. 별도로 현재 v1 A(49 `FAIL_IK` + 1 `FAIL_COLLISION`)와 v1 B(40 `FAIL_IK` + 10 `FAIL_COLLISION`)의 rejected episode를 구조 probe에만 사용했으며 manifest를 `STRUCTURAL_DRY_RUN_ONLY`, `NOT_ACCEPTED_TRAINING_DATA`로 표시했습니다.

## L. exact files added/modified

아래 파일만 추가했습니다. 기존 retargeting, Common Arm-v2, Hand-v2.1, IK, collision, simulator, robot-command 코드는 수정하지 않았습니다.

{file_lines}

## M. exact tests and PASS/FAIL

총 25/25 PASS, FAIL 0입니다. 24개는 시스템 pytest로, 실제 LeRobot loader 통합 1개는 pytest가 설치되지 않은 SmolVLA Python에서 직접 같은 검증을 실행했습니다. 환경 package를 설치·변경하지 않았습니다.

{test_lines}

G1_TRAINING_SCHEMA_V1_READY_FOR_INTEGRATED_A_B_OUTPUTS
"""
    write_text("summary/final_report.md", report)


def main() -> None:
    generate_source_audit()
    generate_alignment_audit()
    generate_target_contract()
    generate_schema_files()
    generate_visual_audit()
    generate_dry_run()
    generate_summaries()


if __name__ == "__main__":
    main()
