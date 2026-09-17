from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from tools.g1_training_schema_v1.constants import ACTION_KEY, IMAGE_KEY, STATE_KEY

BASE_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
BASE_SNAPSHOT = Path(
    "/home/jbnu/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots"
) / BASE_REVISION
PRIMARY_SEED = 1000
OPTIONAL_SEEDS = [1001, 1002]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _policy_features() -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = {
        IMAGE_KEY: {"type": "VISUAL", "shape": [3, 480, 640]},
        STATE_KEY: {"type": "STATE", "shape": [28]},
        "observation.images.empty_camera_0": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.empty_camera_1": {"type": "VISUAL", "shape": [3, 480, 640]},
    }
    outputs = {ACTION_KEY: {"type": "ACTION", "shape": [28]}}
    return inputs, outputs


def make_training_config(
    prior_successful_config: str | Path,
    dataset_root: str | Path,
    dataset_repo_id: str,
    output_dir: str | Path,
    job_name: str,
) -> dict[str, Any]:
    """Derive an executable 0.6.1 config from the prior successful 20k run.

    This does not load weights or start training.  The only target-policy changes
    from that proven machine configuration are the local matched dataset identity,
    the 28D feature contract, and a pinned official SmolVLA base snapshot.
    """

    prior = json.loads(Path(prior_successful_config).read_text(encoding="utf-8"))
    config = copy.deepcopy(prior)
    config["dataset"]["repo_id"] = dataset_repo_id
    config["dataset"]["root"] = str(Path(dataset_root).resolve())
    config["dataset"]["episodes"] = None
    config["dataset"]["revision"] = None
    config["dataset"]["image_transforms"]["enable"] = False
    config["dataset"]["video_backend"] = "torchcodec"
    config["dataset"]["eval_split"] = 0.0
    inputs, outputs = _policy_features()
    config["policy"]["input_features"] = inputs
    config["policy"]["output_features"] = outputs
    config["policy"]["pretrained_path"] = str(BASE_SNAPSHOT)
    config["policy"]["pretrained_revision"] = None
    config["policy"]["empty_cameras"] = 2
    config["policy"]["adapt_to_pi_aloha"] = False
    config["policy"]["use_delta_joint_actions_aloha"] = False
    config["policy"]["use_amp"] = True
    config["policy"]["use_peft"] = False
    config["policy"]["push_to_hub"] = False
    config["policy"]["repo_id"] = None
    config["policy"]["chunk_size"] = 50
    config["policy"]["n_action_steps"] = 50
    config["policy"]["max_state_dim"] = 32
    config["policy"]["max_action_dim"] = 32
    config["policy"]["normalization_mapping"] = {
        "VISUAL": "IDENTITY",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }
    config["output_dir"] = str(Path(output_dir).resolve())
    config["job_name"] = job_name
    config["resume"] = False
    config["seed"] = PRIMARY_SEED
    config["num_workers"] = 4
    config["batch_size"] = 16
    config["steps"] = 20_000
    config["log_freq"] = 100
    config["save_checkpoint"] = True
    config["save_freq"] = 5_000
    config["env"] = None
    config["eval_steps"] = 0
    config["peft"] = None
    config["save_checkpoint_to_hub"] = False
    config["checkpoint_path"] = None
    return config


def write_training_configs(project_root: str | Path, output_root: str | Path) -> tuple[Path, Path]:
    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    prior = project_root / (
        "outputs/smolvla_magsafe_batch16_20k_20260729_140407/"
        "checkpoints/020000/pretrained_model/train_config.json"
    )
    common = {
        "prior_successful_config": prior,
        "project_root": project_root,
    }
    path_a = output_root / "training_configs/policy_a_config.json"
    path_b = output_root / "training_configs/policy_b_config.json"
    config_a = make_training_config(
        common["prior_successful_config"],
        project_root / "lerobot_g1_magsafe_matched51_baseline_a_v1",
        "local/g1_magsafe_matched51_baseline_a_v1",
        project_root / "outputs/policy_a_matched51_smolvla_v1",
        "policy_a_matched51_smolvla_v1",
    )
    config_b = make_training_config(
        common["prior_successful_config"],
        project_root / "lerobot_g1_magsafe_matched51_proposed_b_v1",
        "local/g1_magsafe_matched51_proposed_b_v1",
        project_root / "outputs/policy_b_matched51_smolvla_v1",
        "policy_b_matched51_smolvla_v1",
    )
    _write_json(path_a, config_a)
    _write_json(path_b, config_b)
    return path_a, path_b


def config_fairness(path_a: str | Path, path_b: str | Path) -> dict[str, Any]:
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    differences: list[str] = []

    def walk(left: Any, right: Any, prefix: str = "") -> None:
        if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
            for key in sorted(left):
                walk(left[key], right[key], f"{prefix}.{key}" if prefix else key)
        elif left != right:
            differences.append(prefix)

    walk(a, b)
    allowed = {"dataset.repo_id", "dataset.root", "output_dir", "job_name"}
    return {
        "status": "PASS" if set(differences) == allowed else "FAIL",
        "actual_difference_paths": differences,
        "allowed_difference_paths": sorted(allowed),
        "all_other_fields_identical": set(differences) == allowed,
        "same_pretrained_initialization": a["policy"]["pretrained_path"] == b["policy"]["pretrained_path"],
        "same_primary_seed": a["seed"] == b["seed"] == PRIMARY_SEED,
        "primary_seed": PRIMARY_SEED,
        "optional_future_robustness_seeds": OPTIONAL_SEEDS,
        "gradient_accumulation_steps": 1,
        "gradient_accumulation_note": "LeRobot 0.6.1 TrainPipelineConfig has no accumulation field and steps optimizer once per batch",
        "training_executed": False,
    }
