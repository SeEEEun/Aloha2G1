#!/usr/bin/env python3
"""Project-side UniFoLM adapter for the fixed G1 arms + Dex3 28D contract.

This module deliberately does not edit the official UniFoLM checkout.  The
runtime installation helpers are opt-in and must run before importing the
upstream action head or training entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EMBODIMENT_NAME = "G1_DEX3_28D"
STATE_DIM = 28
ACTION_DIM = 28
ACTION_HORIZON = 25
NORMALIZATION_TYPE = "bounds"
TASK_INSTRUCTION = (
    "Pick up the doll with the left hand, handoff it to the right hand, "
    "and place it in the trash bin."
)
DEFAULT_STATISTICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs/policy_class_diagnostic/unifolm_b/embodiment_28d/dataset_statistics_28d.json"
)

JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)
MAPPED_INDICES = tuple(range(STATE_DIM))

# These are the only released-checkpoint tensors whose shapes are tied to the
# native 23D interface.  Encoder biases are shape-compatible and reusable.
DIMENSION_TIED_CHECKPOINT_KEYS = {
    "action_model.state_encoder.layer1.weight": (1024, STATE_DIM),
    "action_model.action_encoder.layer1.weight": (1536, ACTION_DIM),
    "action_model.action_decoder.layer2.weight": (ACTION_DIM, 1024),
    "action_model.action_decoder.layer2.bias": (ACTION_DIM,),
}


@dataclass(frozen=True)
class EmbodimentSpec:
    name: str = EMBODIMENT_NAME
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    action_horizon: int = ACTION_HORIZON
    normalization_type: str = NORMALIZATION_TYPE
    joint_names: tuple[str, ...] = JOINT_NAMES
    mapped_indices: tuple[int, ...] = MAPPED_INDICES
    task_instruction: str = TASK_INSTRUCTION

    def validate(self) -> None:
        validate_joint_mapping(self.joint_names, self.mapped_indices)
        if self.state_dim != len(self.joint_names):
            raise ValueError("state_dim must equal the joint count")
        if self.action_dim != len(self.joint_names):
            raise ValueError("action_dim must equal the joint count")
        if self.action_horizon != ACTION_HORIZON:
            raise ValueError(f"native UniFoLM G1 horizon must remain {ACTION_HORIZON}")
        if self.task_instruction != TASK_INSTRUCTION:
            raise ValueError("task instruction must remain frozen and exact")


SPEC = EmbodimentSpec()


def validate_joint_mapping(
    names: Sequence[str] = JOINT_NAMES,
    indices: Sequence[int] = MAPPED_INDICES,
) -> None:
    """Require the exact authoritative 28-name, 28-index bijection."""
    names_tuple = tuple(names)
    indices_tuple = tuple(int(index) for index in indices)
    if names_tuple != JOINT_NAMES:
        raise ValueError("joint names/order differ from the authoritative G1+Dex3 order")
    if indices_tuple != MAPPED_INDICES:
        raise ValueError("mapped indices must be exactly 0..27 in authoritative order")
    if len(names_tuple) != STATE_DIM or len(set(names_tuple)) != STATE_DIM:
        raise ValueError("joint mapping must contain 28 unique names")
    if len(indices_tuple) != STATE_DIM or len(set(indices_tuple)) != STATE_DIM:
        raise ValueError("joint mapping must contain 28 unique indices")


def vector_to_joint_dict(vector: np.ndarray | Sequence[float]) -> dict[str, float]:
    array = np.asarray(vector)
    if array.shape != (STATE_DIM,):
        raise ValueError(f"expected ({STATE_DIM},), got {array.shape}")
    return {name: array[index].item() for name, index in zip(JOINT_NAMES, MAPPED_INDICES)}


def joint_dict_to_vector(values: Mapping[str, float], dtype: np.dtype | None = None) -> np.ndarray:
    if set(values) != set(JOINT_NAMES):
        missing = sorted(set(JOINT_NAMES) - set(values))
        extra = sorted(set(values) - set(JOINT_NAMES))
        raise ValueError(f"joint mapping mismatch: missing={missing}, extra={extra}")
    return np.asarray([values[name] for name in JOINT_NAMES], dtype=dtype)


def _validated_stats(stats: Mapping[str, Any], dim: int = STATE_DIM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.asarray(stats["min"])
    high = np.asarray(stats["max"])
    mask = np.asarray(stats.get("mask", np.ones(dim, dtype=bool)), dtype=bool)
    if low.shape != (dim,) or high.shape != (dim,) or mask.shape != (dim,):
        raise ValueError(f"normalization stats and mask must each have shape ({dim},)")
    return low, high, mask


def normalize_bounds(values: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """Match the upstream BOUNDS normalization, including zero-range handling."""
    values = np.asarray(values)
    if values.shape[-1] != STATE_DIM:
        raise ValueError(f"last dimension must be {STATE_DIM}, got {values.shape}")
    low, high, mask = _validated_stats(stats)
    normalized = np.where(mask, np.clip(2 * (values - low) / (high - low + 1e-8) - 1, -1, 1), values)
    return np.where(low == high, 0.0, normalized)


def denormalize_bounds(values: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] != ACTION_DIM:
        raise ValueError(f"last dimension must be {ACTION_DIM}, got {values.shape}")
    low, high, mask = _validated_stats(stats)
    return np.where(mask, 0.5 * (values + 1) * (high - low + 1e-8) + low, values)


def make_loss_mask(
    batch_size: int,
    horizon: int = ACTION_HORIZON,
    valid_lengths: Iterable[int] | None = None,
) -> np.ndarray:
    """Return a [batch, horizon, 28] mask suitable for an elementwise action loss."""
    if batch_size <= 0 or horizon <= 0:
        raise ValueError("batch_size and horizon must be positive")
    mask = np.ones((batch_size, horizon, ACTION_DIM), dtype=bool)
    if valid_lengths is not None:
        lengths = tuple(int(length) for length in valid_lengths)
        if len(lengths) != batch_size or any(length < 0 or length > horizon for length in lengths):
            raise ValueError("valid_lengths must contain one value in [0, horizon] per batch item")
        for batch_index, length in enumerate(lengths):
            mask[batch_index, length:, :] = False
    return mask


def validate_model_io(state: Any, actions: Any, horizon: int = ACTION_HORIZON) -> None:
    state_shape = tuple(state.shape)
    action_shape = tuple(actions.shape)
    if state_shape[-1] != STATE_DIM:
        raise ValueError(f"state must end in {STATE_DIM}, got {state_shape}")
    if action_shape[-2:] != (horizon, ACTION_DIM):
        raise ValueError(f"actions must end in ({horizon}, {ACTION_DIM}), got {action_shape}")


def interface_parameter_audit(
    native_dim: int = 23,
    target_dim: int = ACTION_DIM,
    state_hidden: int = 1024,
    action_embedding: int = 1536,
    decoder_hidden: int = 1024,
) -> dict[str, int]:
    """Count the native and replacement dimension-tied interface parameters."""
    native = native_dim * (state_hidden + action_embedding + decoder_hidden) + native_dim
    target = target_dim * (state_hidden + action_embedding + decoder_hidden) + target_dim
    return {
        "native_dimension_tied_parameters": native,
        "new_28d_specific_parameters": target,
        "net_parameter_increase": target - native,
    }


def split_compatible_checkpoint_state_dict(
    state_dict: Mapping[str, Any], target_shapes: Mapping[str, Sequence[int]] = DIMENSION_TIED_CHECKPOINT_KEYS
) -> tuple[dict[str, Any], dict[str, dict[str, tuple[int, ...]]]]:
    """Drop only shape-incompatible interface tensors from a loaded checkpoint."""
    reusable: dict[str, Any] = {}
    replaced: dict[str, dict[str, tuple[int, ...]]] = {}
    for key, value in state_dict.items():
        actual_shape = tuple(value.shape)
        expected_shape = tuple(target_shapes[key]) if key in target_shapes else None
        if expected_shape is not None and actual_shape != expected_shape:
            replaced[key] = {"checkpoint_shape": actual_shape, "target_shape": expected_shape}
        else:
            reusable[key] = value
    return reusable, replaced


def load_reusable_checkpoint_weights(model: Any, checkpoint_path: str | Path) -> dict[str, Any]:
    """Mmap a checkpoint on CPU and load all tensors except the four 23D interface tensors.

    This function is prepared for the later authorized GPU smoke; it is not used
    by the CPU audit because loading the 19 GB checkpoint is unnecessary there.
    """
    import torch

    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", mmap=True, weights_only=True
    )
    reusable, replaced = split_compatible_checkpoint_state_dict(checkpoint)
    incompatible = model.load_state_dict(reusable, strict=False)
    missing = set(incompatible.missing_keys)
    expected_missing = set(replaced)
    if missing != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={incompatible.unexpected_keys}, expected_missing={sorted(expected_missing)}"
        )
    return {"replaced": replaced, "reused_tensor_count": len(reusable)}


def install_upstream_dimension_overrides(upstream_src: str | Path) -> None:
    """Install 28D globals before importing the upstream action head."""
    action_module = "unifolm_vla.model.modules.action_model.DiT_ActionHeader"
    if action_module in sys.modules:
        raise RuntimeError("dimension overrides must be installed before importing the upstream action head")
    upstream_src = str(Path(upstream_src).resolve())
    if upstream_src not in sys.path:
        sys.path.insert(0, upstream_src)
    constants = importlib.import_module("unifolm_vla.rlds_dataloader.constants")
    constants.ROBOT_PLATFORM = EMBODIMENT_NAME
    constants.NUM_ACTIONS_CHUNK = ACTION_HORIZON
    constants.ACTION_DIM = ACTION_DIM
    constants.PROPRIO_DIM = STATE_DIM
    constants.ACTION_PROPRIO_NORMALIZATION_TYPE = constants.NormalizationType.BOUNDS


def install_upstream_dataset_overrides(
    dataset_name: str = "g1_dex3_28d",
    statistics_path: str | Path = DEFAULT_STATISTICS_PATH,
) -> None:
    """Register the project RLDS schema without editing upstream registry files."""
    configs = importlib.import_module("unifolm_vla.rlds_dataloader.datasets.rlds.oxe.configs")
    transforms = importlib.import_module("unifolm_vla.rlds_dataloader.datasets.rlds.oxe.transforms")
    mixtures = importlib.import_module("unifolm_vla.rlds_dataloader.datasets.rlds.oxe.mixtures")
    materialize = importlib.import_module("unifolm_vla.rlds_dataloader.datasets.rlds.oxe.materialize")
    statistics_path = Path(statistics_path).resolve()
    statistics_payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    dataset_statistics = statistics_payload.get(dataset_name, statistics_payload)
    for field in ("action", "proprio"):
        if len(dataset_statistics[field]["mean"]) != STATE_DIM:
            raise ValueError(f"{field} statistics must contain {STATE_DIM} dimensions")

    def identity_transform(trajectory: Any) -> Any:
        return trajectory

    configs.OXE_DATASET_CONFIGS[dataset_name] = {
        "image_obs_keys": {"primary": "image", "wrist": None},
        "depth_obs_keys": {"primary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": configs.StateEncoding.JOINT_G1,
        "action_encoding": configs.ActionEncoding.JOINT_G1,
    }
    transforms.OXE_STANDARDIZATION_TRANSFORMS[dataset_name] = identity_transform
    mixtures.OXE_NAMED_MIXTURES[dataset_name] = [(dataset_name, 1.0)]

    original = materialize.make_oxe_dataset_kwargs

    def make_28d_kwargs(
        requested_name: str,
        data_root_dir: Path,
        load_camera_views: tuple[str, ...] = ("primary",),
        load_depth: bool = False,
        load_proprio: bool = True,
        load_language: bool = True,
        action_proprio_normalization_type: Any = None,
    ) -> dict[str, Any]:
        if requested_name != dataset_name:
            return original(
                requested_name,
                data_root_dir,
                load_camera_views,
                load_depth,
                load_proprio,
                load_language,
                action_proprio_normalization_type,
            )
        image_keys = {"primary": "image", "wrist": None}
        missing = set(load_camera_views) - set(image_keys)
        if missing:
            raise ValueError(f"unsupported camera views for {dataset_name}: {sorted(missing)}")
        kwargs: dict[str, Any] = {
            "name": dataset_name,
            "data_dir": str(data_root_dir),
            "image_obs_keys": {key: image_keys[key] for key in load_camera_views},
            "state_obs_keys": ["state"] if load_proprio else (),
            "absolute_action_mask": [True] * ACTION_DIM,
            "action_normalization_mask": [True] * ACTION_DIM,
            "action_proprio_normalization_type": action_proprio_normalization_type,
            "dataset_statistics": dataset_statistics,
            "standardize_fn": identity_transform,
        }
        if load_language:
            kwargs["language_key"] = "language_instruction"
        return kwargs

    materialize.make_oxe_dataset_kwargs = make_28d_kwargs


def install_exact_language_batch_transform() -> None:
    """Replace the upstream prompt paraphrase with the frozen dataset instruction."""
    datasets_module = importlib.import_module("unifolm_vla.rlds_dataloader.datasets.datasets")
    Image = importlib.import_module("PIL.Image")
    process_vision_info = importlib.import_module("qwen_vl_utils").process_vision_info

    def exact_call(self: Any, rlds_batch: Mapping[str, Any]) -> dict[str, Any]:
        actions = rlds_batch["action"]
        window_size = rlds_batch["observation"]["image_primary"].shape[0]
        images = [Image.fromarray(rlds_batch["observation"]["image_primary"][index]) for index in range(window_size)]
        instruction = rlds_batch["task"]["language_instruction"].decode("utf-8")
        if instruction != TASK_INSTRUCTION:
            raise ValueError(f"unexpected task instruction: {instruction!r}")
        messages = [{"role": "user", "content": [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": instruction},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(
            text=prompt, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        if window_size > 1:
            actions = actions[window_size - 1 :, :]
        batch_input["actions"] = actions
        batch_input["proprio"] = rlds_batch["observation"].get("proprio") if self.use_proprio else None
        return batch_input

    datasets_module.RLDSBatchTransform.__call__ = exact_call


def install_all_runtime_overrides(upstream_src: str | Path) -> None:
    install_upstream_dimension_overrides(upstream_src)
    install_upstream_dataset_overrides()
    install_exact_language_batch_transform()


def build_interface_shape_probe(device: str = "cpu") -> Any:
    """Construct the exact upstream dimension-bearing interface layers on CPU."""
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, value: Any) -> Any:
            return self.layer2(torch.relu(self.layer1(value)))

    class Probe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = MLP(STATE_DIM, 1024, 1536)
            self.action_input = nn.Linear(ACTION_DIM, 1536)
            self.action_decoder = MLP(1024, 1024, ACTION_DIM)

        def forward(self, state: Any, actions: Any) -> tuple[Any, Any]:
            validate_model_io(state, actions)
            state_features = self.state_encoder(state)
            action_features = self.action_input(actions)
            synthetic_dit_output = torch.zeros(
                (*actions.shape[:-1], 1024), dtype=actions.dtype, device=actions.device
            )
            output = self.action_decoder(synthetic_dit_output)
            validate_model_io(state, output)
            return state_features, action_features, output

    return Probe().to(device)


SPEC.validate()


if __name__ == "__main__":
    import json

    payload = {
        "spec": SPEC.__dict__,
        "interface_parameter_audit": interface_parameter_audit(),
        "joint_mapping": list(zip(MAPPED_INDICES, JOINT_NAMES)),
    }
    print(json.dumps(payload, indent=2))
