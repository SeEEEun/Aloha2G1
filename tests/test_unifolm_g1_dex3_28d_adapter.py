import numpy as np
import pytest

from tools.unifolm_g1_dex3_28d_adapter import (
    ACTION_DIM,
    ACTION_HORIZON,
    DIMENSION_TIED_CHECKPOINT_KEYS,
    JOINT_NAMES,
    MAPPED_INDICES,
    SPEC,
    build_interface_shape_probe,
    denormalize_bounds,
    interface_parameter_audit,
    joint_dict_to_vector,
    make_loss_mask,
    normalize_bounds,
    split_compatible_checkpoint_state_dict,
    validate_joint_mapping,
    vector_to_joint_dict,
)


def test_joint_contract_is_exact_unique_bijection():
    SPEC.validate()
    assert len(JOINT_NAMES) == len(set(JOINT_NAMES)) == 28
    assert MAPPED_INDICES == tuple(range(28))
    with pytest.raises(ValueError):
        validate_joint_mapping(JOINT_NAMES[:-1] + (JOINT_NAMES[0],), MAPPED_INDICES)


def test_joint_round_trip_is_bit_exact():
    vector = np.linspace(-1.25, 1.75, 28, dtype=np.float32)
    recovered = joint_dict_to_vector(vector_to_joint_dict(vector), dtype=np.float32)
    assert recovered.dtype == vector.dtype
    assert np.array_equal(recovered, vector)


def test_normalization_is_28d_and_round_trips_in_range():
    low = np.linspace(-2.0, -0.5, 28, dtype=np.float64)
    high = np.linspace(0.5, 2.0, 28, dtype=np.float64)
    values = np.stack((low, (low + high) / 2, high))
    stats = {"min": low, "max": high, "mask": np.ones(28, dtype=bool)}
    normalized = normalize_bounds(values, stats)
    recovered = denormalize_bounds(normalized, stats)
    assert normalized.shape == (3, 28)
    assert np.allclose(recovered, values, atol=1e-8)


def test_loss_mask_has_action_dimension_and_valid_lengths():
    mask = make_loss_mask(2, valid_lengths=(25, 7))
    assert mask.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert mask[0].all()
    assert mask[1, :7].all()
    assert not mask[1, 7:].any()


def test_checkpoint_split_replaces_only_native_interface_shapes():
    class TensorStub:
        def __init__(self, shape):
            self.shape = shape

    checkpoint = {"backbone.weight": TensorStub((4, 4))}
    for key, target_shape in DIMENSION_TIED_CHECKPOINT_KEYS.items():
        native_shape = tuple(23 if value == 28 else value for value in target_shape)
        checkpoint[key] = TensorStub(native_shape)
    reusable, replaced = split_compatible_checkpoint_state_dict(checkpoint)
    assert set(reusable) == {"backbone.weight"}
    assert set(replaced) == set(DIMENSION_TIED_CHECKPOINT_KEYS)
    assert interface_parameter_audit() == {
        "native_dimension_tied_parameters": 82455,
        "new_28d_specific_parameters": 100380,
        "net_parameter_increase": 17925,
    }


def test_cpu_interface_model_io_shape():
    torch = pytest.importorskip("torch")
    model = build_interface_shape_probe(device="cpu")
    state = torch.zeros((2, 1, 28), dtype=torch.float32)
    actions = torch.zeros((2, 25, 28), dtype=torch.float32)
    state_features, action_features, output = model(state, actions)
    assert state_features.shape == (2, 1, 1536)
    assert action_features.shape == (2, 25, 1536)
    assert output.shape == (2, 25, 28)
