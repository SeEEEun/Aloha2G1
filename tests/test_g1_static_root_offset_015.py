import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/jbnu/aloha_g1_dataset')
LAYOUT = ROOT / 'isaaclab_magsafe_fixed_scene/scene_layout.json'
POSES = ROOT / 'isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json'
PREVIEW = ROOT / 'isaaclab_magsafe_fixed_scene/preview_magsafe_g1_model.py'
REGISTRATION = ROOT / 'configs/magsafe_task_frame_registration.sim.json'


def preview_default_offset():
    tree = ast.parse(PREVIEW.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--root-forward-offset-m':
                return next(k.value.value for k in node.keywords if k.arg == 'default')
    raise AssertionError('preview root offset argument not found')


def test_objects_unchanged_and_original_root_preserved():
    layout = json.loads(LAYOUT.read_text()); poses = json.loads(POSES.read_text())
    assert layout['phone']['bottom_left_xy'] == [0.45, 0.07]
    assert layout['phone']['bottom_right_xy'] == [0.6, 0.07]
    assert layout['charger']['center_xy'] == [0.42, 0.21]
    assert poses['g1']['position_xyz_m'] == [0.4175, -0.5, 0.7922728583]


def test_preview_and_registration_use_final_total_offset_once():
    reg = json.loads(REGISTRATION.read_text())
    assert preview_default_offset() == 0.15
    assert reg['manual_adjustment_log'][0]['value_m'] == 0.15
    original = np.asarray(reg['manual_adjustment_log'][0]['original_root_position_m'])
    applied = np.asarray(reg['manual_adjustment_log'][0]['applied_root_position_m'])
    assert np.isclose(np.linalg.norm((applied-original)[:2]), 0.15)
    assert applied[2] == original[2]
    assert np.allclose(applied, np.asarray(reg['T_scene_from_g1_base'])[:3, 3])


def test_no_lateral_z_or_yaw_adjustment():
    reg = json.loads(REGISTRATION.read_text())
    original = np.asarray(reg['manual_adjustment_log'][0]['original_root_position_m'])
    applied = np.asarray(reg['manual_adjustment_log'][0]['applied_root_position_m'])
    forward = (applied-original) / 0.15
    assert np.isclose(np.linalg.norm(forward[:2]), 1.0)
    assert forward[2] == 0.0
    assert np.allclose(np.asarray(reg['T_scene_from_g1_base'])[:3, :3], [[0,-1,0],[1,0,0],[0,0,1]], atol=1e-9)
