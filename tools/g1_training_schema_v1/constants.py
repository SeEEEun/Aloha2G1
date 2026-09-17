from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_VERSION = "g1_training_schema_v1"
LEROBOT_VERSION = "v3.0"
FPS = 30.0
IMAGE_KEY = "observation.images.cam_high"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_TEXT = (
    "Remove the MagSafe accessory from the phone and place the phone on the MagSafe charger."
)
CHUNK_SIZE = 50
STATE_DIM = 28
ACTION_DIM = 28


@dataclass(frozen=True)
class JointSpec:
    index: int
    joint_name: str
    side: str
    group: str
    unit: str
    minimum: float
    maximum: float
    source_of_truth: list[str]
    command_channel: str


ARM_SOURCES = [
    "/home/jbnu/jaeyoung/unitree/unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py",
    "/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml",
]
HAND_SOURCES = [
    "/home/jbnu/jaeyoung/unitree/unitree_sdk2/example/g1/dex3/g1_dex3_example.cpp",
    "/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml",
    "/home/jbnu/aloha_g1_dataset/tools/record_g1_dex3_magsafe_primitives.py",
]


def _joint(
    index: int,
    name: str,
    side: str,
    group: str,
    limits: tuple[float, float],
    sources: list[str],
    channel: str,
) -> JointSpec:
    return JointSpec(index, name, side, group, "radian", *limits, sources, channel)


JOINT_SPECS = (
    _joint(0, "left_shoulder_pitch_joint", "left", "arm", (-3.0892, 2.6704), ARM_SOURCES, "G1 arm motor 15"),
    _joint(1, "left_shoulder_roll_joint", "left", "arm", (-1.5882, 2.2515), ARM_SOURCES, "G1 arm motor 16"),
    _joint(2, "left_shoulder_yaw_joint", "left", "arm", (-2.618, 2.618), ARM_SOURCES, "G1 arm motor 17"),
    _joint(3, "left_elbow_joint", "left", "arm", (-1.0472, 2.0944), ARM_SOURCES, "G1 arm motor 18"),
    _joint(4, "left_wrist_roll_joint", "left", "arm", (-1.97222, 1.97222), ARM_SOURCES, "G1 arm motor 19"),
    _joint(5, "left_wrist_pitch_joint", "left", "arm", (-1.61443, 1.61443), ARM_SOURCES, "G1 arm motor 20"),
    _joint(6, "left_wrist_yaw_joint", "left", "arm", (-1.61443, 1.61443), ARM_SOURCES, "G1 arm motor 21"),
    _joint(7, "right_shoulder_pitch_joint", "right", "arm", (-3.0892, 2.6704), ARM_SOURCES, "G1 arm motor 22"),
    _joint(8, "right_shoulder_roll_joint", "right", "arm", (-2.2515, 1.5882), ARM_SOURCES, "G1 arm motor 23"),
    _joint(9, "right_shoulder_yaw_joint", "right", "arm", (-2.618, 2.618), ARM_SOURCES, "G1 arm motor 24"),
    _joint(10, "right_elbow_joint", "right", "arm", (-1.0472, 2.0944), ARM_SOURCES, "G1 arm motor 25"),
    _joint(11, "right_wrist_roll_joint", "right", "arm", (-1.97222, 1.97222), ARM_SOURCES, "G1 arm motor 26"),
    _joint(12, "right_wrist_pitch_joint", "right", "arm", (-1.61443, 1.61443), ARM_SOURCES, "G1 arm motor 27"),
    _joint(13, "right_wrist_yaw_joint", "right", "arm", (-1.61443, 1.61443), ARM_SOURCES, "G1 arm motor 28"),
    _joint(14, "left_hand_thumb_0_joint", "left", "dex3", (-1.0472, 1.0472), HAND_SOURCES, "left Dex3 motor 0"),
    _joint(15, "left_hand_thumb_1_joint", "left", "dex3", (-0.724312, 1.0472), HAND_SOURCES, "left Dex3 motor 1"),
    _joint(16, "left_hand_thumb_2_joint", "left", "dex3", (0.0, 1.74533), HAND_SOURCES, "left Dex3 motor 2"),
    _joint(17, "left_hand_middle_0_joint", "left", "dex3", (-1.5708, 0.0), HAND_SOURCES, "left Dex3 motor 3"),
    _joint(18, "left_hand_middle_1_joint", "left", "dex3", (-1.74533, 0.0), HAND_SOURCES, "left Dex3 motor 4"),
    _joint(19, "left_hand_index_0_joint", "left", "dex3", (-1.5708, 0.0), HAND_SOURCES, "left Dex3 motor 5"),
    _joint(20, "left_hand_index_1_joint", "left", "dex3", (-1.74533, 0.0), HAND_SOURCES, "left Dex3 motor 6"),
    _joint(21, "right_hand_thumb_0_joint", "right", "dex3", (-1.0472, 1.0472), HAND_SOURCES, "right Dex3 motor 0"),
    _joint(22, "right_hand_thumb_1_joint", "right", "dex3", (-1.0472, 0.724312), HAND_SOURCES, "right Dex3 motor 1"),
    _joint(23, "right_hand_thumb_2_joint", "right", "dex3", (-1.74533, 0.0), HAND_SOURCES, "right Dex3 motor 2"),
    _joint(24, "right_hand_middle_0_joint", "right", "dex3", (0.0, 1.5708), HAND_SOURCES, "right Dex3 motor 3"),
    _joint(25, "right_hand_middle_1_joint", "right", "dex3", (0.0, 1.74533), HAND_SOURCES, "right Dex3 motor 4"),
    _joint(26, "right_hand_index_0_joint", "right", "dex3", (0.0, 1.5708), HAND_SOURCES, "right Dex3 motor 5"),
    _joint(27, "right_hand_index_1_joint", "right", "dex3", (0.0, 1.74533), HAND_SOURCES, "right Dex3 motor 6"),
)

CANONICAL_JOINT_NAMES = tuple(j.joint_name for j in JOINT_SPECS)


def joint_order_json() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dimension": len(JOINT_SPECS),
        "semantic_order": "left_arm_7 + right_arm_7 + left_Dex3_DDS_7 + right_Dex3_DDS_7",
        "joints": [asdict(j) for j in JOINT_SPECS],
    }


def policy_features(camera_feature: dict | None = None) -> dict:
    camera = camera_feature or {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    return {
        IMAGE_KEY: camera,
        STATE_KEY: {"dtype": "float32", "shape": [STATE_DIM], "names": list(CANONICAL_JOINT_NAMES)},
        ACTION_KEY: {"dtype": "float32", "shape": [ACTION_DIM], "names": list(CANONICAL_JOINT_NAMES)},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
