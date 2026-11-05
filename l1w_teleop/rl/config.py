"""One deployment file selects a policy and its matching runtime contract."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .mapping import DEFAULT_MARGIN_M, ArmWorkspace
from .arm import gripper_settings
from .mapping import (
    ARM_POSITION_LOWER,
    ARM_POSITION_UPPER,
    RAW_POSITION_LOWER,
    RAW_POSITION_UPPER,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/rl/deploy.json"
MODEL = ROOT / "models/l1w_rl/policy_ppo_1028_19700.pt"
POLICY_TO_MOTOR = [4, 5, 6, 0, 1, 2, 8, 9, 10, 12, 13, 14, 7, 3, 11, 15]
POLICY_JOINT_NAMES = (
    "FBL_ABAD_JOINT",
    "FBL_HIP_JOINT",
    "FBL_KNEE_JOINT",
    "FAR_ABAD_JOINT",
    "FAR_HIP_JOINT",
    "FAR_KNEE_JOINT",
    "RAR_ABAD_JOINT",
    "RAR_HIP_JOINT",
    "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT",
    "RBL_HIP_JOINT",
    "RBL_KNEE_JOINT",
    "FBL_FOOT_JOINT",
    "FAR_FOOT_JOINT",
    "RAR_FOOT_JOINT",
    "RBL_FOOT_JOINT",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)

# Physical validation status is deliberately unchanged by file consolidation.
# Trial selection remains explicit; editing gains does not grant calibration.
VERIFICATION_KEYS = (
    "calibration_verified",
    "shutdown_verified",
    "imu_alignment_verified",
    "arm_alignment_verified",
)
REFERENCE_STATES = {
    "initial": "passive",
    "transition_seconds": 1.5,
    "hip_return_seconds": 0.5,
    "leg_kp": 60.0,
    "leg_kd": 2.0,
    "stand": "home",
    "wheel_kd": 20.0,
    "gain_transition_seconds": 0.3,
    "require_stand_settling": False,
    "settle_seconds": 0.3,
    "settle_position_tolerance_rad": 0.2,
    "settle_velocity_tolerance_rad_s": 0.5,
    "stand_timeout_seconds": 8.0,
}


def read_document(path=None):
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


def is_deployment(document):
    return "rl_gains" in document


def _check_document(document):
    expected = {
        "rl_gains",
        "stand",
        "rest",
        "mapping",
    }
    if not expected <= set(document) or set(document) - expected - {"model", "arm"}:
        raise ValueError(
            "deployment config requires its four settings sections and optional model/arm settings"
        )
    arm_settings(document)


def model_path(document=None):
    document = read_document() if document is None else document
    value = document.get("model", str(MODEL))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("deployment model must be a nonempty file path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.suffix not in (".pt", ".onnx"):
        raise ValueError("deployment model must be an exported .pt or .onnx file")
    if not path.is_file():
        raise ValueError(f"selected model does not exist: {path}")
    return path


def model_metadata(document=None):
    path = model_path(document).with_suffix(".json")
    if not path.is_file():
        raise ValueError(f"selected model metadata does not exist: {path}")
    metadata = json.loads(path.read_text())
    if (
        not isinstance(metadata, dict)
        or not {"policy_sha256", "runtime_contract", "body_reference"}
        <= metadata.keys()
    ):
        raise ValueError(f"selected model metadata requires a runtime contract: {path}")
    return metadata


def policy_settings(document):
    # Raw expanded documents are also accepted by offline/reference tests.
    if not is_deployment(document):
        return copy.deepcopy(document)
    _check_document(document)
    metadata = model_metadata(document)
    contract = copy.deepcopy(metadata["runtime_contract"])
    robot, algorithm = contract["robot"], contract["algorithm"]
    algo = metadata.get("algo", "ppo")
    embedded_normalization = metadata.get("embedded_normalization")
    if algo not in ("ppo", "him_ppo"):
        raise ValueError(f"unsupported deployment policy algorithm: {algo!r}")
    if not isinstance(embedded_normalization, bool):
        raise ValueError(  # noqa: TRY004 - Invalid metadata value.
            "policy metadata must declare embedded_normalization"
        )
    if algo == "ppo" and not embedded_normalization:
        raise ValueError("PPO policy requires embedded observation normalization")
    if algo == "him_ppo" and embedded_normalization:
        raise ValueError("HIM-PPO policy must consume raw observation history")
    gains = copy.deepcopy(document["rl_gains"])
    # Keep deployment defaults shared; per-algorithm overrides replace only
    # the gains explicitly listed for that policy family.
    per_algorithm = gains.pop("by_algorithm", {})
    if not isinstance(per_algorithm, dict):
        raise ValueError(  # noqa: TRY004 - Invalid deployment config value.
            "rl_gains.by_algorithm must be an object"
        )
    override = per_algorithm.get(algo, {})
    if not isinstance(override, dict):
        raise ValueError(  # noqa: TRY004 - Invalid deployment config value.
            f"rl_gains.by_algorithm.{algo} must be an object"
        )
    gains.update(override)
    for key in ("leg_kp", "leg_kd", "wheel_kd"):
        try:
            value = float(gains[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"rl_gains.{key} must be finite") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"rl_gains.{key} must be finite and nonnegative")
    robot["kp"] = [float(gains["leg_kp"])] * 12 + [0.0] * 4 + contract["arm_kp"]
    robot["kd"] = (
        [float(gains["leg_kd"])] * 12
        + [float(gains["wheel_kd"])] * 4
        + contract["arm_kd"]
    )
    robot["wheel_kv"] = [float(gains["wheel_kd"])] * 4
    algorithm.update(
        policy=str(model_path(document)),
        sha256=metadata["policy_sha256"],
        name=algo,
        embedded_normalization=embedded_normalization,
    )
    states = dict(REFERENCE_STATES)
    states["gain_transition_seconds"] = document["stand"]["gain_transition_seconds"]
    return {
        "robot": robot,
        "algorithm": algorithm,
        "states": states,
        "warmup": copy.deepcopy(contract["warmup"]),
        "arm": arm_settings(document),
    }


def mock_hardware_settings(document=None):
    """Identity coordinates for the simulated bridge; never physical calibration."""
    identity = {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}
    result = {
        **dict.fromkeys(VERIFICATION_KEYS, False),
        "policy_to_motor": POLICY_TO_MOTOR.copy(),
        "sign": [1] * 16,
        "offset": [0] * 16,
        "ratio": [1] * 16,
        "imu_quaternion_order": "wxyz",
        "imu_to_model_wxyz": [1, 0, 0, 0],
        "arm_default_for_mock": model_metadata(document)["runtime_contract"]["robot"][
            "default_dof_pos"
        ][16:],
        "arm_policy_to_sdk": list(range(6)),
        "arm_sign": [1] * 6,
        "arm_offset": [0] * 6,
        "arm_ratio": [1] * 6,
        "arm_raw_lower": None,
        "arm_raw_upper": None,
        "arm_sdk_base_to_policy": copy.deepcopy(identity),
        "arm_sdk_tool_to_policy_endpoint": copy.deepcopy(identity),
        "raw_lower": [-10.0] * 16,
        "raw_upper": [10.0] * 16,
    }
    for name in (
        "shutdown_raw_pose",
        "hold_kp",
        "hold_kd",
        "passive_kd",
    ):
        result[name] = None
    return result


def _matrix(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} requires a finite {shape} array")
    return array


def _with_wheel(legs, wheel):
    return np.column_stack((legs, np.broadcast_to(wheel, 4))).tolist()


def body_settings(document):
    if not is_deployment(document):
        return copy.deepcopy(document)
    _check_document(document)
    stand, rest = document["stand"], document["rest"]
    kp = _matrix(stand["leg_kp"], (4,), "stand.leg_kp")
    kd = _matrix(stand["leg_kd"], (4,), "stand.leg_kd")
    feedforward = _matrix(stand["wheel_feedforward"], (4,), "stand.wheel_feedforward")
    return {
        "version": 1,
        "coordinate_system": "raw_motor",
        "leg_order": ["FR", "FL", "RR", "RL"],
        "joint_order": ["abad", "hip", "knee", "foot"],
        "source": model_metadata(document)["body_reference"],
        "stand_raw_pose": _with_wheel(
            _matrix(stand["pose_raw"], (4, 3), "stand.pose_raw"), 0
        ),
        "lie_raw_pose": _with_wheel(
            _matrix(rest["pose_raw"], (4, 3), "rest.pose_raw"), 0
        ),
        "stand_raw_kp": _with_wheel(np.repeat(kp[:, None], 3, axis=1), 0),
        "stand_raw_kd": _with_wheel(
            np.repeat(kd[:, None], 3, axis=1), stand["wheel_kd"]
        ),
        "stand_raw_feedforward": _with_wheel(np.zeros((4, 3)), feedforward),
        "passive_raw_kd": np.full((4, 4), float(rest["kd"])).tolist(),
        "stand_seconds": float(stand["duration_seconds"]),
        "stand_hip_return_seconds": float(
            stand.get("hip_return_seconds", float(stand["duration_seconds"]) / 3)
        ),
        "stand_curve": "two_stage_cubic",
        "stand_timeout_seconds": max(15.0, float(stand["duration_seconds"]) + 1.0),
    }


def hardware_settings(document, path=None):
    if not is_deployment(document):
        return copy.deepcopy(document)
    _check_document(document)
    mapping = document["mapping"]
    result = mock_hardware_settings(document)
    result.update(
        calibration_status="candidate",
        trial_assumptions=[
            "Candidate leg mapping from recorded SDK/SHM data; ratio=1.",
            "Wheel raw forward signs +,-,+,- follow the vendor example.",
            "Physical IMU is xyzw; mounting and AIRBOT transforms remain unverified.",
            "Rest is the measured lying pose; trial disables arm commands.",
        ],
        sign=mapping["sign"],
        offset=mapping["offset"],
        imu_quaternion_order="xyzw",
        imu_to_model_wxyz=mapping["imu_mount_wxyz"],
        body_config=str(Path(path or DEFAULT_CONFIG).resolve()),
        raw_lower=list(RAW_POSITION_LOWER),
        raw_upper=list(RAW_POSITION_UPPER),
        arm_raw_lower=list(ARM_POSITION_LOWER),
        arm_raw_upper=list(ARM_POSITION_UPPER),
    )
    body = body_settings(document)
    for target, source in (
        ("shutdown_raw_pose", "lie_raw_pose"),
        ("hold_kp", "stand_raw_kp"),
        ("hold_kd", "stand_raw_kd"),
        ("passive_kd", "passive_raw_kd"),
    ):
        result[target] = np.asarray(body[source]).ravel().tolist()
    return result


def arm_settings(document):
    value = document.get("arm", {})
    if not isinstance(value, dict) or set(value) - {
        "workspace_margin_m",
        "return_zero_on_exit",
        "gripper",
    }:
        raise ValueError("invalid deployment arm settings")
    margin = value.get("workspace_margin_m", DEFAULT_MARGIN_M)
    ArmWorkspace(margin)
    returning = value.get("return_zero_on_exit", True)
    if not isinstance(returning, bool):
        raise ValueError("arm.return_zero_on_exit must be boolean")  # noqa: TRY004 - Invalid config value.
    return {
        "workspace_margin_m": float(margin),
        "return_zero_on_exit": returning,
        "gripper": gripper_settings(value.get("gripper")),
    }


def load_config(path=None, *, policy=None):
    document = read_document(path)
    if policy is not None:
        if not is_deployment(document):
            raise ValueError("--policy requires a single deployment config")
        # Select the model and its sidecar together, before contract/hash checks.
        # This only changes the in-memory settings, never the deployment file.
        document["model"] = str(policy)
    cfg = policy_settings(document)
    a, r = cfg["algorithm"], cfg["robot"]
    if (a["num_one_step_obs"], a["num_actor_history"], a["num_actions"]) != (88, 5, 22):
        raise ValueError(
            "this deployment supports only the verified 88 x 5 -> 22 contract"
        )
    if (r["leg_slice"], r["wheel_slice"], r["arm_slice"]) != (
        [0, 12],
        [12, 16],
        [16, 22],
    ):
        raise ValueError("invalid policy joint slices")
    if float(r["ctrl_dt"]) != 0.02:
        raise ValueError("verified policy requires 20 ms control period")
    if a.get("name") not in ("ppo", "him_ppo"):
        raise ValueError("unsupported policy algorithm")
    if not isinstance(a.get("embedded_normalization"), bool):
        raise ValueError(  # noqa: TRY004 - Invalid metadata value.
            "policy normalization contract is missing"
        )
    states = cfg["states"]
    if states["initial"] not in ("passive", "stand", "locomotion"):
        raise ValueError("invalid initial FSM state")
    if not isinstance(states.get("require_stand_settling", True), bool):
        raise ValueError("require_stand_settling must be boolean")  # noqa: TRY004 - Invalid config value.
    durations = np.asarray(
        [
            states["hip_return_seconds"],
            states["transition_seconds"],
            states.get("gain_transition_seconds", 0.3),
            states.get("settle_seconds", 0.3),
            states.get("stand_timeout_seconds", 8),
        ],
        dtype=float,
    )
    if not np.isfinite(durations).all() or not (
        0 < durations[0] < durations[1] < durations[4]
        and 0 < durations[2] <= durations[1]
        and durations[3] > 0
    ):
        raise ValueError("invalid two-stage stand timing")
    tolerances = np.asarray(
        [
            states.get("settle_position_tolerance_rad", 0.2),
            states.get("settle_velocity_tolerance_rad_s", 0.5),
        ],
        dtype=float,
    )
    if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
        raise ValueError("stand settling tolerances must be finite and positive")
    gains = np.asarray(
        [states["leg_kp"], states["leg_kd"], states["wheel_kd"]], dtype=float
    )
    if not np.isfinite(gains).all() or np.any(gains < 0):
        raise ValueError("invalid stand gains")
    if len(r["joint_names"]) != 22 or len(set(r["joint_names"])) != 22:
        raise ValueError("policy requires 22 unique joint names")
    if tuple(r["joint_names"]) != POLICY_JOINT_NAMES:
        raise ValueError("policy joint order differs from the L1-W motor mapping")
    if r.get("ignore_wheel_position") is not True:
        raise ValueError(
            "policy must use the supported wheel-position-free observation"
        )
    for key, size in (
        ("default_dof_pos", 22),
        ("kp", 22),
        ("kd", 22),
        ("wheel_kv", 4),
        ("feet_phase_offsets", 4),
        ("ee_goal", 3),
        ("ee_goal_quaternion", 4),
    ):
        array = np.asarray(r[key], dtype=float)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"invalid robot.{key}")
    gait_scalars = np.asarray(
        [
            r["gait_frequency"],
            r["command_deadzone"],
            r.get("gait_command_deadzone", r["command_deadzone"]),
        ],
        dtype=float,
    )
    if not np.isfinite(gait_scalars).all() or np.any(gait_scalars < 0):
        raise ValueError("invalid gait frequency/deadzone")
    if "gait_activation_thresholds" in r:
        thresholds = np.asarray(r["gait_activation_thresholds"], dtype=float)
        if (
            thresholds.shape != (3,)
            or not np.isfinite(thresholds).all()
            or np.any(thresholds < 0)
        ):
            raise ValueError("invalid gait activation thresholds")
    ranges = np.asarray(r["ctrl_ranges"], dtype=float)
    if (
        ranges.shape != (22, 2)
        or not np.isfinite(ranges).all()
        or np.any(ranges[:, 0] > ranges[:, 1])
    ):
        raise ValueError("invalid policy actuator limits")
    model = Path(a["policy"])
    if not model.is_absolute():
        model = ROOT / model
    if hashlib.sha256(model.read_bytes()).hexdigest() != a["sha256"]:
        raise ValueError("policy SHA-256 mismatch")
    a["policy"] = str(model)
    return cfg
