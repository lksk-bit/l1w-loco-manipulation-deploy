"""Joint/pose coordinate conversions, workspace and position bounds."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from .bridge import MOCK

LEG_POSITION_LOWER = (
    (-0.0503, -4.169, -0.05),
    (-1.03592, -0.05, -2.171),
    (-1.04241, -4.169, -0.05),
    (-0.0503, -0.05, -2.171),
)
LEG_POSITION_UPPER = (
    (1.03249, 0.05, 2.171),
    (0.0497, 4.169, 0.05),
    (0.0503, 0.05, 2.171),
    (1.03325, 4.169, 0.05),
)

# Wheel position and Kp remain zero; wheels operate in velocity mode.
RAW_POSITION_LOWER = tuple(value for row in LEG_POSITION_LOWER for value in (*row, 0))
RAW_POSITION_UPPER = tuple(value for row in LEG_POSITION_UPPER for value in (*row, 0))

# AIRBOT play_g2 service's URDF limits, joint1..joint6 in SDK coordinates.
# Source: KinematicsInterface joints_limit in the user's 2026-10-20 service log,
# /usr/share/airbot_controllers/asset/play_arm/urdf/play_g2.urdf.
# These fixed hardware bounds must not change when a policy model is selected.
# Frame/joint alignment validation remains separate from these position bounds.
ARM_POSITION_LOWER = (-3.1416, -2.9671, -0.087266, -3.0107, -1.7628, -3.0107)
ARM_POSITION_UPPER = (2.0944, 0.17453, 3.1416, 3.0107, 1.7628, 3.0107)

def quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

def orientation_error(goal: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Canonical xyz of goal * inverse(current), in wxyz convention."""
    a = np.asarray(goal) / np.linalg.norm(goal)
    b = np.asarray(current) / np.linalg.norm(current)
    w = a[0] * b[0] + np.dot(a[1:], b[1:])
    xyz = -a[0] * b[1:] + b[0] * a[1:] - np.cross(a[1:], b[1:])
    return xyz if w >= 0 else -xyz

def quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("invalid arm pose quaternion")
    return q / np.linalg.norm(q)


def multiply(a, b):
    """Hamilton product, wxyz."""
    w, x, y, z = a
    v, i, j, k = b
    return np.array(
        [
            w * v - x * i - y * j - z * k,
            w * i + x * v + y * k - z * j,
            w * j - x * k + y * v + z * i,
            w * k + x * j - y * i + z * v,
        ]
    )


def conjugate(q):
    return q * np.array([1, -1, -1, -1])


class ArmPoseMapping:
    """T_policy_endpoint = T_policy_sdkbase T_sdkbase_sdktool T_sdktool_endpoint."""

    def __init__(self, config):

        self.base_position, self.base_quat = self._transform(
            config, "arm_sdk_base_to_policy"
        )
        self.tool_position, self.tool_quat = self._transform(
            config, "arm_sdk_tool_to_policy_endpoint"
        )
        self.base_rotation = quat_matrix(self.base_quat)
        self.quat_matrix = quat_matrix

    @staticmethod
    def _transform(config, name):
        value = config.get(name)
        if not isinstance(value, dict):
            raise ValueError(  # noqa: TRY004 - Missing configuration is a validation error.
                f"{name} requires position and quaternion_wxyz; update hardware profile"
            )
        p = np.asarray(value.get("position"), dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError(f"invalid {name}.position")
        return p, quaternion(value.get("quaternion_wxyz"))

    @staticmethod
    def _position(value):
        p = np.asarray(value, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("invalid AIRBOT Cartesian position")
        return p

    def read(self, position, quaternion_xyzw):
        p = self._position(position)
        q = quaternion(quaternion_xyzw)[[3, 0, 1, 2]]
        output_p = self.base_position + self.base_rotation @ (
            p + self.quat_matrix(q) @ self.tool_position
        )
        output_q = multiply(multiply(self.base_quat, q), self.tool_quat)
        return {"ee_position": output_p, "ee_quaternion": quaternion(output_q)}

    def write(self, position, quaternion_wxyz):
        p = self._position(position)
        q = quaternion(quaternion_wxyz)
        sdk_q = multiply(
            multiply(conjugate(self.base_quat), q), conjugate(self.tool_quat)
        )
        sdk_p = (
            self.base_rotation.T @ (p - self.base_position)
            - self.quat_matrix(sdk_q) @ self.tool_position
        )
        return sdk_p, quaternion(sdk_q)[[1, 2, 3, 0]]

DEFAULT_MARGIN_M = 0.02


class ArmWorkspace:
    SHOULDER_HEIGHT_M = 0.1117
    UPPER_ARM_M = 0.27009
    FOREARM_M = 0.29015
    # end_convert translation [0,0,.2466] expressed in the SDK tool axes:
    # R_end.T @ [0,0,.2466] = [.2466,0,0].
    TOOL_OFFSET_M = (0.2466, 0, 0)

    def __init__(self, margin_m=DEFAULT_MARGIN_M):
        self.margin = float(margin_m)
        self.minimum = abs(self.FOREARM_M - self.UPPER_ARM_M) + self.margin
        self.maximum = self.UPPER_ARM_M + self.FOREARM_M - self.margin
        if (
            not np.isfinite(self.margin)
            or self.margin < 0
            or self.minimum >= self.maximum
        ):
            raise ValueError("invalid AIRBOT workspace margin")
        self.shoulder = np.array([0, 0, self.SHOULDER_HEIGHT_M])

    def limit(self, position, quaternion_xyzw, reference_position=None):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("invalid AIRBOT workspace position")
        rotation = quat_matrix(quaternion(quaternion_xyzw)[[3, 0, 1, 2]])
        tool = rotation @ self.TOOL_OFFSET_M
        delta = position - tool - self.shoulder
        radius = float(np.linalg.norm(delta))
        bounded = float(np.clip(radius, self.minimum, self.maximum))
        if bounded == radius:
            return position.copy(), False
        if radius > 1e-10:
            direction = delta / radius
        else:
            reference = (
                position
                if reference_position is None
                else np.asarray(reference_position, dtype=float)
            )
            if reference.shape != (3,) or not np.isfinite(reference).all():
                raise ValueError("invalid AIRBOT workspace reference")
            direction = reference - tool - self.shoulder
            length = np.linalg.norm(direction)
            direction = direction / length if length > 1e-10 else np.array([1.0, 0, 0])
        return self.shoulder + bounded * direction + tool, True

PROFILE_MAGIC = 0x31504652
PROFILE_VERSION = 3


class HardwareMapping:
    def __init__(self, path=None):
        # Config validates against the pure bounds above. Defer its loader
        # until construction so either module can be imported independently.
        from .config import DEFAULT_CONFIG, hardware_settings

        if isinstance(path, dict):
            data = json.dumps(path, sort_keys=True).encode()
            document = path
        else:
            data = Path(path or DEFAULT_CONFIG).read_bytes()
            document = json.loads(data)
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.config = hardware_settings(
            document, None if isinstance(path, dict) else path
        )
        c = self.config
        self.arm_pose = ArmPoseMapping(c)
        self.indices = np.asarray(c["policy_to_motor"], dtype=int)
        if sorted(self.indices.tolist()) != list(range(16)):
            raise ValueError("policy_to_motor must be a permutation of 0..15")
        self.sign = self.vector("sign")
        if not np.all(np.isin(self.sign, [-1, 1])):
            raise ValueError("sign must contain only -1/+1")
        self.offset, self.ratio = self.vector("offset"), self.vector("ratio")
        if np.any(self.ratio <= 0):
            raise ValueError("ratio must be positive")
        self.factor = self.sign * self.ratio
        self.arm_indices = np.asarray(
            c.get("arm_policy_to_sdk", list(range(6))), dtype=int
        )
        if sorted(self.arm_indices.tolist()) != list(range(6)):
            raise ValueError("arm_policy_to_sdk must be a permutation of 0..5")
        self.arm_sign = np.asarray(c.get("arm_sign", [1] * 6), dtype=float)
        self.arm_offset = np.asarray(c.get("arm_offset", [0] * 6), dtype=float)
        self.arm_ratio = np.asarray(c.get("arm_ratio", [1] * 6), dtype=float)
        if (
            any(
                a.shape != (6,) or not np.isfinite(a).all()
                for a in (self.arm_sign, self.arm_offset, self.arm_ratio)
            )
            or not np.all(np.isin(self.arm_sign, [-1, 1]))
            or np.any(self.arm_ratio <= 0)
        ):
            raise ValueError("invalid arm joint mapping")
        self.arm_factor = self.arm_sign * self.arm_ratio
        self.imu_quaternion_order = c.get("imu_quaternion_order", "wxyz")
        if self.imu_quaternion_order not in ("wxyz", "xyzw"):
            raise ValueError("imu_quaternion_order must be wxyz or xyzw")
        quaternion = np.asarray(c["imu_to_model_wxyz"], dtype=float)
        if (
            quaternion.shape != (4,)
            or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) < 1e-8
        ):
            raise ValueError("invalid IMU mounting quaternion")
        self.imu_rotation = quat_matrix(quaternion)
        if not np.isfinite(self.imu_rotation).all():
            raise ValueError("invalid IMU rotation")

    def vector(self, key):
        value = np.asarray(self.config[key], dtype=float)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError(f"{key} requires 16 finite values")
        return value

    def require_calibration(self, *, allow_trial=False):
        # Legacy API name retained for callers; metadata is not an enable gate.
        # Validate executable data without claiming physical calibration.
        self.profile_bytes()

    def imu_order(self, frame):
        # Mock bridge synthesizes wxyz; physical bridge copies firmware SHM verbatim.
        return (
            "wxyz" if getattr(frame, "flags", 0) & MOCK else self.imu_quaternion_order
        )

    def read(self, frame):
        q = (np.asarray(frame.q)[self.indices] - self.offset) / self.factor
        dq = np.asarray(frame.dq)[self.indices] / self.factor
        quaternion = np.asarray(frame.quaternion, dtype=float)
        if self.imu_order(frame) == "xyzw":
            quaternion = quaternion[[3, 0, 1, 2]]
        # Reorder once at the physical input boundary, before mounting rotation.
        # IMU quaternion describes sensor->world; model->world uses inverse mounting rotation.
        world_model = quat_matrix(quaternion) @ self.imu_rotation.T
        # Training observes world up expressed in the model frame.  For a
        # model-to-world rotation this is R.T @ world_up (row 2 of R), not
        # the model's up axis expressed in world coordinates (column 2).
        gravity = -world_model[2, :]
        gyro = self.imu_rotation @ np.asarray(frame.gyro)
        return q, dq, gyro, gravity

    def read_arm(self, q, dq):
        return (
            np.asarray(q)[self.arm_indices] - self.arm_offset
        ) / self.arm_factor, np.asarray(dq)[self.arm_indices] / self.arm_factor

    def require_arm_control(self):
        lo = np.asarray(self.config.get("arm_raw_lower"), dtype=float)
        hi = np.asarray(self.config.get("arm_raw_upper"), dtype=float)
        if (
            lo.shape != (6,)
            or hi.shape != (6,)
            or not np.isfinite([lo, hi]).all()
            or np.any(lo >= hi)
        ):
            raise ValueError(
                "arm control requires six finite arm_raw_lower/arm_raw_upper values with lower < upper"
            )

    def write_arm(self, q):
        self.require_arm_control()
        raw = np.zeros(6)
        raw[self.arm_indices] = np.asarray(q) * self.arm_factor + self.arm_offset
        if (
            not np.isfinite(raw).all()
            or np.any(raw < self.config["arm_raw_lower"])
            or np.any(raw > self.config["arm_raw_upper"])
        ):
            raise ValueError("arm target exceeds hardware limits")
        return raw

    def write(self, q, dq, kp, kd):
        result = np.zeros((16, 5), dtype=np.float32)
        result[self.indices, 0] = np.asarray(q[:16]) * self.factor + self.offset
        result[self.indices, 1] = np.asarray(dq[:16]) * self.factor
        # Mapping ratios change both target coordinates and equivalent motor PD stiffness.
        result[self.indices, 2] = np.asarray(kp[:16]) / self.ratio**2
        result[self.indices, 3] = np.asarray(kd[:16]) / self.ratio**2
        # Keep raw observations/action history intact; clip only executable positions.
        if (
            self.config.get("raw_lower") is not None
            and self.config.get("raw_upper") is not None
        ):
            result[:, 0] = np.clip(
                result[:, 0], self.vector("raw_lower"), self.vector("raw_upper")
            )
        result[[3, 7, 11, 15], 0] = 0
        result[[3, 7, 11, 15], 2] = 0
        return result

    def validate_targets(self, targets):
        t = np.asarray(targets)
        if t.shape != (16, 5) or not np.isfinite(t).all():
            raise ValueError("invalid mapped targets")
        if np.any(t[:, 2:4] < 0):
            raise ValueError("mapped gains must be nonnegative")
        if np.any(t[3::4, 2] != 0):
            raise ValueError("wheel position Kp must be zero")

    def profile_bytes(self):
        keys = [
            "raw_lower",
            "raw_upper",
            "shutdown_raw_pose",
            "hold_kp",
            "hold_kd",
            "passive_kd",
        ]
        vectors = {key: self.vector(key) for key in keys}
        legs = np.arange(16) % 4 != 3
        lo, hi, rest = (
            vectors["raw_lower"],
            vectors["raw_upper"],
            vectors["shutdown_raw_pose"],
        )
        if (
            np.any(lo[legs] >= hi[legs])
            or np.any(rest[legs] < lo[legs])
            or np.any(rest[legs] > hi[legs])
        ):
            raise ValueError("invalid leg soft bounds/shutdown pose")
        for key, value in vectors.items():
            if key not in ("raw_lower", "raw_upper", "shutdown_raw_pose") and np.any(
                value < 0
            ):
                raise ValueError(f"{key} must be nonnegative")
        wheels = np.zeros(16, dtype=np.uint32)
        wheels[[3, 7, 11, 15]] = 1
        if np.any(vectors["hold_kp"][wheels.astype(bool)] != 0):
            raise ValueError("wheel hold Kp must be zero")
        size = 16 + len(keys) * 64 + 64
        return (
            struct.pack("<IIII", PROFILE_MAGIC, PROFILE_VERSION, size, 1)
            + b"".join(struct.pack("<16f", *a) for a in vectors.values())
            + struct.pack("<16I", *wheels)
        )
