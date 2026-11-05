"""AIRBOT adapter and session worker; native SDK loading stays lazy."""

import math
import json
import os
import select
import sys
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from .mapping import quaternion, DEFAULT_MARGIN_M, ArmWorkspace
from .arm import (
    ArmSnapshot,
    GRIPPER_MAX_LEAD_M,
    GripperControl,
    gripper_settings,
    STATUS_FD_ENV,
)
from .bridge import CONTROL_TIMEOUT_NS


class ArmClient(ABC):
    """Boundary for the AIRBOT official SDK client."""

    @abstractmethod
    def hold(self) -> None:
        """Keep the arm in its current commanded pose."""

    @abstractmethod
    def get_end_pose(self):
        """Return ((x, y, z), (qx, qy, qz, qw)), or None if unavailable."""

    @abstractmethod
    def send_cartesian_velocity(
        self, vx: float, vy: float, vz: float, gripper: float
    ) -> None:
        """Send a bounded Cartesian velocity and gripper target."""

    @abstractmethod
    def send_cartesian_pose(self, position, orientation, gripper: float) -> None:
        """Send a Cartesian pose and gripper target."""

    @abstractmethod
    def return_zero(self) -> None:
        """Return the arm to its configured home pose."""

    @abstractmethod
    def move_joint(
        self, angles, blocking: bool = False, timeout_ms: int = 5000
    ) -> bool:
        """Command a joint-space target."""


class AirbotArmClient(ArmClient):
    """AIRBOT SDK adapter.

    Construction and get_end_pose are read-only. A caller must explicitly call
    enable_control before any motion method can reach the arm.
    """

    G2_GRIPPER_RANGE = (0.0, 0.072)
    POSE_COMMAND_EPSILON_M = 0.001
    POSE_COMMAND_EPSILON_RAD = 0.02
    REJECTED_POSE_RETRY_S = 0.5

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        arm_dof: int = 6,
    ):
        from arm_sdk import AirbotClient, ArmControlOptions, CartesianPose

        self._cartesian_pose_type = CartesianPose
        self._options_type = ArmControlOptions
        self._client = AirbotClient(host=host, port=port, arm_dof=arm_dof)
        self._control_enabled = False
        self._last_pose = None
        self._last_gripper = None
        self._hold_pose = None
        self._hold_gripper = None
        self._hold_command_active = False
        self._last_command_rejected = False
        self._rejected_at = None

    def enable_control(self) -> None:
        from arm_sdk import Controller

        if self._control_enabled:
            return
        if not self._client.acquire_control():
            raise RuntimeError("failed to acquire AIRBOT control lease")
        if not self._client.switch_controller(Controller.servo_control):
            raise RuntimeError("failed to switch AIRBOT to servo control")
        self._control_enabled = True
        self._hold_command_active = False

    def current_gripper(self) -> float:
        state = self._client.get_eef_joint_state()
        if state is None:
            return 0.036
        return self._clamped_gripper(float(state.eef_pos))

    def get_gripper_feedback(self):
        """Strict SDK position/velocity/effort; no cached or guessed feedback."""
        state = self._client.get_eef_joint_state()
        if state is None:
            return None
        values = tuple(
            float(getattr(state, key)) for key in ("eef_pos", "eef_vel", "eef_eff")
        )
        if not all(math.isfinite(v) for v in values):
            return None
        return values

    def configure_gripper(self, effort_limit: float) -> None:
        limit = float(effort_limit)
        if not math.isfinite(limit) or not 0 < limit <= 20:
            raise ValueError("invalid SDK gripper effort limit")
        self._gripper_effort_limit = limit

    def send_gripper_target(self, position: float) -> bool:
        """Independent, nonblocking servo EEF command with native effort request."""
        self._require_control()
        if not math.isfinite(position):
            raise ValueError("invalid gripper target")
        position = self._clamped_gripper(position)
        options = self._options_type()
        options.eef_pos = position
        options.eef_eff = getattr(self, "_gripper_effort_limit", 6.0)
        options.blocking = False
        accepted = self._client.move_eef(position, options, timeout_ms=100)
        if accepted:
            self._last_gripper = self._hold_gripper = position
        return accepted

    def hold(self) -> None:
        if self._hold_command_active:
            return
        pose = self._hold_pose or self.get_end_pose() or self._last_pose
        if pose is None:
            return
        gripper = self._hold_gripper
        if gripper is None:
            gripper = self.current_gripper()
        self._send_pose(pose[0], pose[1], gripper)
        self._hold_pose = (
            tuple(float(value) for value in pose[0]),
            tuple(float(value) for value in pose[1]),
        )
        self._hold_gripper = gripper
        self._hold_command_active = True

    def get_end_pose(self):
        return self.get_end_pose_feedback() or self._last_pose

    def get_end_pose_feedback(self):
        """Read SDK feedback without substituting a cached pose on failure."""
        pose = self._client.get_end_pose()
        if pose is None:
            return None
        result = (
            tuple(float(value) for value in pose.position),
            tuple(float(value) for value in pose.orientation),
        )
        self._last_pose = result
        return result

    def get_joint_angles(self):
        state = self._client.get_arm_joint_state()
        if state is None:
            return None
        return tuple(float(value) for value in state.angles)

    def send_cartesian_velocity(
        self, vx: float, vy: float, vz: float, gripper: float
    ) -> None:
        raise NotImplementedError("AIRBOT velocity control is not enabled yet")

    def send_cartesian_pose(self, position, orientation, gripper: float) -> None:
        position = tuple(float(value) for value in position)
        orientation = tuple(float(value) for value in orientation)
        if self._last_command_rejected:
            return
        if self._pose_matches_hold(position, orientation):
            return
        self._last_command_rejected = False
        if not self.send_cartesian_target(position, orientation, gripper):
            self._last_command_rejected = True
            self._rejected_at = time.monotonic()
            return

    def send_cartesian_target(self, position, orientation, gripper: float) -> bool:
        """Submit a nonblocking Cartesian target, returning SDK acceptance.

        The RL worker handles deadlines/rejection itself, so this path does not
        substitute cached poses or suppress a new target after an earlier rejection.
        """
        if not self._send_pose(position, orientation, gripper):
            return False
        self._last_command_rejected = False
        self._last_gripper = self._clamped_gripper(gripper)
        self._hold_pose = (
            tuple(float(value) for value in position),
            tuple(float(value) for value in orientation),
        )
        self._hold_gripper = self._clamped_gripper(gripper)
        self._hold_command_active = False
        return True

    def recover_rejected_pose(self) -> None:
        """Retry one rejected Cartesian target, then clear the failure state."""
        if not self._last_command_rejected:
            return
        if (
            self._rejected_at is not None
            and time.monotonic() - self._rejected_at
            < AirbotArmClient.REJECTED_POSE_RETRY_S
        ):
            return
        self._last_command_rejected = False
        self._rejected_at = None
        if self._hold_pose is None:
            return
        if not self._send_pose(
            self._hold_pose[0], self._hold_pose[1], self._hold_gripper
        ):
            self._last_command_rejected = True
            self._rejected_at = time.monotonic()
            return
        self._last_gripper = self._hold_gripper
        self._hold_command_active = False

    @property
    def last_command_rejected(self) -> bool:
        return self._last_command_rejected

    def return_zero(self) -> None:
        self._require_control()
        if not self._client.return_zero():
            raise RuntimeError("AIRBOT return_zero was rejected")

    def move_joint(
        self, angles, blocking: bool = False, timeout_ms: int = 5000
    ) -> bool:
        self._require_control()
        gripper = self._last_gripper
        if gripper is None:
            gripper = self.current_gripper()
        options = self._options_type()
        options.eef_pos = gripper
        options.eef_eff = getattr(self, "_gripper_effort_limit", 6.0)
        options.blocking = blocking
        accepted = self._client.move_joint(
            [float(value) for value in angles], options, timeout_ms=timeout_ms
        )
        if accepted and blocking:
            self._hold_pose = None
            self._hold_gripper = gripper
            self._hold_command_active = False
        return accepted

    def close(self) -> None:
        self._client.close()

    def emergency_hold(self) -> None:
        if self._control_enabled and self._hold_pose is not None:
            self._send_pose(self._hold_pose[0], self._hold_pose[1], self._hold_gripper)

    def _send_pose(self, position, orientation, gripper: float) -> bool:
        self._require_control()
        pose = self._cartesian_pose_type(
            position=tuple(float(value) for value in position),
            orientation=tuple(float(value) for value in orientation),
        )
        options = self._options_type()
        options.eef_pos = self._clamped_gripper(gripper)
        options.eef_eff = getattr(self, "_gripper_effort_limit", 6.0)
        options.blocking = False
        accepted = self._client.move_end_pose(pose, options)
        if not accepted:
            print(
                "AIRBOT rejected move_end_pose; keeping the last accepted target",
                flush=True,
            )
        return accepted

    def _pose_matches_hold(self, position, orientation) -> bool:
        if self._hold_pose is None:
            return False
        return (
            _distance(position, self._hold_pose[0])
            <= AirbotArmClient.POSE_COMMAND_EPSILON_M
            and _rotation_distance(orientation, self._hold_pose[1])
            <= AirbotArmClient.POSE_COMMAND_EPSILON_RAD
        )

    def _require_control(self) -> None:
        if not self._control_enabled:
            raise RuntimeError("AIRBOT control was not explicitly enabled")

    @staticmethod
    def _clamped_gripper(value: float) -> float:
        minimum, maximum = AirbotArmClient.G2_GRIPPER_RANGE
        return max(minimum, min(maximum, float(value)))


def _distance(a, b) -> float:
    return sum((float(av) - float(bv)) ** 2 for av, bv in zip(a, b)) ** 0.5


def _rotation_distance(a, b) -> float:
    norm_a = math.sqrt(sum(float(v) ** 2 for v in a))
    norm_b = math.sqrt(sum(float(v) ** 2 for v in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return math.pi
    dot = abs(sum(float(av) * float(bv) for av, bv in zip(a, b))) / (norm_a * norm_b)
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


class MockArmClient(ArmClient):
    def __init__(self):
        self.hold_count = 0
        self.commands = []
        self.pose_commands = []
        self.joint_commands = []
        self.zero_count = 0
        self.control_enabled = False
        self._pose = ((0.30, 0.0, 0.30), (0.0, 0.0, 0.5, -0.5))

    def enable_control(self) -> None:
        self.control_enabled = True

    def hold(self) -> None:
        self.hold_count += 1

    def get_end_pose(self):
        return self._pose

    def get_joint_angles(self):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def send_cartesian_velocity(
        self, vx: float, vy: float, vz: float, gripper: float
    ) -> None:
        self.commands.append((vx, vy, vz, gripper))

    def send_cartesian_pose(self, position, orientation, gripper: float) -> None:
        self.pose_commands.append((tuple(position), tuple(orientation), gripper))
        self._pose = (tuple(position), tuple(orientation))

    def return_zero(self) -> None:
        self.zero_count += 1
        self._pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def move_joint(
        self, angles, blocking: bool = False, timeout_ms: int = 5000
    ) -> bool:
        self.joint_commands.append((tuple(angles), blocking, timeout_ms))
        return True

    def close(self) -> None:
        self.control_enabled = False


class ArmSdkWorker:
    def __init__(
        self, control=False, workspace_margin_m=DEFAULT_MARGIN_M, gripper_config=None
    ):
        self.control = control
        self.workspace = ArmWorkspace(workspace_margin_m)
        self.gripper_config = (
            gripper_settings(gripper_config) if gripper_config is not None else None
        )
        self.gripper_status = None
        self.lock = threading.Lock()
        self.latest = None
        self.target = None
        self.error = None
        self.last_error = None
        self.disconnects = 0
        self.sent_count = 0
        self.rejected_count = 0
        self.last_rejection = None
        self.control_state = "idle"
        self.prepare_requested = False
        self.zero_requested = False
        self.return_zero_state = None
        self.return_zero_error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            self._session()
            # Connection, SDK cleanup and retry waits never run in the body loop.
            self.stop.wait(1)

    def _failed(self, error):
        with self.lock:
            self.error = self.last_error = str(error)
            self.disconnects += 1
            self.latest = self.target = None
            self.control_state = "idle"
            self.prepare_requested = False
            if self.gripper_config is not None:
                self.gripper_status = {"state": "unavailable", "error": str(error)}
            self.zero_requested = False
            if self.return_zero_state in ("pending", "moving"):
                self.return_zero_state, self.return_zero_error = "failed", str(error)

    def _session(self):
        arm = None
        enabled = False
        latest = None
        gripper = None
        last_target = None
        rejected_target = None
        zero_stable = 0
        gripper_control = (
            GripperControl(self.gripper_config)
            if self.control and self.gripper_config
            else None
        )
        last_gripper_sent = None
        prepare_stage = None
        try:
            arm = AirbotArmClient()
            position_tolerance = getattr(arm, "POSE_COMMAND_EPSILON_M", 0.001)
            rotation_tolerance = getattr(arm, "POSE_COMMAND_EPSILON_RAD", 0.02)
            previous, previous_ns = None, 0
            while not self.stop.is_set():
                tick = time.monotonic_ns()
                angles = arm.get_joint_angles()
                pose = arm.get_end_pose_feedback()
                gripper_feedback = (
                    arm.get_gripper_feedback() if gripper_control else None
                )
                if angles is None or pose is None:
                    raise RuntimeError("AIRBOT joint/Cartesian feedback unavailable")
                q = np.asarray(angles, dtype=float)
                if q.shape != (6,) or not np.isfinite(q).all():
                    raise ValueError("invalid AIRBOT joint feedback")
                position = np.asarray(pose[0], dtype=float)
                rotation = quaternion(
                    pose[1]
                )  # SDK xyzw; normalization is order independent.
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError("invalid AIRBOT Cartesian feedback")
                # Current adapter exposes no sensor timestamp/velocity. Estimate velocity separately.
                dq = (
                    np.zeros(6)
                    if previous is None
                    else (q - previous) / max((tick - previous_ns) / 1e9, 1e-6)
                )
                previous, previous_ns = q.copy(), tick
                poll_ms = (time.monotonic_ns() - tick) / 1e6
                if poll_ms >= 100:
                    raise RuntimeError("AIRBOT feedback reads exceeded 100 ms")
                if self.stop.is_set():
                    break
                latest = ArmSnapshot(
                    q, dq, position, rotation, tick, poll_ms, gripper_feedback
                )
                with self.lock:
                    self.latest = latest
                    self.error = None
                    if prepare_stage == 2:
                        # Publish readiness only with feedback read after both
                        # controller activation and the initial measured hold.
                        self.control_state = "ready"
                        prepare_stage = None
                    if self.prepare_requested:
                        self.prepare_requested = False
                        prepare_stage = 0
                        self.target = None
                    target = self.target
                    self.target = None
                    zero_requested = self.zero_requested
                    self.zero_requested = False
                if prepare_stage is not None:
                    if prepare_stage == 0:
                        arm.enable_control()
                        enabled = True
                        prepare_stage = 1
                    else:
                        if gripper_control and gripper_feedback is None:
                            raise RuntimeError(
                                "G2 feedback unavailable during preparation"
                            )
                        gripper = (
                            gripper_feedback[0]
                            if gripper_feedback is not None
                            else arm.current_gripper()
                        )
                        if not arm.send_cartesian_target(position, rotation, gripper):
                            raise RuntimeError("AIRBOT initial measured hold rejected")
                        last_target = (position.copy(), rotation.copy())
                        self.sent_count += 1
                        prepare_stage = 2
                    # Discard pre-activation targets; poll again after each
                    # potentially slow SDK operation, outside the body loop.
                    continue
                if gripper_control and gripper_feedback is None:
                    gripper_control.pause(unavailable=True)
                    self.gripper_status = {
                        "state": "unavailable",
                        "error": "G2 feedback unavailable",
                    }
                    target = None
                if zero_requested:
                    if not enabled:
                        arm.enable_control()
                        enabled = True
                    # Use the vendor operation, not a Cartesian guess at zero.
                    arm.return_zero()
                    with self.lock:
                        self.return_zero_state = "moving"
                    target = None
                if self.return_zero_state == "moving":
                    # SDK zero is joint1..6 ~= 0 (the original bring-up procedure).
                    # Confirm several feedback samples, rather than SDK acceptance.
                    zero_stable = (
                        zero_stable + 1
                        if (
                            not zero_requested
                            and np.max(np.abs(q)) <= 0.03
                            and np.max(np.abs(dq)) <= 0.05
                        )
                        else 0
                    )
                    if zero_stable >= 5:
                        with self.lock:
                            self.return_zero_state = "done"
                    target = None
                elif self.return_zero_state == "done":
                    target = None
                if target is not None and self.control:
                    submitted, target_pos, target_quat, opening = target
                    if target_pos is None and not enabled:
                        continue
                    if time.monotonic_ns() - submitted > CONTROL_TIMEOUT_NS:
                        raise RuntimeError("AIRBOT Cartesian target expired")
                    # Reconnection only reads feedback. A new explicit command
                    # is required before reacquiring the control lease.
                    if not enabled:
                        arm.enable_control()
                        enabled = True
                        gripper = arm.current_gripper()
                    if time.monotonic_ns() - submitted > CONTROL_TIMEOUT_NS:
                        raise RuntimeError("AIRBOT Cartesian target expired")
                    holding = target_pos is None
                    if gripper_control:
                        gripper = gripper_control.step(
                            None if holding else opening,
                            gripper_feedback[0],
                            gripper_feedback[2],
                            tick,
                        )
                        gripper_error = None
                        if gripper_control.armed and (
                            gripper_control.force_send
                            or last_gripper_sent is None
                            or abs(gripper - last_gripper_sent) >= 0.0001
                        ):
                            if arm.send_gripper_target(gripper):
                                last_gripper_sent = gripper
                                gripper_control.force_send = False
                            else:
                                gripper_control.pause(gripper_feedback[0])
                                gripper = gripper_feedback[0]
                                gripper_error = "SDK gripper target rejected"
                        self.gripper_status = {
                            "state": gripper_control.state,
                            "target_m": gripper,
                            "position_m": gripper_feedback[0],
                            "effort": gripper_feedback[2],
                            "max_position_lead_m": GRIPPER_MAX_LEAD_M,
                            "speed_m_s": self.gripper_config["speed_m_s"],
                            "error": gripper_error,
                        }
                    if holding:
                        target_pos, target_quat = position, rotation
                    unchanged = (
                        not holding
                        and last_target is not None
                        and (
                            np.linalg.norm(target_pos - last_target[0])
                            <= position_tolerance
                            and 2
                            * math.acos(
                                min(
                                    1.0, abs(float(np.dot(target_quat, last_target[1])))
                                )
                            )
                            <= rotation_tolerance
                        )
                    )
                    # Same 1 mm / 0.02 rad tolerance as AirbotArmClient's
                    # original pose workflow; retain the last accepted goal.
                    if not unchanged:
                        repeated_rejection = rejected_target is not None and (
                            np.linalg.norm(target_pos - rejected_target[0])
                            <= position_tolerance
                            and abs(float(np.dot(target_quat, rejected_target[1])))
                            >= math.cos(rotation_tolerance / 2)
                        )
                        if not repeated_rejection:
                            if arm.send_cartesian_target(
                                target_pos, target_quat, gripper
                            ):
                                last_target = (
                                    np.array(target_pos),
                                    np.array(target_quat),
                                )
                                rejected_target = None
                                with self.lock:
                                    self.sent_count += 1
                            else:
                                # A rejected target is not a transport failure.
                                # Keep polling and keep the last accepted servo goal.
                                rejected_target = (
                                    np.array(target_pos),
                                    np.array(target_quat),
                                )
                                with self.lock:
                                    self.rejected_count += 1
                                    self.last_rejection = "AIRBOT Cartesian target rejected; retaining last accepted target"
                self.stop.wait(max(0, 0.02 - (time.monotonic_ns() - tick) / 1e9))
        except Exception as error:  # noqa: BLE001 - Isolate SDK failures and retry off the body loop.
            self._failed(error)
        finally:
            if arm is not None:
                try:
                    # Best effort measured-pose hold; never use stale feedback on exit.
                    if (
                        enabled
                        and self.stop.is_set()
                        and self.error is None
                        and self.return_zero_state is None
                        and latest is not None
                        and time.monotonic_ns() - latest.sampled_ns < 100000000
                        and not arm.send_cartesian_target(
                            latest.position, latest.quaternion_xyzw, gripper
                        )
                    ):
                        self._failed("AIRBOT exit hold rejected")
                except Exception as error:  # noqa: BLE001 - Report exit failures without skipping SDK cleanup.
                    self._failed(f"AIRBOT exit hold failed: {error}")
                finally:
                    try:
                        arm.close()
                    except Exception as error:  # noqa: BLE001 - Cleanup must not kill the reconnect worker.
                        self._failed(f"AIRBOT SDK close failed: {error}")

    def snapshot(self):
        with self.lock:
            return self.latest if self.error is None else None

    def prepare_control(self):
        """Explicit support-state request; initialization runs only on this worker."""
        if not self.control:
            return False
        with self.lock:
            if self.control_state in ("preparing", "ready"):
                return True
            # IPC may arrive just before the next sample. Preparation itself
            # always reads and validates fresh feedback before any SDK action.
            if self.return_zero_state is not None:
                return False
            self.target = None
            self.control_state = "preparing"
            self.prepare_requested = True
        return True

    def command(
        self, position, quaternion_xyzw, *, submitted_ns=None, gripper_open=None
    ):
        if not self.control:
            raise RuntimeError("AIRBOT control is not enabled")
        p = np.asarray(position, dtype=float)
        q = quaternion(quaternion_xyzw)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("invalid AIRBOT Cartesian target")
        p, _ = self.workspace.limit(p, q)
        if gripper_open is not None and not isinstance(gripper_open, bool):
            raise ValueError("gripper open request must be boolean")
        with self.lock:
            if (
                self.error is not None
                or self.latest is None
                or time.monotonic_ns() - self.latest.sampled_ns > CONTROL_TIMEOUT_NS
                or self.return_zero_state is not None
                or self.control_state == "preparing"
            ):
                return False
            self.target = (
                time.monotonic_ns() if submitted_ns is None else submitted_ns,
                p.copy(),
                q,
                gripper_open,
            )
        return True

    def request_return_zero(self):
        if not self.control:
            return False
        with self.lock:
            if (
                self.error is not None
                or self.latest is None
                or (time.monotonic_ns() - self.latest.sampled_ns > CONTROL_TIMEOUT_NS)
            ):
                return False
            self.target = None
            self.zero_requested = True
            self.return_zero_state, self.return_zero_error = "pending", None
        return True

    def hold_current(self):
        if self.control:
            with self.lock:
                # Do not acquire a new lease just to hold after a disconnect.
                self.target = (
                    (time.monotonic_ns(), None, None, None)
                    if self.error is None
                    and self.latest is not None
                    and self.return_zero_state is None
                    else None
                )

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            self.error = self.error or "AIRBOT SDK worker did not exit within 1 s"


def sdk_child():
    # Only the child starts the actual SDK worker/imports arm_sdk.

    fd = int(os.environ[STATUS_FD_ENV])
    margin = (
        float(sys.argv[sys.argv.index("--workspace-margin-m") + 1])
        if "--workspace-margin-m" in sys.argv
        else DEFAULT_MARGIN_M
    )
    worker = ArmSdkWorker(
        control="--control" in sys.argv[1:],
        workspace_margin_m=margin,
        gripper_config=json.loads(sys.argv[sys.argv.index("--gripper-config") + 1])
        if "--gripper-config" in sys.argv
        else None,
    )
    buffer = b""
    try:
        with os.fdopen(fd, "w", buffering=1) as status:
            while True:
                if select.select([0], [], [], 0.02)[0]:
                    data = os.read(0, 65536)
                    if not data:
                        break
                    lines = (buffer + data).split(b"\n")
                    buffer = lines.pop()
                    if len(buffer) > 4096:
                        raise ValueError("invalid AIRBOT command record")
                    messages = [json.loads(line) for line in lines]
                    if any(m["kind"] == "stop" for m in messages):
                        break
                    if messages:
                        command = messages[-1]
                        if command["kind"] == "hold":
                            worker.hold_current()
                        elif command["kind"] == "prepare":
                            worker.prepare_control()
                        elif command["kind"] == "return_zero":
                            if not worker.request_return_zero():
                                with worker.lock:
                                    worker.return_zero_state = "failed"
                                    worker.return_zero_error = (
                                        "AIRBOT feedback unavailable for return-zero"
                                    )
                        elif command["kind"] == "target":
                            worker.command(
                                command["position"],
                                command["quaternion_xyzw"],
                                submitted_ns=command["submitted_ns"],
                                gripper_open=command.get("gripper_open"),
                            )
                with worker.lock:
                    sample = worker.latest if worker.error is None else None
                    payload = {
                        "sample": {
                            "q": sample.q.tolist(),
                            "dq": sample.dq.tolist(),
                            "position": sample.position.tolist(),
                            "quaternion_xyzw": sample.quaternion_xyzw.tolist(),
                            "sampled_ns": sample.sampled_ns,
                            "poll_ms": sample.poll_ms,
                            "gripper": sample.gripper,
                        }
                        if sample
                        else None,
                        "error": worker.error,
                        "last_error": worker.last_error,
                        "disconnects": worker.disconnects,
                        "sent_count": worker.sent_count,
                        "rejected_count": worker.rejected_count,
                        "last_rejection": worker.last_rejection,
                        "return_zero_state": worker.return_zero_state,
                        "return_zero_error": worker.return_zero_error,
                        "gripper_status": worker.gripper_status,
                        "control_state": worker.control_state,
                    }
                status.write(json.dumps(payload, allow_nan=False) + "\n")
    except BrokenPipeError:
        pass
    finally:
        worker.close()


if __name__ == "__main__":
    if "--sdk-child" not in sys.argv[1:]:
        raise SystemExit("Internal AIRBOT process; use python -m l1w_teleop.rl")
    sdk_child()
