"""Body posture parameters and state transitions (adapted under Apache-2.0)."""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIG, body_settings

@dataclass(frozen=True)
class StandParameters:
    position: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    raw_feedforward: np.ndarray
    seconds: float
    hip_seconds: float
    timeout_seconds: float


class BodyProfile:
    def __init__(self, path=None):
        data = Path(path or DEFAULT_CONFIG).read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.config = c = body_settings(json.loads(data))
        if (
            c.get("version") != 1
            or c.get("coordinate_system") != "raw_motor"
            or c.get("leg_order") != ["FR", "FL", "RR", "RL"]
            or c.get("joint_order") != ["abad", "hip", "knee", "foot"]
        ):
            raise ValueError("body profile requires FR/FL/RR/RL raw motor coordinates")
        for key in (
            "stand_raw_pose",
            "lie_raw_pose",
            "stand_raw_kp",
            "stand_raw_kd",
            "stand_raw_feedforward",
            "passive_raw_kd",
        ):
            values = np.asarray(c[key], dtype=float)
            if values.shape != (4, 4) or not np.isfinite(values).all():
                raise ValueError(f"body {key} requires a finite 4 x 4 array")
            setattr(self, key, values.reshape(16))
        for key in ("stand_raw_kp", "stand_raw_kd", "passive_raw_kd"):
            if np.any(getattr(self, key) < 0):
                raise ValueError(f"body {key} must be nonnegative")
        if np.any(self.stand_raw_kp[3::4] != 0):
            raise ValueError("body stand wheel Kp must be zero")
        self.seconds = float(c["stand_seconds"])
        self.hip_seconds = float(c.get("stand_hip_return_seconds", self.seconds / 3))
        self.timeout = float(c["stand_timeout_seconds"])
        if (
            c["stand_curve"] != "two_stage_cubic"
            or not np.isfinite([self.hip_seconds, self.seconds, self.timeout]).all()
            or not 0 < self.hip_seconds < self.seconds < self.timeout
        ):
            raise ValueError(
                "body requires two_stage_cubic and 0 < hip_return_seconds < stand_seconds < timeout"
            )

    def apply(self, mapping):
        """Compose local stand/descent data without changing verification metadata."""
        c = mapping.config
        c.update(
            shutdown_raw_pose=self.lie_raw_pose.tolist(),
            hold_kp=self.stand_raw_kp.tolist(),
            hold_kd=self.stand_raw_kd.tolist(),
            passive_kd=self.passive_raw_kd.tolist(),
        )

    def validate(self, mapping):
        """Check executable endpoint/gain data without deployment amplitude caps."""
        for pose, feedforward in (
            (self.stand_raw_pose, self.stand_raw_feedforward),
            (self.lie_raw_pose, np.zeros(16)),
        ):
            targets = np.column_stack(
                (pose, np.zeros(16), self.stand_raw_kp, self.stand_raw_kd, feedforward)
            )
            mapping.validate_targets(targets)

    def stand_parameters(self, mapping, default):
        # The FSM uses policy ordering/coordinates; the final map must reproduce
        # the exact raw goals and gains, even for sign/ratio/zero-point changes.
        q = np.asarray(default, dtype=float).copy()
        q[:16] = (
            self.stand_raw_pose[mapping.indices] - mapping.offset
        ) / mapping.factor
        kp, kd = np.zeros(22), np.zeros(22)
        kp[:16] = self.stand_raw_kp[mapping.indices] * mapping.ratio**2
        kd[:16] = self.stand_raw_kd[mapping.indices] * mapping.ratio**2
        return StandParameters(
            q,
            kp,
            kd,
            self.stand_raw_feedforward.copy(),
            self.seconds,
            self.hip_seconds,
            self.timeout,
        )

class State(str, Enum):
    PASSIVE = "passive"
    STAND = "stand"
    LOCOMOTION = "locomotion"
    HOLD = "hold"
    STOPPING = "stopping"
    FAULT = "fault"


def blend(ratio):
    r = np.clip(ratio, 0, 1)
    return r * r * (3 - 2 * r)


class FSMController:
    """Two-stage get-up and optionally direct RL-to-stand return."""

    def __init__(self, cfg, core, initial=None, passive_kd=None, stand=None):
        self.cfg = cfg["states"]
        self.core = core
        self.home = core.default.copy()
        self.passive_kd = (
            core.kd.copy() if passive_kd is None else np.asarray(passive_kd).copy()
        )
        self.duration = float(self.cfg["transition_seconds"])
        self.hip_duration = float(self.cfg["hip_return_seconds"])
        self.rl_return_single_stage = self.cfg.get("rl_return_single_stage", True)
        if not isinstance(self.rl_return_single_stage, bool):
            raise ValueError("rl_return_single_stage must be boolean")
        self.stand_two_stage = True
        self.gain_duration = float(self.cfg.get("gain_transition_seconds", 0.3))
        self.stable_duration = float(self.cfg.get("settle_seconds", 0.3))
        self.require_settling = self.cfg.get("require_stand_settling", True)
        if not isinstance(self.require_settling, bool):
            raise ValueError("require_stand_settling must be boolean")  # noqa: TRY004 - Invalid config value.
        self.position_tolerance = float(
            self.cfg.get("settle_position_tolerance_rad", 0.2)
        )
        self.velocity_tolerance = float(
            self.cfg.get("settle_velocity_tolerance_rad_s", 0.5)
        )
        tolerances = np.asarray([self.position_tolerance, self.velocity_tolerance])
        if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
            raise ValueError("stand settling tolerances must be finite and positive")
        self.timeout = float(self.cfg.get("stand_timeout_seconds", 8))
        if not (0 < self.hip_duration < self.duration < self.timeout):
            raise ValueError(
                "stand requires 0 < hip_return_seconds < transition_seconds < stand_timeout_seconds"
            )
        if not (0 < self.gain_duration <= self.duration and self.stable_duration > 0):
            raise ValueError("invalid stand settling/gain transition duration")
        self.stand_kp, self.stand_kd = core.kp.copy(), core.kd.copy()
        self.stand_kp[:12] = self.cfg["leg_kp"]
        self.stand_kd[:12] = self.cfg["leg_kd"]
        self.stand_kd[12:16] = self.cfg["wheel_kd"]
        self.stand_kp[16:] = 0
        self.stand_feedforward = np.zeros(16)
        self.feedforward = np.zeros(
            16
        )  # Raw motor order, independent of policy torque units.
        self.origin_feedforward = np.zeros(16)
        if stand is not None:
            self.home = stand.position.copy()
            self.stand_kp, self.stand_kd = stand.kp.copy(), stand.kd.copy()
            self.stand_feedforward = stand.raw_feedforward.copy()
            self.duration, self.timeout = stand.seconds, stand.timeout_seconds
            self.hip_duration = stand.hip_seconds
            if not (
                0 < self.hip_duration < self.duration < self.timeout
                and 0 < self.gain_duration <= self.duration
            ):
                raise ValueError("invalid hardware stand timing")
        self.state = State.PASSIVE
        self.requested = State(initial or self.cfg["initial"])
        if self.requested not in (State.PASSIVE, State.STAND, State.LOCOMOTION):
            raise ValueError("invalid initial state")
        self.phase = "waiting_bridge"
        self.entered = None
        self.stable_since = None
        self.deadline = None
        self.last = None
        self.origin = None
        self.hip_pose = None
        self.reason = None
        self.hold_reason = None
        self.generation = 0

    @property
    def running_policy(self):
        return self.state == State.LOCOMOTION

    @property
    def arm_allowed(self):
        return self.state in (State.STAND, State.LOCOMOTION)

    def _reset_policy(self):
        self.core.reset()
        self.core.velocity.fill(0)
        self.generation += 1

    def _enter_stand(self, q, now):
        # Capture the source state before switching to STAND. Only a direct
        # locomotion return skips abad alignment; passive/recovery still get up.
        self.stand_two_stage = not (
            self.state == State.LOCOMOTION and self.rl_return_single_stage
        )
        # Last submitted goal preserves target continuity. Feedback settling,
        # when enabled, uses measurements rather than these target positions.
        if self.last is None or self.state == State.PASSIVE:
            self.origin = (q.copy(), np.zeros(22), np.zeros(22), self.passive_kd.copy())
        else:
            self.origin = tuple(a.copy() for a in self.last)
        self.origin_feedforward = self.feedforward.copy()
        # Releasing the locomotion deadman stops wheels immediately, including
        # during the leg return trajectory; never fade the old wheel speed.
        self.origin[1][12:16] = 0
        self.origin_feedforward[3::4] = 0
        self.hip_pose = self.origin[0].copy()
        self.hip_pose[:12:3] = self.home[:12:3]
        self.state = State.STAND
        self.phase = "hip_return" if self.stand_two_stage else "direct_return"
        self.entered, self.stable_since = now, None
        self.deadline = now + self.timeout if self.require_settling else None
        self._reset_policy()

    def request(self, state, q, now):
        requested = State(state)
        if requested not in (State.PASSIVE, State.STAND, State.LOCOMOTION):
            raise ValueError("use stop/fail for terminal states")
        if self.state in (State.STOPPING, State.FAULT) or requested == self.requested:
            return
        if self.state == State.HOLD and requested != State.PASSIVE:
            self.requested = requested
            self.hold_reason = None
            self._enter_stand(q, now)
            return
        self.requested = requested
        if requested == State.PASSIVE:
            self.hold_reason = None
            self.state, self.phase = State.PASSIVE, "descending"
            self.last = None
            self.feedforward.fill(0)
            self._reset_policy()
        elif self.state == State.LOCOMOTION and requested == State.STAND:
            self._enter_stand(q, now)
        # Requests from passive are queued until the bridge reports local rest.

    def hold(self, q, reason):
        """Pause without a descent or automatic resumption of policy actions."""
        if self.state in (State.PASSIVE, State.STOPPING, State.FAULT):
            return
        if self.state != State.HOLD:
            kp, kd = (
                self.last[2:4]
                if self.last is not None
                else (self.stand_kp, self.stand_kd)
            )
            self.last = (q.copy(), np.zeros(22), kp.copy(), kd.copy())
            self.feedforward.fill(0)
            self.state = self.requested = State.HOLD
            self._reset_policy()
        self.phase, self.hold_reason = "holding", str(reason)

    def advance(
        self, q, dq, now, *, passive_ready=False, descending=False, policy_ready=True
    ):
        if self.state in (State.STOPPING, State.FAULT):
            return
        if self.state == State.PASSIVE:
            self.phase = (
                "descending"
                if descending
                else ("passive" if passive_ready else "waiting_bridge")
            )
            self.core.velocity.fill(0)
            if passive_ready and self.requested != State.PASSIVE:
                self._enter_stand(q, now)
            return
        if self.state != State.STAND:
            return
        elapsed = now - self.entered
        self.phase = (
            ("hip_return" if elapsed < self.hip_duration else "rise")
            if self.stand_two_stage else "direct_return"
        )
        if elapsed < self.duration:
            return
        ready = not self.require_settling
        if self.require_settling:
            position_error = np.max(np.abs(q[:12] - self.home[:12]))
            speed = np.max(np.abs(dq[:12]))
            settled = (
                position_error < self.position_tolerance
                and speed < self.velocity_tolerance
            )
            self.stable_since = (
                (now if self.stable_since is None else self.stable_since)
                if settled
                else None
            )
            ready = (
                self.stable_since is not None
                and now - self.stable_since >= self.stable_duration
            )
            if ready:
                self.deadline = None
            elif self.deadline is None:
                self.deadline = now + self.timeout
            elif now > self.deadline:
                raise RuntimeError(
                    "stand tracking did not settle before timeout: "
                    f"max leg error={position_error:.4f} rad (limit {self.position_tolerance:g}), "
                    f"max leg speed={speed:.4f} rad/s (limit {self.velocity_tolerance:g}); "
                    f"requires {self.stable_duration:g} s continuously within limits"
                )
        else:
            self.deadline = None
        self.phase = "stand" if ready else "settling"
        if ready and self.requested == State.LOCOMOTION:
            if not policy_ready:
                self.phase = "waiting_arm"
                return
            self.origin = tuple(a.copy() for a in self.last)
            self.origin_feedforward = self.feedforward.copy()
            self.state, self.phase, self.entered = State.LOCOMOTION, "policy", now
            self._reset_policy()

    def command(self, state, now):
        q = state["joint_pos"]
        if self.state == State.HOLD:
            return tuple(a.copy() for a in self.last)
        if self.state == State.PASSIVE:
            self.feedforward.fill(0)
            return q.copy(), np.zeros(22), np.zeros(22), self.passive_kd.copy()
        if self.state == State.STAND:
            elapsed = now - self.entered
            if not self.stand_two_stage:
                start, goal = self.origin[0], self.home
                ratio = elapsed / self.duration
            elif elapsed <= self.hip_duration:
                start, goal = self.origin[0], self.hip_pose
                ratio = elapsed / self.hip_duration
            else:
                start, goal = self.hip_pose, self.home
                ratio = (elapsed - self.hip_duration) / (
                    self.duration - self.hip_duration
                )
            position = start + blend(ratio) * (goal - start)
            gain_blend = blend(elapsed / self.gain_duration)
            velocity = (1 - gain_blend) * self.origin[1]
            kp = self.origin[2] + gain_blend * (self.stand_kp - self.origin[2])
            kd = self.origin[3] + gain_blend * (self.stand_kd - self.origin[3])
            self.feedforward = self.origin_feedforward + gain_blend * (
                self.stand_feedforward - self.origin_feedforward
            )
            position[16:] = q[16:]
            result = (position, velocity, kp, kd)
        else:
            raise RuntimeError(
                "policy targets must be submitted through policy_command"
            )
        self.last = tuple(a.copy() for a in result)
        return result

    def policy_command(self, targets, now):
        if self.state != State.LOCOMOTION:
            raise RuntimeError("policy is not enabled")
        # Match the runner: forward policy targets/gains without a deployment blend.
        result = tuple(a.copy() for a in targets)
        self.feedforward.fill(0)
        self.phase = "policy"
        self.last = tuple(a.copy() for a in result)
        return result

    def stop(self):
        self.state, self.phase = State.STOPPING, "stopping"
        self._reset_policy()

    def rearm(self):
        """Reuse the loaded policy after recovery; always go through stand."""
        self.state, self.phase = State.PASSIVE, "waiting_bridge"
        self.requested = State.STAND
        self.reason = None
        self.hold_reason = None
        self.entered = self.stable_since = self.deadline = None
        self.last = self.origin = self.hip_pose = None
        self.stand_two_stage = True
        self.feedforward.fill(0)
        self.origin_feedforward.fill(0)
        self._reset_policy()

    def fail(self, reason):
        self.state, self.phase, self.reason = State.FAULT, "fault", str(reason)
        self._reset_policy()
