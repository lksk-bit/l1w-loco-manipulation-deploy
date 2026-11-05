"""Ordered control cycle, session state and resident fault recovery.

Runtime supplies configuration and resources before run(). Each stage retains
its original place inside the per-cycle exception boundary.
"""

import secrets
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import bridge as wire
from .arm import ArmFeedback
from .body import State
from .diagnostics import (
    CycleBudget,
    CycleStages,
    FeedbackStale,
    cycle_record,
    require_fresh_command,
)


@dataclass
class ControlCycle:
    """Values shared by consecutive stages of one control iteration."""

    tick: int
    keys: str
    requested: State | None = None
    previous_arm_allowed: bool | None = None
    frame: wire.StateFrame | None = None
    received: int | None = None
    state_age: float | None = None
    pico_arm_enabled: bool | None = None
    gripper_open_request: bool | None = None
    now: float | None = None
    arm_preparing: bool | None = None
    arm_ready: bool | None = None
    workspace_clamped: bool | None = None
    workspace_requested: np.ndarray | None = None
    obs: np.ndarray | None = None
    action: np.ndarray | None = None
    q: np.ndarray | None = None
    dq: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    targets: np.ndarray | None = None
    arm_target: tuple | None = None
    arm_submit_accepted: bool | None = None
    policy_drop_reason: str | None = None
    packet: bytes | None = None


class Controller:
    """Control session state; runtime opens and closes its external resources."""

    def __init__(self, args):
        self.args = args
        self.link = self.arm = self.terminal = self.recorder = self.pico = self.fsm = (
            self.last_frame
        ) = None
        self.last_command_boot = None
        self.arm_feedback = ArmFeedback()
        self.snapshot = None
        self.bridge_failure_snapshot = None
        self.recovery_events = deque(maxlen=100)
        self.old_handlers = {}
        self.stop = False
        self.stop_requests = 0
        self.return_zero_on_stop = False
        self.arm_exit_result = None
        self.workspace_clamps = 0
        self.session, self.sequence = secrets.randbits(64) or 1, 0
        self.writes = 0
        self.timings, self.ages, self.wake_late = (
            deque(maxlen=10000),
            deque(maxlen=10000),
            deque(maxlen=10000),
        )
        self.ack_times = deque(maxlen=10000)
        self.arm_ages, self.arm_polls = deque(maxlen=10000), deque(maxlen=10000)
        self.pending_sends = {}
        self.last_ack = 0
        self.budget, self.stages = CycleBudget(), CycleStages()
        self.command_times = deque(maxlen=10000)

    def _start_loop(self):
        self.generation = -1
        self.state_name = "locomotion"
        self.state_phase = "policy"
        self.next_tick = time.monotonic_ns()
        self.count = 0
        self.period_ns = 20000000
        self.pico_armed = False
        self.pico_previous_x = False
        self.pico_motion = False
        self.pico_rl_enabled = False
        self.ee_initialized = False
        self.arm_was_active = False
        self.passive_generation = -1
        self.passive_sequence = 0
        self.state = self.base_state

    def run(self):
        self._start_loop()
        while not self.stop and (self.args.steps == 0 or self.count < self.args.steps):
            if self.args.mode != "benchmark":
                remaining = (self.next_tick - time.monotonic_ns()) / 1e9
                if remaining > 0:
                    time.sleep(remaining)
            tick = time.monotonic_ns()
            self.stages.start(tick)
            self.wake_late.append(max(0, tick - self.next_tick) / 1e6)
            self.next_tick += self.period_ns
            if tick - self.next_tick >= self.period_ns:
                self.next_tick = (
                    tick + self.period_ns
                )  # Skip missed periods; never burst stale actions.
            keys = self.terminal.read()
            if "\x03" in keys or "\x1b" in keys:
                self.return_zero_on_stop = True
                break
            cycle = ControlCycle(tick, keys)
            try:
                self._read_input(cycle)
                self._read_feedback(cycle)
                self._read_pico(cycle)
                self._update_posture(cycle)
                self._infer(cycle)
                self._map_command(cycle)
                self._send(cycle)
                self._record(cycle)
            except FeedbackStale as error:
                if not self.controlled:
                    raise
                self.fsm.hold(self.state["joint_pos"], error)
                self.core.velocity.fill(0)
                if self.arm_was_active:
                    self.arm.hold_current()
                self.arm_was_active = self.ee_initialized = False
                self.recorder.append(
                    {
                        "cycle": self.count,
                        "state": self.fsm.state.value,
                        "phase": "waiting_feedback",
                        "requested": self.fsm.requested.value,
                        "hold_reason": str(error),
                        "policy_action_applied": False,
                    }
                )
                elapsed_ms = (time.monotonic_ns() - cycle.tick) / 1e6
                self.timings.append(elapsed_ms)
                self.budget.record(elapsed_ms)
                self.count += 1
            except (ValueError, RuntimeError, OSError) as error:
                if not self.controlled or self.args.steps != 0:
                    raise
                self.fsm.fail(error)
                self.recovery_events.append(
                    {"reason": str(error), "at_ns": time.monotonic_ns()}
                )
                if self.arm_was_active:
                    self.arm.hold_current()
                self.arm_was_active = self.ee_initialized = False
                failed_boot = (
                    self.last_frame.boot if self.last_frame is not None else None
                )
                if (
                    self.last_frame is not None
                    and self.last_frame.boot == self.last_command_boot
                ):
                    try:
                        self.sequence += 1
                        self.link.send(
                            wire.pack_command(
                                self.session,
                                self.sequence,
                                time.monotonic_ns(),
                                self.last_frame,
                                np.zeros((16, 5)),
                                mode=wire.SHUTDOWN,
                            )
                        )
                    except (ValueError, OSError):
                        pass  # The bridge's command watchdog also descends locally.

                def recovery_status(phase, reason=str(error)):
                    self.recorder.append(
                        {
                            "state": "fault",
                            "phase": phase,
                            "requested": "passive",
                            "recovery_reason": reason,
                        }
                    )

                if not wait_for_rearm(
                    self.link,
                    self.terminal,
                    self.pico,
                    lambda: self.stop,
                    self.state,
                    self.mapping,
                    failed_boot,
                    recovery_status,
                ):
                    break
                # Do not reload TorchScript or the SDK clients. Clear all old
                # command/session history and require a new passive handshake.
                self.fsm.rearm()
                self.session, self.sequence = secrets.randbits(64) or 1, 0
                self.pending_sends.clear()
                self.last_ack = self.passive_sequence = 0
                self.last_command_boot = None
                self.last_frame = None
                self.generation = self.passive_generation = -1
                self.pico_armed = self.pico_motion = self.pico_rl_enabled = False
                self.next_tick = time.monotonic_ns()
        if self.fsm:
            self.fsm.stop()
        return 0

    def _read_input(self, cycle):
        cycle.requested = None
        cycle.previous_arm_allowed = bool(self.fsm and self.fsm.arm_allowed)
        for key in cycle.keys:
            if key in "123":
                cycle.requested = {
                    "1": State.PASSIVE,
                    "2": State.STAND,
                    "3": State.LOCOMOTION,
                }[key]
        for key, index, amount in [
            ("w", 0, 0.2),
            ("s", 0, -0.2),
            ("a", 1, 0.2),
            ("d", 1, -0.2),
            ("q", 2, 0.2),
            ("e", 2, -0.2),
        ]:
            if key in cycle.keys:
                self.core.velocity[index] += amount
        if " " in cycle.keys:
            self.core.velocity.fill(0)
        self.stages.mark("input")

    def _read_feedback(self, cycle):
        cycle.frame = None
        if self.link is None:
            self.state = self.base_state
            cycle.state_age = 0
        else:
            cycle.frame, cycle.received, boot_changed = self.link.snapshot()
            cycle.state_age = (time.monotonic_ns() - cycle.received) / 1e6
            if cycle.frame is None or boot_changed or cycle.frame.flags & wire.FAULT:
                self.bridge_failure_snapshot = {
                    "received_age_ms": cycle.state_age,
                    "wake_lateness_ms": self.wake_late[-1],
                    "boot_changed": boot_changed,
                    "flags": cycle.frame.flags if cycle.frame else None,
                    "state_sequence": cycle.frame.sequence if cycle.frame else None,
                    "accepted_command": getattr(cycle.frame, "accepted_command", None),
                    "command_age_ms": getattr(cycle.frame, "command_age_ms", None),
                    "motor_feedback_age_ms": getattr(
                        cycle.frame, "feedback_age_ms", None
                    ),
                    "motor_lateness_us": getattr(cycle.frame, "lateness_us", None),
                    "motor_flags": getattr(cycle.frame, "motor_flags", None),
                }
                reasons = []
                if cycle.frame is None:
                    reasons.append("no state")
                if boot_changed:
                    reasons.append("bridge restarted")
                if cycle.state_age > wire.CONTROL_TIMEOUT_MS:
                    reasons.append(f"feedback age={cycle.state_age:.1f} ms")
                if cycle.frame is not None and cycle.frame.flags & wire.FAULT:
                    reasons.append(f"bridge FAULT flags=0x{cycle.frame.flags:04x}")
                raise RuntimeError(
                    "bridge state rejected: "
                    + "; ".join(reasons)
                    + f"; wake lateness={self.wake_late[-1]:.1f} ms; stopping inference"
                )
            if cycle.state_age > wire.CONTROL_TIMEOUT_MS:
                raise FeedbackStale(
                    f"bridge feedback age={cycle.state_age:.1f} ms; holding until fresh feedback"
                )
            if (
                cycle.frame.flags & (wire.MOTORS_VALID | wire.IMU_VALID)
                != wire.MOTORS_VALID | wire.IMU_VALID
            ):
                raise RuntimeError("bridge motor/IMU feedback invalid")
            if self.args.mode == "mock" and not cycle.frame.flags & wire.MOCK:
                raise RuntimeError("mock mode refuses a physical bridge")
            if self.args.mode == "mock" and not cycle.frame.flags & wire.CONTROL:
                raise RuntimeError("mock command test requires --mock --control bridge")
            if self.args.mode == "hardware" and (
                cycle.frame.flags & wire.MOCK
                or not cycle.frame.flags & wire.CONTROL
                or not cycle.frame.flags & wire.RT_READY
            ):
                raise RuntimeError(
                    "hardware requires a physical RT bridge with control enabled"
                )
            if (
                self.args.mode == "hardware"
                and cycle.frame.profile_crc != self.profile_crc
            ):
                raise RuntimeError("NX/3588 hardware profiles differ")
            if self.args.mode == "observe" and cycle.frame.flags & wire.CONTROL:
                raise RuntimeError("observe requires bridge read-only mode")
            q16, dq16, gyro, gravity = self.mapping.read(cycle.frame)
            if self.arm is None:
                aq = np.asarray(self.cfg["robot"]["default_dof_pos"][16:])
                adq = np.zeros(6)
                extra = {
                    k: self.base_state[k] for k in ("ee_position", "ee_quaternion")
                }
            else:
                self.arm_feedback.update(self.arm)
                self.snapshot = self.arm_feedback.snapshot
                if self.snapshot is not None:
                    self.arm_ages.append(
                        (time.monotonic_ns() - self.snapshot.sampled_ns) / 1e6
                    )
                    self.arm_polls.append(self.snapshot.poll_ms)
                    aq = self.snapshot.q
                    adq = (
                        self.snapshot.dq if self.arm_feedback.available else np.zeros(6)
                    )
                    aq, adq = self.mapping.read_arm(aq, adq)
                    extra = self.mapping.arm_pose.read(
                        self.snapshot.position, self.snapshot.quaternion_xyzw
                    )
                else:
                    # Body-only support bookkeeping, never policy input.
                    aq = self.base_state["joint_pos"][16:]
                    adq = np.zeros(6)
                    extra = {
                        k: self.base_state[k] for k in ("ee_position", "ee_quaternion")
                    }
                if self.args.arm_control and self.arm_feedback.available:
                    arm_lower = np.asarray(self.mapping.config["arm_raw_lower"])
                    arm_upper = np.asarray(self.mapping.config["arm_raw_upper"])
                    bad_joints = np.flatnonzero(
                        (self.snapshot.q < arm_lower) | (self.snapshot.q > arm_upper)
                    )
                    if bad_joints.size:
                        details = "; ".join(
                            f"J{i + 1}={self.snapshot.q[i]:+.5f} rad, "
                            f"range=[{arm_lower[i]:+.5f}, {arm_upper[i]:+.5f}], "
                            f"excess={max(arm_lower[i] - self.snapshot.q[i], self.snapshot.q[i] - arm_upper[i]):.5f} rad"
                            for i in bad_joints
                        )
                        raise RuntimeError(
                            "AIRBOT measured joints exceed SDK bounds: " + details
                        )
                if self.arm_feedback.available and not self.ee_initialized:
                    self.core.ee_target[:] = extra["ee_position"]
                    self.core.ee_quat[:] = extra["ee_quaternion"]
                    self.ee_initialized = True
                if not self.arm_feedback.available or self.arm_feedback.lost:
                    self.ee_initialized = False
                    if self.arm_was_active:
                        self.arm.hold_current()
                    self.arm_was_active = False
            cycle.q = np.concatenate([q16, aq])
            cycle.dq = np.concatenate([dq16, adq])
            self.state = dict(
                joint_pos=cycle.q,
                joint_vel=cycle.dq,
                local_ang_vel=gyro,
                projected_gravity=gravity,
                **extra,
            )
            if self.controlled and cycle.frame.flags & wire.HOLDING:
                self.fsm.hold(self.state["joint_pos"], "3588 command stream paused")
            self.last_frame = cycle.frame
            if (
                self.args.mode in ("mock", "hardware")
                and cycle.frame.accepted_command > self.sequence
            ):
                raise RuntimeError(
                    "bridge still owns an earlier session; waiting for local session shutdown"
                )
            if cycle.frame.accepted_command > self.last_ack:
                sent = self.pending_sends.get(cycle.frame.accepted_command)
                ack_received = self.link.ack_time(cycle.frame.accepted_command)
                if sent is not None and ack_received is not None:
                    self.ack_times.append(max(0, ack_received - sent) / 1e6)
                self.last_ack = cycle.frame.accepted_command
                self.pending_sends = {
                    seq: stamp
                    for seq, stamp in self.pending_sends.items()
                    if seq > self.last_ack
                }
        self.ages.append(cycle.state_age)
        self.stages.mark("feedback")

    def _read_pico(self, cycle):
        cycle.pico_arm_enabled = None
        cycle.gripper_open_request = None
        if self.args.arm_control and self.fsm is not None and not self.fsm.arm_allowed:
            # Stand and RL share one arm goal. Passive/hold discard it.
            self.core.ee_target[:] = self.state["ee_position"]
            self.core.ee_quat[:] = self.state["ee_quaternion"]
        if self.pico:
            # The original PICO mapper anchors in AIRBOT SDK coordinates.
            # Convert its goal to the policy frame only at the boundary.
            sdk_position, sdk_quaternion = self.mapping.arm_pose.write(
                self.state["ee_position"], self.state["ee_quaternion"]
            )
            pico_error = None
            try:
                sample = self.pico.read(sdk_position, sdk_quaternion[[3, 0, 1, 2]])
            except (ValueError, RuntimeError, OSError) as error:
                sample, pico_error = None, str(error)
            if (
                sample is None
                or sample[1]
                or time.monotonic_ns() - sample[2] > 100000000
            ):
                if self.pico_armed:
                    self.fsm.hold(
                        self.state["joint_pos"], pico_error or "PICO input stale"
                    )
                self.core.velocity.fill(0)
                self.pico_motion = self.pico_rl_enabled = False
            else:
                command = sample[0]
                pico_inputs = getattr(self.pico, "input_state", None) or {}
                current_x = bool(pico_inputs.get("left_x"))
                rearm_x = current_x and not self.pico_previous_x
                self.pico_previous_x = current_x
                if self.args.arm_control:
                    cycle.gripper_open_request = pico_inputs.get("gripper_open_request")
                    if cycle.gripper_open_request is not None:
                        cycle.gripper_open_request = bool(cycle.gripper_open_request)
                cycle.pico_arm_enabled = command.arm.enabled
                if command.motion_mode == "motion":
                    was_motion = self.pico_motion
                    self.pico_motion = True
                    self.pico_armed = True
                    cycle.requested = (
                        State.LOCOMOTION
                        if command.dog.enabled and not self.pico_rl_enabled
                        else (
                            State.STAND
                            if not was_motion
                            or (self.pico_rl_enabled and not command.dog.enabled)
                            else cycle.requested
                        )
                    )
                    self.pico_rl_enabled = command.dog.enabled
                    if self.fsm and self.fsm.state == State.HOLD:
                        cycle.requested = State.STAND if rearm_x else None
                    self.core.velocity[:] = [
                        command.dog.forward,
                        command.dog.lateral,
                        command.dog.rotate,
                    ]
                    if (
                        self.args.arm_control
                        and self.arm_feedback.available
                        and command.arm.enabled
                    ):
                        goal = self.mapping.arm_pose.read(
                            command.arm.position, command.arm.orientation
                        )
                        self.core.update_ee_goal(
                            goal["ee_position"],
                            goal["ee_quaternion"],
                            constrain_workspace=False,
                        )
                elif self.pico_armed:
                    self.pico_motion = False
                    self.pico_rl_enabled = False
                    self.core.velocity.fill(0)
                    if self.fsm.state != State.HOLD or pico_inputs.get("right_b"):
                        cycle.requested = State.PASSIVE
        # Fixed-arm commissioning still provides real arm observations and 22 raw actions.

    def _update_posture(self, cycle):
        if self.args.mode in ("observe", "hardware") and not self.args.arm_control:
            self.core.ee_target[:] = self.state["ee_position"]
            self.core.ee_quat[:] = self.state["ee_quaternion"]
        cycle.now = cycle.tick / 1e9
        if (
            self.args.arm_control
            and self.fsm
            and not self.fsm.running_policy
            and (cycle.requested or self.fsm.requested)
            in (State.STAND, State.LOCOMOTION)
            and self.arm_feedback.available
        ):
            self.arm.prepare_control()
        cycle.arm_preparing = bool(
            self.args.arm_control
            and self.arm.control_state == "preparing"
            and self.fsm
            and not self.fsm.running_policy
            and self.arm.error is None
        )
        if (
            self.arm is not None
            and (not self.arm_feedback.available or self.arm_feedback.lost)
            and not cycle.arm_preparing
        ):
            self.core.velocity.fill(0)
            if (
                self.fsm
                and cycle.requested != State.PASSIVE
                and (
                    self.arm_feedback.lost
                    or self.fsm.requested == State.LOCOMOTION
                    or cycle.requested == State.LOCOMOTION
                )
            ):
                self.fsm.hold(
                    self.state["joint_pos"],
                    self.arm_feedback.error or "AIRBOT feedback interrupted",
                )
                cycle.requested = None
        if self.fsm:
            if cycle.requested is not None:
                self.fsm.request(cycle.requested, self.state["joint_pos"], cycle.now)
            if (
                self.fsm.state == State.PASSIVE
                and self.passive_generation != self.fsm.generation
            ):
                self.passive_generation, self.passive_sequence = self.fsm.generation, 0
            self.fsm.advance(
                self.state["joint_pos"],
                self.state["joint_vel"],
                cycle.now,
                passive_ready=bool(cycle.frame.flags & wire.PASSIVE_READY)
                and self.passive_sequence > 0
                and cycle.frame.accepted_command >= self.passive_sequence,
                descending=bool(cycle.frame.flags & wire.DESCENDING),
                policy_ready=not self.args.arm_control
                or (
                    self.arm.control_state == "ready"
                    and self.arm_feedback.available
                    and not self.arm_feedback.lost
                ),
            )
            if self.generation != self.fsm.generation:
                self.generation = self.fsm.generation
                # Clear body velocity on mode changes. Stand/RL share
                # the arm goal; passive/hold discard it.
                self.core.velocity.fill(0)
                if not (cycle.previous_arm_allowed and self.fsm.arm_allowed):
                    self.core.ee_target[:] = self.state["ee_position"]
                    self.core.ee_quat[:] = self.state["ee_quaternion"]
            if not self.fsm.running_policy:
                self.core.velocity.fill(0)
            self.state_name, self.state_phase = self.fsm.state.value, self.fsm.phase
        cycle.workspace_clamped = False
        cycle.workspace_requested = None
        cycle.arm_ready = bool(
            self.args.arm_control
            and self.fsm
            and self.fsm.arm_allowed
            and self.arm.control_state == "ready"
            and self.arm_feedback.available
            and not self.arm_feedback.lost
        )
        if cycle.arm_ready:
            sdk_goal, sdk_quat = self.mapping.arm_pose.write(
                self.core.ee_target, self.core.ee_quat
            )
            cycle.workspace_requested = sdk_goal.copy()
            limited_goal, cycle.workspace_clamped = self.arm_workspace.limit(
                sdk_goal, sdk_quat
            )
            if cycle.workspace_clamped:
                self.workspace_clamps += 1
                limited = self.mapping.arm_pose.read(limited_goal, sdk_quat)
                self.core.update_ee_goal(
                    limited["ee_position"],
                    limited["ee_quaternion"],
                    constrain_workspace=False,
                )
        self.stages.mark("fsm")

    def _infer(self, cycle):
        if (self.fsm is None or self.fsm.running_policy) and (
            self.arm is None or self.arm_feedback.available
        ):
            if self.fsm is None:
                self.state_name, self.state_phase = "locomotion", "policy"
            cycle.obs = self.core.observation(self.state)
            cycle.action = self.core.policy(cycle.obs)
            command_targets = self.core.targets(cycle.action, self.state)
            cycle.q, cycle.dq, cycle.kp, cycle.kd = (
                self.fsm.policy_command(command_targets, cycle.now)
                if self.fsm
                else command_targets
            )
            if self.fsm:
                self.state_phase = self.fsm.phase
        elif self.fsm:
            # No inference occurred in support states; do not serialize
            # hundreds of fictitious zero observations on every heartbeat.
            cycle.obs = cycle.action = None
            cycle.q, cycle.dq, cycle.kp, cycle.kd = self.fsm.command(
                self.state, cycle.now
            )
        else:
            # Read-only observe mode waits for real arm observations too.
            cycle.obs = cycle.action = None
            self.state_name, self.state_phase = "offline", "waiting_arm"
            cycle.q = self.state["joint_pos"].copy()
            cycle.dq = cycle.kp = cycle.kd = np.zeros(22)
        self.stages.mark("policy")

    def _map_command(self, cycle):
        cycle.targets = self.mapping.write(cycle.q, cycle.dq, cycle.kp, cycle.kd)
        if self.body and self.fsm:
            cycle.targets[:, 4] = self.fsm.feedforward
        cycle.arm_target = None
        cycle.arm_submit_accepted = None
        cycle.policy_drop_reason = None
        if cycle.arm_ready:
            cycle.arm_target = self.mapping.arm_pose.write(
                self.core.ee_target, self.core.ee_quat
            )
        # Startup validates the mapping. Packet packing checks finite
        # values; the bridge validates executable gains/wheel mode.
        # Avoid repeating the same target checks in the NX send path.
        cycle.packet = None
        if self.args.mode in ("mock", "hardware"):
            self.sequence += 1
            cycle.packet = wire.pack_command(
                self.session,
                self.sequence,
                time.monotonic_ns(),
                cycle.frame,
                cycle.targets,
                mode=wire.PASSIVE if self.fsm.state == State.PASSIVE else wire.ACTIVE,
            )
            if self.fsm.state == State.PASSIVE and self.passive_sequence == 0:
                self.passive_sequence = self.sequence
        self.stages.mark("mapping")

    def _send(self, cycle):
        if self.controlled:
            if self.arm is not None and (
                self.fsm.running_policy or cycle.arm_target is not None
            ):
                # Recheck immediately before sending: never apply an action
                # computed with arm feedback that has since expired/failed.
                fresh = self.arm_feedback.update(self.arm)
                if (
                    not fresh
                    or self.arm_feedback.lost
                    or time.monotonic_ns() - self.snapshot.sampled_ns
                    > wire.CONTROL_TIMEOUT_NS
                ):
                    cycle.policy_drop_reason = self.arm_feedback.error or (
                        "AIRBOT feedback interrupted or observation aged beyond 100 ms"
                    )
                    self.fsm.hold(self.state["joint_pos"], cycle.policy_drop_reason)
                    cycle.q, cycle.dq, cycle.kp, cycle.kd = self.fsm.command(
                        self.state, cycle.now
                    )
                    cycle.targets = self.mapping.write(
                        cycle.q, cycle.dq, cycle.kp, cycle.kd
                    )
                    cycle.targets[:, 4] = self.fsm.feedforward
                    self.state_name, self.state_phase = (
                        self.fsm.state.value,
                        self.fsm.phase,
                    )
                    cycle.arm_target = None
                    if self.arm_was_active:
                        self.arm.hold_current()
                    self.arm_was_active = self.ee_initialized = False
                    self.passive_generation, self.passive_sequence = (
                        self.fsm.generation,
                        self.sequence,
                    )
                    cycle.packet = wire.pack_command(
                        self.session,
                        self.sequence,
                        time.monotonic_ns(),
                        cycle.frame,
                        cycle.targets,
                        mode=wire.ACTIVE,
                    )
            require_fresh_command(time.monotonic_ns(), cycle.received)
            self.pending_sends[self.sequence] = time.monotonic_ns()
            # ACKs are diagnostic only. Bound retained timestamps even if
            # the bridge keeps publishing feedback without acknowledging.
            self.pending_sends.pop(self.sequence - 10000, None)
            self.link.send(cycle.packet)
            self.last_command_boot = cycle.frame.boot
            self.writes += 1
            if cycle.arm_target is not None:
                cycle.arm_submit_accepted = bool(
                    self.arm.command(
                        *cycle.arm_target, gripper_open=cycle.gripper_open_request
                    )
                )
                self.arm_was_active = True
            elif self.arm_was_active:
                self.arm.hold_current()
                self.arm_was_active = False
        self.command_times.append((time.monotonic_ns() - cycle.tick) / 1e6)
        self.stages.mark("send")

    def _record(self, cycle):
        arm_command_state = None
        if self.args.arm_control:
            if cycle.arm_preparing:
                arm_command_state = "preparing"
            elif not self.arm_feedback.available:
                arm_command_state = "unavailable"
            elif not self.fsm.arm_allowed:
                arm_command_state = "waiting_support"
            elif not cycle.pico_arm_enabled:
                arm_command_state = "waiting_grip"
            elif not cycle.arm_submit_accepted:
                arm_command_state = "queue_busy"
            else:
                arm_command_state = "tracking"
        # Wake JSON serialization only after motor submission. The full
        # cycle timing below still includes log preparation/enqueue time.
        self.recorder.append(cycle_record(self, cycle, arm_command_state))
        self.stages.mark("logging")
        complete_ms = (time.monotonic_ns() - cycle.tick) / 1e6
        self.timings.append(complete_ms)
        self.budget.record(complete_ms)
        self.count += 1


def wait_for_rearm(link, terminal, pico, stopped, state, mapping, failed_boot, status):
    previous_phase = None
    previous_x = bool((getattr(pico, "input_state", None) or {}).get("left_x"))
    position, xyzw = mapping.arm_pose.write(
        state["ee_position"], state["ee_quaternion"]
    )
    while not stopped():
        keys = terminal.read()
        if "\x03" in keys or "\x1b" in keys:
            return False
        manual = "2" in keys or "3" in keys
        input_ready = True
        if pico is not None:
            try:
                sample = pico.read(position, xyzw[[3, 0, 1, 2]])
                input_ready = (
                    sample is not None
                    and not sample[1]
                    and 0 <= time.monotonic_ns() - sample[2] <= wire.CONTROL_TIMEOUT_NS
                )
            except (RuntimeError, OSError):
                input_ready = False
            current_x = bool((getattr(pico, "input_state", None) or {}).get("left_x"))
            manual = input_ready and current_x and not previous_x
            previous_x = current_x
        frame, received, _ = link.snapshot()
        ready = (
            frame is not None
            and (failed_boot is None or frame.boot != failed_boot)
            and 0 <= time.monotonic_ns() - received <= wire.CONTROL_TIMEOUT_NS
            and frame.flags & (wire.MOTORS_VALID | wire.IMU_VALID)
            == wire.MOTORS_VALID | wire.IMU_VALID
            and not frame.flags & (wire.FAULT | wire.DESCENDING)
            and input_ready
        )
        phase = "waiting_rearm" if ready else "waiting_bridge"
        if phase != previous_phase:
            status(phase)
            previous_phase = phase
        if ready and manual and link.confirm_restart(frame.boot):
            return True
        time.sleep(0.02)
    return False
