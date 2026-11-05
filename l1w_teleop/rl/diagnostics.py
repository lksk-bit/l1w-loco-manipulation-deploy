"""Asynchronous recording and control-cycle diagnostics."""

import contextlib
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path
from time import thread_time_ns

import numpy as np

from . import bridge as wire
from .body import State
from .bridge import CONTROL_TIMEOUT_MS, CONTROL_TIMEOUT_NS


class FeedbackStale(RuntimeError):
    """Skip submission and hold locally; never request descent for this timeout."""


class CycleBudget:
    PERIOD_MS = 20.0

    def __init__(self):
        self.overruns = 0
        self.consecutive = 0
        self.max_consecutive = 0

    def record(self, elapsed_ms):
        if elapsed_ms >= self.PERIOD_MS:
            self.overruns += 1
            self.consecutive += 1
            self.max_consecutive = max(self.max_consecutive, self.consecutive)
        else:
            self.consecutive = 0


class CycleStages:
    NAMES = ("input", "feedback", "fsm", "policy", "mapping", "send", "logging")

    def __init__(self):
        self.samples = {name: deque(maxlen=10000) for name in self.NAMES}
        self.cpu_samples = {name: deque(maxlen=10000) for name in self.NAMES}
        self.previous_ns = 0
        self.previous_cpu_ns = 0

    def start(self, now_ns):
        self.previous_ns = now_ns
        self.previous_cpu_ns = thread_time_ns()

    def mark(self, name):
        now = time.monotonic_ns()
        cpu_now = thread_time_ns()
        self.samples[name].append((now - self.previous_ns) / 1e6)
        self.cpu_samples[name].append((cpu_now - self.previous_cpu_ns) / 1e6)
        self.previous_ns = now
        self.previous_cpu_ns = cpu_now


def require_fresh_command(now_ns, received_ns, arm_sampled_ns=None):
    """Recheck feedback at transmission, after computation/scheduling delays."""
    if now_ns - received_ns > CONTROL_TIMEOUT_NS:
        raise FeedbackStale(
            f"bridge feedback aged beyond {CONTROL_TIMEOUT_MS} ms before submission; command was not sent"
        )
    if arm_sampled_ns is not None and now_ns - arm_sampled_ns > CONTROL_TIMEOUT_NS:
        raise FeedbackStale(
            f"AIRBOT feedback aged beyond {CONTROL_TIMEOUT_MS} ms before submission; command was not sent"
        )


@contextlib.contextmanager
def log_file(path):
    if path:
        with Path(path).open("x") as file:
            yield file
    else:
        yield None


class Recorder:
    def __init__(self, path, status=False):
        self.queue = queue.Queue(maxsize=128)
        self.dropped = 0
        self.error = None
        self.status = status
        self.resources = contextlib.ExitStack()
        self.file = self.resources.enter_context(log_file(path))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        previous = None
        last_status_at = 0.0
        while True:
            item = self.queue.get()
            if item is None:
                break
            status = (
                item["state"],
                item["phase"],
                item["requested"],
                item.get("arm_available"),
                item.get("arm_error"),
                item.get("arm_command_state"),
                item.get("arm_target_rejections"),
                (item.get("gripper_status") or {}).get("state"),
                item.get("hold_reason"),
            )
            now = time.monotonic()
            if self.status and (
                status != previous
                or (
                    status[5] in ("tracking", "waiting_grip", "queue_busy")
                    and now - last_status_at >= 1.0
                )
            ):
                gravity = item.get("projected_gravity")
                imu_status = (
                    " gravity=(" + ", ".join(f"{v:.3f}" for v in gravity) + ")"
                    if gravity is not None
                    else ""
                )
                print(
                    f"RL state={status[0]} phase={status[1]} requested={status[2]}{imu_status}"
                    + (
                        f" arm={'ready' if status[3] else 'unavailable'}"
                        if status[3] is not None
                        else ""
                    )
                    + (f" arm_error={status[4]}" if status[4] else "")
                    + (f" arm_command={status[5]}" if status[5] else ""),
                    flush=True,
                )
                if item.get("recovery_reason"):
                    print(
                        f"Control session stopped: {item['recovery_reason']}",
                        flush=True,
                    )
                if item.get("hold_reason"):
                    print(f"Control paused: {item['hold_reason']}", flush=True)
                if status[5] in ("tracking", "waiting_grip", "queue_busy"):
                    inputs = item.get("pico_input") or {}
                    goal = item.get("arm_sdk_target")
                    feedback = item.get("arm_sdk_feedback")
                    print(
                        "AIRBOT "
                        f"right_grip={inputs.get('right_grip')} "
                        f"input_enabled={item.get('pico_arm_enabled')} "
                        f"reference_ready={inputs.get('arm_reference_ready')} "
                        f"queued={item.get('arm_submit_accepted')} "
                        f"sdk_sent={item.get('arm_sdk_sent_count')} "
                        f"sdk_rejected={item.get('arm_target_rejections', 0)} "
                        f"right_trigger={inputs.get('right_trigger')} "
                        f"right_stick_click={inputs.get('right_stick_click')} "
                        f"gripper={item.get('gripper_status')} "
                        f"workspace_clamped={item.get('arm_workspace_clamped', False)} "
                        f"target={tuple(round(float(v), 3) for v in goal[0]) if goal else None} "
                        f"feedback={tuple(round(float(v), 3) for v in feedback[0]) if feedback else None}",
                        flush=True,
                    )
                previous = status
                last_status_at = now
            if self.file:
                try:
                    self.file.write(
                        json.dumps(item, allow_nan=False, default=lambda a: a.tolist())
                        + "\n"
                    )
                except (OSError, ValueError) as error:
                    self.error = str(error)
        self.resources.close()

    def append(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def close(self):
        self.queue.put(None)
        self.thread.join()


def cycle_record(control, cycle, arm_command_state):
    return {
        "cycle": control.count,
        "state_seq": cycle.frame.sequence if cycle.frame else 0,
        "bridge_boot": cycle.frame.boot if cycle.frame else 0,
        "command_session": control.session if control.controlled else 0,
        "state": control.state_name,
        "phase": control.state_phase,
        "requested": control.fsm.requested.value if control.fsm else "locomotion",
        "body_profile_sha256": control.body.sha256 if control.body else None,
        "mapping_profile_sha256": control.mapping.sha256,
        "policy_algorithm": control.cfg["algorithm"]["name"],
        "policy_sha256": control.cfg["algorithm"]["sha256"],
        "trial": control.args.trial,
        "arm_observation_reference": control.core.observation_reference[16:22].copy(),
        "joint_pos": control.state["joint_pos"].copy(),
        "joint_vel": control.state["joint_vel"].copy(),
        "imu_quaternion_raw": cycle.frame.quaternion if cycle.frame else None,
        "imu_quaternion_order": control.mapping.imu_order(cycle.frame)
        if cycle.frame
        else None,
        "imu_acceleration_raw": cycle.frame.acceleration if cycle.frame else None,
        "imu_gyro_raw": cycle.frame.gyro if cycle.frame else None,
        "projected_gravity": control.state["projected_gravity"].copy(),
        "local_ang_vel": control.state["local_ang_vel"].copy(),
        "velocity_command": control.core.velocity.copy(),
        "motor_pos_raw": cycle.frame.q if cycle.frame else None,
        "bridge_flags": cycle.frame.flags if cycle.frame else 0,
        "obs": cycle.obs,
        "action": cycle.action,
        "policy_action_applied": bool(
            control.controlled and control.fsm.running_policy
        ),
        "policy_drop_reason": cycle.policy_drop_reason,
        "hold_reason": control.fsm.hold_reason
        if control.fsm and control.fsm.state == State.HOLD
        else None,
        "motor_targets": cycle.targets,
        "ee_position": control.state["ee_position"],
        "ee_target": control.core.ee_target.copy(),
        "arm_sdk_target": cycle.arm_target,
        "arm_workspace_clamped": cycle.workspace_clamped,
        "arm_workspace_requested": cycle.workspace_requested,
        "arm_target_rejections": getattr(control.arm, "rejected_count", 0),
        "arm_last_rejection": getattr(control.arm, "last_rejection", None),
        "pico_input": getattr(control.pico, "input_state", None),
        "pico_arm_enabled": cycle.pico_arm_enabled,
        "arm_command_state": arm_command_state,
        "arm_submit_accepted": cycle.arm_submit_accepted,
        "arm_sdk_sent_count": getattr(control.arm, "sent_count", 0),
        "arm_sdk_feedback": (
            control.snapshot.position,
            control.snapshot.quaternion_xyzw,
        )
        if control.arm and control.snapshot is not None
        else None,
        "gripper_open_request": cycle.gripper_open_request,
        "gripper_status": getattr(control.arm, "gripper_status", None),
        "gripper_feedback": control.snapshot.gripper
        if control.arm and control.snapshot is not None
        else None,
        "arm_available": control.arm_feedback.available if control.arm else None,
        "arm_error": control.arm_feedback.error if control.arm else None,
        "arm_control_state": control.arm.control_state
        if control.args.arm_control
        else None,
        "arm_feedback_sampled_ns": control.snapshot.sampled_ns
        if control.snapshot
        else None,
        "arm_disconnects": control.arm_feedback.disconnects if control.arm else 0,
    }


def deployment_report(control):
    def summary(values):
        return {
            "max": float(max(values, default=0)),
            "p99": float(np.percentile(values, 99)) if values else 0,
            "p999": float(np.percentile(values, 99.9)) if values else 0,
        }

    report = {
        "mode": control.args.mode,
        "policy": control.cfg["algorithm"]["policy"],
        "policy_algorithm": control.cfg["algorithm"]["name"],
        "policy_sha256": control.cfg["algorithm"]["sha256"],
        "body_profile_sha256": control.body.sha256 if control.body else None,
        "mapping_profile_sha256": control.mapping.sha256,
        "imu_quaternion_order": control.mapping.imu_order(control.last_frame)
        if control.last_frame
        else None,
        "trial": control.args.trial,
        "arm_observation_reference": control.core.observation_reference[16:22].tolist(),
        "stand_trajectory": {
            "curve": "two_stage_cubic",
            "hip_return_seconds": control.fsm.hip_duration,
            "rise_seconds": control.fsm.duration - control.fsm.hip_duration,
            "total_seconds": control.fsm.duration,
        }
        if control.fsm
        else None,
        "stand_settling": {
            "required": control.fsm.require_settling,
            "position_tolerance_rad": control.fsm.position_tolerance,
            "velocity_tolerance_rad_s": control.fsm.velocity_tolerance,
            "stable_seconds": control.fsm.stable_duration,
            "timeout_seconds": control.fsm.timeout,
        }
        if control.fsm
        else None,
        "final_state": control.fsm.state.value if control.fsm else "offline",
        "fault_reason": control.fsm.reason if control.fsm else None,
        "hold_reason": control.fsm.hold_reason if control.fsm else None,
        "bridge_failure_snapshot": control.bridge_failure_snapshot,
        "recovery_events": list(control.recovery_events),
        "steps_recorded": len(control.timings),
        "quantile_window": 10000,
        "work_ms": summary(control.timings),
        "command_work_ms": summary(control.command_times),
        "stage_work_ms": {
            name: summary(values) for name, values in control.stages.samples.items()
        },
        "stage_thread_cpu_ms": {
            name: summary(values) for name, values in control.stages.cpu_samples.items()
        },
        "deadline_limits_ms": {
            "period": control.budget.PERIOD_MS,
            "policy": None,
            "passive_stand": None,
            "command_ttl": wire.CONTROL_TIMEOUT_MS,
            "feedback_age": wire.CONTROL_TIMEOUT_MS,
            "command_ack": None,
        },
        "timeout_action": "hold_without_descent",
        "wake_lateness_ms": summary(control.wake_late),
        "state_age_ms": summary(control.ages),
        "arm_feedback_age_ms": summary(control.arm_ages),
        "arm_poll_ms": summary(control.arm_polls),
        "arm_worker_error": control.arm.error if control.arm else None,
        "arm_sdk_process_pid": getattr(control.arm, "process_pid", None),
        "pico_sdk_process_pid": getattr(control.pico, "process_pid", None),
        "pico_worker_error": getattr(control.pico, "error", None),
        "arm_last_error": getattr(control.arm, "last_error", None),
        "arm_workspace": {
            "frame": "AIRBOT SDK base",
            "margin_m": control.arm_workspace.margin,
            "wrist_min_radius_m": control.arm_workspace.minimum,
            "wrist_max_radius_m": control.arm_workspace.maximum,
            "clamped_cycles": control.workspace_clamps,
        },
        "arm_exit_result": control.arm_exit_result,
        "gripper_config": control.arm_config["gripper"]
        if control.args.arm_control and control.args.pico
        else None,
        "gripper_status": getattr(control.arm, "gripper_status", None),
        "arm_target_rejections": getattr(control.arm, "rejected_count", 0),
        "arm_last_rejection": getattr(control.arm, "last_rejection", None),
        "arm_disconnects": getattr(control.arm, "disconnects", 0),
        "validation_ack_rtt_ms": summary(control.ack_times),
        "overruns": control.budget.overruns,
        "max_consecutive_overruns": control.budget.max_consecutive,
        "writes": control.writes,
        "log_dropped": control.recorder.dropped if control.recorder else 0,
        "log_error": control.recorder.error if control.recorder else None,
    }
    return report
