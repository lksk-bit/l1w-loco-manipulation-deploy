"""AIRBOT process communication, feedback freshness and gripper commands."""

import json
import math
import os
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from .mapping import quaternion, DEFAULT_MARGIN_M, ArmWorkspace
from .bridge import CONTROL_TIMEOUT_NS


DEFAULT_GRIPPER = {
    "open_m": 0.072,
    "closed_m": 0.0,
    "speed_m_s": 0.03,
}


GRIPPER_MAX_LEAD_M = 0.001


def gripper_settings(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - DEFAULT_GRIPPER.keys():
        raise ValueError("invalid gripper settings")
    result = {**DEFAULT_GRIPPER, **value}
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in result.values()
    ):
        raise ValueError("gripper settings must be finite numbers")
    if not all(math.isfinite(v) for v in result.values()) or not (
        0 <= result["closed_m"] < result["open_m"] <= 0.072
        and 0 < result["speed_m_s"] <= 0.072
    ):
        raise ValueError("invalid G2 travel/speed bounds")
    return {k: float(v) for k, v in result.items()}


def trigger_open(value, previous=False):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("invalid right Trigger")
    return value >= (0.4 if previous else 0.5)


def gripper_button_command(trigger, close_button, previous=None):
    """Return a held open/close command, or None when both controls are released."""
    if not isinstance(close_button, bool):
        raise TypeError("invalid right stick button")
    if previous is not None and not isinstance(previous, bool):
        raise TypeError("invalid previous gripper command")
    if close_button:
        return False
    if trigger_open(trigger, previous is True):
        return True
    return None


class GripperControl:
    def __init__(self, settings):
        self.settings = gripper_settings(settings)
        self.target = None
        self.armed = False
        self.force_send = False
        self.last_ns = None
        self.last_opening = None
        self.state = "waiting_input"

    def pause(self, position=None, *, unavailable=False):
        self.target = position
        self.armed = False
        self.force_send = False
        self.last_ns = None
        self.last_opening = None
        self.state = "unavailable" if unavailable else "waiting_input"

    def step(self, opening, position, effort, now_ns):
        c = self.settings
        if (
            not all(math.isfinite(v) for v in (position, effort))
            or not -0.002 <= position <= 0.074
        ):
            raise ValueError("invalid G2 feedback")
        position = max(0.0, min(0.072, position))
        dt = (
            0.02
            if self.last_ns is None
            else max(0, min((now_ns - self.last_ns) / 1e9, 0.1))
        )
        self.last_ns = now_ns
        if self.target is None:
            self.target = position

        if opening is None:
            released = self.last_opening is not None
            self.target = position
            self.armed = released
            self.force_send = released
            self.last_opening = None
            self.state = "holding" if released else "waiting_input"
            return self.target
        if not isinstance(opening, bool):
            raise ValueError("gripper direction must be boolean")  # noqa: TRY004

        if opening != self.last_opening:
            self.target = position
        self.last_opening = opening
        self.armed = True
        self.force_send = False
        change = c["speed_m_s"] * dt
        if opening:
            goal = c["open_m"]
            self.target = min(
                goal,
                position + GRIPPER_MAX_LEAD_M,
                max(self.target, position) + change,
            )
            self.state = (
                "open"
                if abs(position - goal) <= 0.0001 and abs(self.target - goal) < 1e-7
                else "opening"
            )
        else:
            goal = c["closed_m"]
            self.target = max(
                goal,
                position - GRIPPER_MAX_LEAD_M,
                min(self.target, position) - change,
            )
            self.state = (
                "closed"
                if abs(position - goal) <= 0.0001 and abs(self.target - goal) < 1e-7
                else "closing"
            )
        return self.target


@dataclass(frozen=True)
class ArmSnapshot:
    q: np.ndarray
    dq: np.ndarray
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    # Start of the two SDK reads, so blocking latency counts towards state age.
    sampled_ns: int
    poll_ms: float
    gripper: tuple | None = None  # position, velocity, SDK effort


class ArmFeedback:
    """Cache the last valid arm pose for body RL; freshness authorizes arm IO."""

    def __init__(self):
        self.snapshot = None
        self.available = False
        self.lost = False
        self.error = "waiting for AIRBOT feedback"
        self.disconnects = 0

    def update(self, worker, now_ns=None, *, joint_bounds=None):
        sample = worker.snapshot()
        if now_ns is None:
            now_ns = time.monotonic_ns()
        disconnects = getattr(worker, "disconnects", 0)
        interrupted = disconnects != self.disconnects
        self.disconnects = disconnects
        error = worker.error
        if error is None and (
            sample is None or not 0 <= now_ns - sample.sampled_ns <= CONTROL_TIMEOUT_NS
        ):
            error = "AIRBOT feedback stale/unavailable"
        if error is None and joint_bounds is not None:
            lower, upper = (np.asarray(v) for v in joint_bounds)
            bad = np.flatnonzero((sample.q < lower) | (sample.q > upper))
            if bad.size:
                details = "; ".join(
                    f"J{i + 1}={sample.q[i]:+.5f} rad, "
                    f"range=[{lower[i]:+.5f}, {upper[i]:+.5f}], "
                    f"excess={max(lower[i] - sample.q[i], sample.q[i] - upper[i]):.5f} rad"
                    for i in bad
                )
                error = "AIRBOT measured joints exceed SDK bounds: " + details
        available = error is None
        self.lost = self.available and (interrupted or not available)
        self.available, self.error = available, error
        if available:
            self.snapshot = sample
        return available


STATUS_FD_ENV = "L1W_RL_ARM_STATUS_FD"


PROCESS_STALL_SECONDS = 10


class ArmWorker:
    def __init__(
        self,
        control=False,
        *,
        workspace_margin_m=DEFAULT_MARGIN_M,
        gripper_config=None,
        _child_command=None,
    ):
        self.control = control
        self.workspace = ArmWorkspace(workspace_margin_m)
        self.gripper_config = (
            gripper_settings(gripper_config) if gripper_config is not None else None
        )
        self.gripper_status = None
        self.lock = threading.Lock()
        self.latest = None
        self.error = self.last_error = None
        self.disconnects = 0
        self.sent_count = 0
        self.rejected_count = 0
        self.last_rejection = None
        self.control_state = "idle"
        self.prepare_pending = False
        self.return_zero_state = self.return_zero_error = None
        self.process = None
        self.process_pid = None
        self.stop = threading.Event()
        self.child_command = _child_command or [
            sys.executable,
            "-u",
            "-m",
            "l1w_teleop.rl.airbot_sdk",
            "--sdk-child",
            *(["--control"] if control else []),
            "--workspace-margin-m",
            str(workspace_margin_m),
            *(
                ["--gripper-config", json.dumps(self.gripper_config)]
                if self.gripper_config
                else []
            ),
        ]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _failed(self, error):
        with self.lock:
            self.error = self.last_error = str(error)
            self.disconnects += 1
            self.latest = None
            self.control_state = "idle"
            self.prepare_pending = False
            if self.gripper_config is not None:
                self.gripper_status = {"state": "unavailable", "error": str(error)}
            if self.return_zero_state in ("pending", "moving"):
                self.return_zero_state, self.return_zero_error = "failed", str(error)

    def _accept(self, payload, previous_disconnects):
        sample = payload["sample"]
        state = None
        if sample is not None:
            arrays = [
                np.asarray(sample[name], dtype=float)
                for name in ("q", "dq", "position", "quaternion_xyzw")
            ]
            if [a.shape for a in arrays] != [(6,), (6,), (3,), (4,)] or not all(
                np.isfinite(a).all() for a in arrays
            ):
                raise ValueError("invalid AIRBOT process feedback")
            state = ArmSnapshot(
                *arrays,
                int(sample["sampled_ns"]),
                float(sample["poll_ms"]),
                sample.get("gripper"),
            )
        disconnects = int(payload["disconnects"])
        with self.lock:
            control_state = payload.get("control_state", "idle")
            if (
                control_state == "ready"
                or payload["error"] is not None
                or disconnects > previous_disconnects
            ):
                self.prepare_pending = False
            self.control_state = "preparing" if self.prepare_pending else control_state
            self.disconnects += max(0, disconnects - previous_disconnects)
            self.latest = state
            self.error = payload["error"]
            self.last_error = payload["last_error"] or self.last_error
            self.sent_count = int(payload.get("sent_count", 0))
            self.rejected_count = int(payload.get("rejected_count", 0))
            self.last_rejection = payload.get("last_rejection")
            self.return_zero_state = payload.get("return_zero_state")
            self.return_zero_error = payload.get("return_zero_error")
            self.gripper_status = payload.get("gripper_status")
        return disconnects

    @staticmethod
    def _close_process(process):
        process.stdin.close()  # EOF asks the child to close its SDK session.
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.2)

    def _run(self):
        while not self.stop.is_set():
            read_fd, write_fd = os.pipe()
            process = None
            try:
                env = dict(os.environ)
                env.update(
                    {
                        STATUS_FD_ENV: str(write_fd),
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    }
                )
                process = subprocess.Popen(
                    self.child_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    pass_fds=(write_fd,),
                    env=env,
                    start_new_session=True,
                )
                os.close(write_fd)
                write_fd = None
                os.set_blocking(process.stdin.fileno(), False)
                with self.lock:
                    self.process = process
                    self.process_pid = process.pid
                buffer = b""
                last_message = time.monotonic()
                previous_disconnects = 0
                while not self.stop.is_set():
                    if not select.select([read_fd], [], [], 0.05)[0]:
                        if time.monotonic() - last_message > PROCESS_STALL_SECONDS:
                            raise RuntimeError(
                                "AIRBOT SDK process unresponsive for 10 s"
                            )
                        continue
                    data = os.read(read_fd, 65536)
                    if not data:
                        code = process.poll()
                        raise RuntimeError(f"AIRBOT SDK process exited (code={code})")
                    lines = (buffer + data).split(b"\n")
                    buffer = lines.pop()
                    if len(buffer) > 16384:
                        raise ValueError("AIRBOT process status exceeded size limit")
                    if lines:
                        # Keep the most recent full status; its cumulative error
                        # counter still exposes disconnects between publications.
                        previous_disconnects = self._accept(
                            json.loads(lines[-1]), previous_disconnects
                        )
                        last_message = time.monotonic()
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                if not self.stop.is_set():
                    self._failed(error)
            finally:
                with self.lock:
                    self.process = None
                os.close(read_fd)
                if write_fd is not None:
                    os.close(write_fd)
                if process is not None:
                    try:
                        self._close_process(process)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        self._failed(f"AIRBOT process cleanup failed: {error}")
            self.stop.wait(1)

    def snapshot(self):
        with self.lock:
            return self.latest if self.error is None else None

    def _send(self, message, *, require_feedback=True):
        data = (json.dumps(message, allow_nan=False) + "\n").encode()
        # Commands fit below PIPE_BUF; a full nonblocking pipe rejects the whole
        # write instead of blocking control or leaving a partial JSON record.
        if len(data) > 4096:
            raise ValueError("AIRBOT process command too large")
        with self.lock:
            process = self.process
            if process is None or (
                require_feedback
                and (
                    self.error is not None
                    or self.latest is None
                    or time.monotonic_ns() - self.latest.sampled_ns > CONTROL_TIMEOUT_NS
                )
            ):
                return False
            try:
                return os.write(process.stdin.fileno(), data) == len(data)
            except OSError:
                return False

    def command(self, position, quaternion_xyzw, *, gripper_open=None):
        if not self.control:
            raise RuntimeError("AIRBOT control is not enabled")
        if self.control_state == "preparing":
            return False
        p = np.asarray(position, dtype=float)
        q = quaternion(quaternion_xyzw)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("invalid AIRBOT Cartesian target")
        p, _ = self.workspace.limit(p, q)
        if gripper_open is not None and not isinstance(gripper_open, bool):
            raise ValueError("gripper open request must be boolean")
        return self._send(
            {
                "kind": "target",
                "submitted_ns": time.monotonic_ns(),
                "position": p.tolist(),
                "quaternion_xyzw": q.tolist(),
                "gripper_open": gripper_open,
            }
        )

    def prepare_control(self):
        if not self.control:
            return False
        if self.control_state in ("preparing", "ready"):
            return True
        if not self._send({"kind": "prepare"}):
            return False
        with self.lock:
            self.prepare_pending = True
            self.control_state = "preparing"
        return True

    def hold_current(self):
        if self.control:
            self._send({"kind": "hold"})

    def return_zero(self, timeout=8.0, cancelled=lambda: False):
        """Exit-only bounded wait. All SDK work remains in the SDK process."""
        if not self.control or not self._send({"kind": "return_zero"}):
            self.return_zero_error = "AIRBOT unavailable; return-zero was not requested"
            return False
        deadline = time.monotonic() + timeout
        disconnects = self.disconnects
        while (
            time.monotonic() < deadline and not self.stop.is_set() and not cancelled()
        ):
            with self.lock:
                state, error = self.return_zero_state, self.error
            if state == "done":
                return True
            if (
                state == "failed"
                or error is not None
                or self.disconnects != disconnects
            ):
                self.return_zero_error = (
                    self.return_zero_error
                    or error
                    or "AIRBOT disconnected during return-zero"
                )
                return False
            self.stop.wait(0.02)
        self.return_zero_error = (
            "AIRBOT return-zero cancelled"
            if cancelled()
            else "AIRBOT return-zero feedback did not settle before timeout"
        )
        return False

    def close(self):
        self._send({"kind": "stop"}, require_feedback=False)
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            self._failed("AIRBOT process supervisor did not exit within 2 s")
