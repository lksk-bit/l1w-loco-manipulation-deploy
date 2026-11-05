"""PICO samples, source polling and isolated SDK child entry point."""

import json
import math
import os
import select
import sys
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, replace

from .teleop import ControllerFrame, TeleopInputs, ArmCommand, DogCommand, RobotCommands
from typing import Tuple

STATUS_FD_ENV = "L1W_RL_PICO_STATUS_FD"


@dataclass
class SourceState:
    timestamp_ns: int = 0
    left_pose: tuple = (0.0,) * 7
    right_pose: tuple = (0.0,) * 7
    left_grip: float = 0.0
    right_grip: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_axis: tuple = (0.0, 0.0)
    right_axis: tuple = (0.0, 0.0)
    left_buttons: tuple = (False, False)
    right_buttons: tuple = (False, False)
    left_axis_click: bool = False
    right_axis_click: bool = False


class InputSource:
    def read(self) -> SourceState:
        raise NotImplementedError

    def close(self):
        pass


class MockInputSource(InputSource):
    """Produces a small deterministic motion for offline UI tests."""

    def __init__(self, rate_hz: float = 50.0):
        self._period_ns = int(1_000_000_000 / rate_hz)
        self._tick = 0
        self._started_ns = time.monotonic_ns()

    def read(self) -> SourceState:
        self._tick += 1
        elapsed = self._tick * self._period_ns
        value = min(1.0, self._tick / 25.0)
        return SourceState(
            timestamp_ns=elapsed,
            left_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            right_pose=(value * 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            left_grip=value if self._tick % 100 < 50 else 0.0,
            right_grip=value,
            left_trigger=0.0,
            right_trigger=0.5,
            left_axis=(0.0, value if self._tick % 200 < 100 else 0.0),
            right_axis=(0.0, 0.0),
            left_buttons=(self._tick % 400 < 200, False),
            right_buttons=(self._tick % 600 < 300, False),
            left_axis_click=False,
            right_axis_click=False,
        )


class XRobotToolkitSource(InputSource):
    """Reads the official PICO PC-Service pybind API."""

    def __init__(self):
        import xrobotoolkit_sdk as xrt

        self._xrt = xrt
        self._xrt.init()

    def read(self) -> SourceState:
        xrt = self._xrt
        return SourceState(
            timestamp_ns=int(xrt.get_time_stamp_ns()),
            left_pose=tuple(xrt.get_left_controller_pose()),
            right_pose=tuple(xrt.get_right_controller_pose()),
            left_grip=float(xrt.get_left_grip()),
            right_grip=float(xrt.get_right_grip()),
            left_trigger=float(xrt.get_left_trigger()),
            right_trigger=float(xrt.get_right_trigger()),
            left_axis=tuple(xrt.get_left_axis()),
            right_axis=tuple(xrt.get_right_axis()),
            left_buttons=(bool(xrt.get_X_button()), bool(xrt.get_Y_button())),
            right_buttons=(bool(xrt.get_A_button()), bool(xrt.get_B_button())),
            left_axis_click=bool(xrt.get_left_axis_click()),
            right_axis_click=bool(xrt.get_right_axis_click()),
        )

    def close(self):
        self._xrt.close()


class PicoInputMonitor:
    """Adds local timing, finite-value checks, and stale-data detection."""

    def __init__(self, source: InputSource, stale_after_s: float = 0.30):
        self._source = source
        self._stale_after_ns = int(stale_after_s * 1_000_000_000)
        self._last_timestamp_ns = None
        self._last_fingerprint = None
        self._first_unchanged_ns = None

    def read(self) -> Tuple[TeleopInputs, bool]:
        state = self._source.read()
        local_ns = time.monotonic_ns()
        fingerprint = (
            state.timestamp_ns,
            state.left_pose,
            state.right_pose,
            state.left_grip,
            state.right_grip,
            state.left_trigger,
            state.right_trigger,
            state.left_axis,
            state.right_axis,
            state.left_buttons,
            state.right_buttons,
            state.left_axis_click,
            state.right_axis_click,
        )
        changed = fingerprint != self._last_fingerprint
        self._last_timestamp_ns = state.timestamp_ns
        if changed or self._first_unchanged_ns is None:
            self._first_unchanged_ns = local_ns
        self._last_fingerprint = fingerprint
        stale = local_ns - self._first_unchanged_ns > self._stale_after_ns
        frames = []
        for side in ("left", "right"):
            pose = tuple(getattr(state, f"{side}_pose"))
            axis = tuple(getattr(state, f"{side}_axis"))
            grip = float(getattr(state, f"{side}_grip"))
            trigger = float(getattr(state, f"{side}_trigger"))
            buttons = tuple(bool(v) for v in getattr(state, f"{side}_buttons"))
            axis_click = bool(getattr(state, f"{side}_axis_click"))
            finite = all(_finite(value) for value in pose + axis + (grip, trigger))
            if not finite:
                pose = (0.0,) * 7
                axis = (0.0, 0.0)
                grip = 0.0
                trigger = 0.0
                stale = True
            frames.append(
                ControllerFrame(
                    local_monotonic_ns=local_ns,
                    sdk_timestamp_ns=state.timestamp_ns,
                    pose=pose,
                    grip=grip,
                    trigger=trigger,
                    axis=axis,
                    buttons=buttons,
                    axis_click=axis_click,
                )
            )
        return TeleopInputs(left=frames[0], right=frames[1]), stale

    def close(self):
        self._source.close()


def _finite(value) -> bool:
    try:
        return math.isfinite(value)
    except TypeError:
        return False


class PicoWorker:
    def __init__(self, *, _child_command=None):
        self.lock = threading.Lock()
        self.latest = None
        self.pose = None
        self.error = None
        self.input_state = None
        self.process_pid = None
        self.stop = threading.Event()
        self.child_command = _child_command or [
            sys.executable,
            "-u",
            "-m",
            "l1w_teleop.rl.pico_sdk",
            "--sdk-child",
        ]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _accept(self, payload):
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        value = payload["command"]
        command = RobotCommands(
            dog=DogCommand(**value["dog"]),
            arm=ArmCommand(**value["arm"]),
            arm_mode_command=value["arm_mode_command"],
            dog_mode_command=value["dog_mode_command"],
            motion_mode=value["motion_mode"],
        )
        # Preserve the child sample time: IPC delivery never makes old input new.
        sample = command, bool(payload["stale"]), int(payload["sampled_ns"])
        with self.lock:
            self.latest = sample
            self.input_state = payload.get("input")
            self.error = None

    @staticmethod
    def _close_process(process):
        process.stdin.close()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.2)

    def _run(self):
        while not self.stop.is_set():
            self._session()
            self.stop.wait(1)

    def _session(self):
        read_fd, write_fd = os.pipe()
        process = None
        try:
            env = dict(os.environ)
            env[STATUS_FD_ENV] = str(write_fd)
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
            self.process_pid = process.pid
            buffer = b""
            last_message = time.monotonic()
            while not self.stop.is_set():
                with self.lock:
                    pose = self.pose
                if pose is not None:
                    data = (json.dumps(pose, allow_nan=False) + "\n").encode()
                    if len(data) > 4096:
                        raise ValueError("PICO pose record exceeded PIPE_BUF")
                    try:
                        os.write(process.stdin.fileno(), data)
                    except BlockingIOError:
                        pass  # Do not queue stale arm poses behind a slow SDK.
                if not select.select([read_fd], [], [], 0.01)[0]:
                    if time.monotonic() - last_message > 10:
                        raise RuntimeError("PICO SDK process unresponsive for 10 s")
                    continue
                data = os.read(read_fd, 65536)
                if not data:
                    raise RuntimeError(
                        f"PICO SDK process exited (code={process.poll()})"
                    )
                lines = (buffer + data).split(b"\n")
                buffer = lines.pop()
                if len(buffer) > 16384:
                    raise ValueError("PICO process status exceeded size limit")
                if lines:
                    self._accept(json.loads(lines[-1]))
                    last_message = time.monotonic()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            if not self.stop.is_set():
                with self.lock:
                    self.error = str(error)
                    self.latest = None
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
            if process is not None:
                try:
                    self._close_process(process)
                except (OSError, subprocess.TimeoutExpired) as error:
                    with self.lock:
                        self.error = f"PICO process cleanup failed: {error}"

    def read(self, position, quaternion_wxyz):
        q = quaternion_wxyz
        p = tuple(float(v) for v in position)
        xyzw = (float(q[1]), float(q[2]), float(q[3]), float(q[0]))
        if len(p) != 3 or not all(math.isfinite(v) for v in p + xyzw):
            raise ValueError("invalid PICO arm reference pose")
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            self.pose = {
                "position": p,
                "quaternion_xyzw": xyzw,
                "sampled_ns": time.monotonic_ns(),
            }
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            with self.lock:
                self.error = "PICO process supervisor did not exit within 2 s"


def sdk_child():
    # Never import or initialize the native binding in the body process.
    from l1w_teleop.rl.teleop import InputMapping
    from l1w_teleop.rl.teleop import MappingConfig
    from .arm import gripper_button_command

    monitor = None
    status = os.fdopen(int(os.environ[STATUS_FD_ENV]), "wb", buffering=0)
    os.set_blocking(status.fileno(), False)
    # Match the original hardware entry point: one-to-one translation with the
    # SDK return-zero orientation held fixed. Input/targets here use SDK frames.
    # RL observes both arm state/goal and body velocity; keep both input paths
    # enabled together instead of stopping locomotion when the grip is held.
    mapper = InputMapping(
        MappingConfig(
            allow_return_zero=False,
            lock_orientation=True,
            arm_motion_stops_dog=False,
        )
    )
    buffer = b""
    pose = None
    gripper_open = None
    try:
        monitor = PicoInputMonitor(XRobotToolkitSource(), stale_after_s=0.3)
        while True:
            if select.select([0], [], [], 0)[0]:
                data = os.read(0, 65536)
                if not data:
                    break
                lines = (buffer + data).split(b"\n")
                buffer = lines.pop()
                if len(buffer) > 4096:
                    raise ValueError("invalid PICO pose record")
                if lines:
                    pose = json.loads(lines[-1])
            inputs, stale = monitor.read()
            sampled_ns = time.monotonic_ns()
            current_pose = None
            if pose and sampled_ns - pose["sampled_ns"] <= 100000000:
                current_pose = pose["position"], pose["quaternion_xyzw"]
            command = mapper.map(
                inputs, input_healthy=not stale, current_arm_pose=current_pose
            )
            if stale:
                # Do not replay a latched jaw direction after headset/input
                # recovery; wait for a new explicit open or close button.
                gripper_open = None
            else:
                gripper_open = gripper_button_command(
                    inputs.right.trigger,
                    bool(inputs.right.axis_click),
                    gripper_open,
                )
            # Hardware testing found the longitudinal direction reversed in RL.
            # Correct only this adapter; the shared vendor-control mapper keeps
            # its original convention, as do lateral/yaw/arm commands.
            command = replace(
                command,
                dog=replace(command.dog, forward=-command.dog.forward),
                arm=replace(
                    command.arm,
                    gripper=(
                        command.arm.gripper
                        if gripper_open is None
                        else (0.072 if gripper_open else 0.0)
                    ),
                ),
            )
            payload = {
                "command": asdict(command),
                "stale": stale,
                "sampled_ns": sampled_ns,
                "input": {
                    "left_x": bool(inputs.left.buttons[0]),
                    "right_b": bool(inputs.right.buttons[1]),
                    "right_grip": inputs.right.grip,
                    "right_trigger": inputs.right.trigger,
                    "right_stick_click": bool(inputs.right.axis_click),
                    "gripper_open_request": gripper_open,
                    "right_pose": inputs.right.pose,
                    "arm_reference_ready": current_pose is not None,
                },
            }
            data = (json.dumps(payload, allow_nan=False) + "\n").encode()
            if len(data) > 4096:
                raise ValueError("PICO sample exceeded PIPE_BUF")
            try:
                os.write(status.fileno(), data)
            except BlockingIOError:
                pass  # A slow reader gets new samples, never a growing queue.
            select.select([0], [], [], 0.01)
    except BrokenPipeError:
        pass
    except Exception as error:  # noqa: BLE001 - Report native wrapper errors to parent.
        try:
            os.write(
                status.fileno(),
                (json.dumps({"error": str(error)}) + "\n").encode(),
            )
        except OSError:
            pass
    finally:
        status.close()
        if monitor:
            monitor.close()


if __name__ == "__main__":
    if "--sdk-child" not in sys.argv[1:]:
        raise SystemExit("Internal PICO process; use python -m l1w_teleop.rl --pico")
    sdk_child()
