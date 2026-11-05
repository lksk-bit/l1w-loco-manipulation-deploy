"""NX standalone RL. Default benchmark is offline; motor writes require --mode hardware."""

import argparse
import json
import signal
import time
import zlib
from pathlib import Path

import numpy as np

from . import bridge as wire
from .config import (
    DEFAULT_CONFIG,
    ROOT,
    arm_settings,
    load_config,
    mock_hardware_settings,
    model_metadata,
)
from .policy import PolicyCore
from .body import BodyProfile, FSMController
from .mapping import ArmWorkspace, HardwareMapping
from .controller import Controller
from .diagnostics import Recorder, deployment_report
from .bridge import BridgeLink
from .teleop import Keyboard


def synthetic_state(cfg):
    sample = cfg.get("warmup")
    if sample is None:
        sample = model_metadata({"model": cfg["algorithm"]["policy"]})[
            "runtime_contract"
        ]["warmup"]
    q = np.asarray(cfg["robot"]["default_dof_pos"], dtype=float)
    return {
        "joint_pos": q,
        "joint_vel": np.zeros(22),
        "local_ang_vel": np.zeros(3),
        "projected_gravity": np.array([0, 0, -1]),
        "ee_position": np.asarray(sample["ee_position"]),
        "ee_quaternion": np.asarray(sample["ee_quaternion"]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["benchmark", "observe", "mock", "hardware"],
        default="benchmark",
    )
    parser.add_argument("--host", default="192.168.234.1")
    parser.add_argument("--port", type=int, default=15051)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument(
        "--steps",
        type=int,
        help="Controlled modes: 0 runs until exit; offline defaults to 500",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Single deployment settings file (default: configs/rl/deploy.json)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="Override config model with an exported .pt/.onnx and its same-name JSON; relative to project root",
    )
    # Expanded inputs remain accepted for offline regression fixtures/older callers.
    parser.add_argument("--policy-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--profile", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Use the candidate L1 offset profile for an unverified body-only hardware trial",
    )
    parser.add_argument(
        "--body-config",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--state", choices=["passive", "stand", "locomotion"])
    parser.add_argument("--velocity", nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument(
        "--pico",
        action="store_true",
        help="Use existing PICO input in an isolated SDK process",
    )
    parser.add_argument(
        "--arm-control",
        action="store_true",
        help="Enable separate AIRBOT Cartesian command worker using SDK IK",
    )
    parser.add_argument("--log", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--export-bridge-profile",
        type=Path,
        help="Validate calibrated profile and export, no control",
    )
    args = parser.parse_args(argv)
    if args.profile is None:
        args.profile = (
            args.config
            if args.trial
            or args.mode in ("hardware", "observe")
            or args.export_bridge_profile
            else None
        )
    if args.steps is None:
        args.steps = 0 if args.mode in ("mock", "hardware") else 500
    if (
        args.steps < 0
        or (args.steps == 0 and args.mode in ("benchmark", "observe"))
        or args.torch_threads < 1
        or not np.isfinite(args.velocity).all()
    ):
        parser.error("invalid steps/threads/velocity")
    if args.arm_control and args.mode != "hardware":
        parser.error("arm-control is only available in hardware mode")
    if args.trial and args.arm_control:
        parser.error(
            "candidate trial supports body control and read-only AIRBOT feedback; no arm-control"
        )
    if args.pico and args.keyboard:
        parser.error("choose one input: keyboard or pico")
    return args


def create_controller(args):
    control = Controller(args)
    control.cfg = load_config(
        control.args.policy_config or control.args.config, policy=control.args.policy
    )
    control.arm_config = arm_settings(control.cfg)
    control.arm_workspace = ArmWorkspace(control.arm_config["workspace_margin_m"])
    if control.cfg["algorithm"]["arm_action_scale"] != 0:
        raise ValueError("AIRBOT Cartesian IK runtime requires arm_action_scale=0")
    control.mapping = HardwareMapping(
        control.args.profile
        if control.args.profile is not None
        else mock_hardware_settings({"model": control.cfg["algorithm"]["policy"]})
    )
    control.body = None
    if control.args.mode == "hardware" or control.args.export_bridge_profile:
        if control.args.body_config is None:
            control.args.body_config = Path(
                control.mapping.config.get("body_config", control.args.config)
            )
            if not control.args.body_config.is_absolute():
                control.args.body_config = ROOT / control.args.body_config
        control.body = BodyProfile(control.args.body_config)
        control.body.apply(control.mapping)
        control.mapping.require_calibration(
            allow_trial=control.args.trial
        )  # Before any hardware client/socket.
        control.body.validate(control.mapping)
    if control.args.export_bridge_profile:
        with control.args.export_bridge_profile.open("xb") as file:
            file.write(control.mapping.profile_bytes())
        return None
    if control.args.mode == "hardware":
        if control.args.arm_control:
            control.mapping.require_arm_control()
        robot = control.cfg["robot"]
        default = np.asarray(robot["default_dof_pos"])
        # Policy gains/home keep their training meaning; hardware stand has its
        # own raw profile and is checked separately above.
        control.mapping.validate_targets(
            control.mapping.write(default, np.zeros(22), robot["kp"], robot["kd"])
        )
    control.profile_crc = (
        zlib.crc32(control.mapping.profile_bytes())
        if control.args.mode == "hardware"
        else 0
    )
    import torch

    torch.set_num_threads(control.args.torch_threads)
    torch.set_num_interop_threads(1)
    control.trial_him_arm_zero_reference = (
        control.args.trial and control.cfg["algorithm"]["name"] == "him_ppo"
    )
    control.core = (
        PolicyCore(control.cfg, arm_observation_reference=np.zeros(6))
        if control.trial_him_arm_zero_reference
        else PolicyCore(control.cfg)
    )
    control.base_state = synthetic_state(control.cfg)
    for _ in range(10):
        control.core.policy(control.core.observation(control.base_state))
    control.core.reset()
    control.core.velocity[:] = control.args.velocity
    control.controlled = control.args.mode in ("mock", "hardware")
    if control.controlled:
        passive_kd = control.core.kd.copy()
        passive_kd[:16] = (
            control.mapping.vector("passive_kd")[control.mapping.indices]
            * control.mapping.ratio**2
            if control.args.mode == "hardware"
            else 0.3
        )
        control.fsm = FSMController(
            control.cfg,
            control.core,
            control.args.state,
            passive_kd,
            stand=control.body.stand_parameters(control.mapping, control.core.default)
            if control.body
            else None,
        )
    return control


def open_resources(control, on_stop):
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        control.old_handlers[sig] = signal.signal(sig, on_stop)
    control.recorder = Recorder(
        control.args.log, status=control.args.keyboard or control.args.pico
    )
    control.terminal = Keyboard(control.args.keyboard)
    print(
        f"Policy: {control.cfg['algorithm']['name']} {control.cfg['algorithm']['policy']} "
        f"sha256={control.cfg['algorithm']['sha256']}",
        flush=True,
    )
    print(
        f"RL leg gains: Kp={control.core.kp[0]:g}, Kd={control.core.kd[0]:g}",
        flush=True,
    )
    if control.controlled:
        if control.args.steps == 0:
            print(
                "Resident control: faults stop the session; after recovery press "
                "2 / PICO left X to stand again. Ctrl+C exits this process.",
                flush=True,
            )
        print(
            "Timing: 50 Hz target; cycle and ACK timeouts disabled; "
            f"stale command/feedback >{wire.CONTROL_TIMEOUT_MS} ms pauses without descent; "
            "20 ms overruns are diagnostic only",
            flush=True,
        )
        print(
            (
                f"Stand gate: leg error < {control.fsm.position_tolerance:g} rad, "
                f"speed < {control.fsm.velocity_tolerance:g} rad/s "
                f"for {control.fsm.stable_duration:g} s"
            )
            if control.fsm.require_settling
            else "Stand gate: disabled; RL may start after the stand trajectory completes",
            flush=True,
        )
    if control.args.keyboard:
        print(
            "1 passive/descent | 2 stand | 3 RL | W/S A/D Q/E velocity | SPACE stop | Esc/Ctrl+C exit",
            flush=True,
        )
    if control.args.mode == "hardware":
        print(
            f"IMU source quaternion order: {control.mapping.imu_quaternion_order}; model uses wxyz",
            flush=True,
        )
    if control.body:
        print(
            f"Hardware body: {control.args.body_config} sha256={control.body.sha256}; "
            f"two-stage cubic stand: abad {control.body.hip_seconds:g}s + "
            f"rise {control.body.seconds - control.body.hip_seconds:g}s; local lie/descent then passive",
            flush=True,
        )
    if control.args.trial:
        print(
            f"Candidate trial: {control.args.profile}; physical verification flags remain false. "
            f"Assumptions: {control.mapping.config.get('trial_assumptions')}",
            flush=True,
        )
    if control.trial_him_arm_zero_reference:
        print(
            "HIM trial observation: arm joint reference = AIRBOT SDK zero "
            "[0, 0, 0, 0, 0, 0] rad; measured joints and motor targets unchanged",
            flush=True,
        )
    if control.args.pico:
        from .pico_sdk import PicoWorker

        control.pico = PicoWorker()
    if control.args.mode != "benchmark":
        control.link = BridgeLink(
            control.args.host, control.args.port, control.args.bind
        )
        if control.args.mode != "mock":
            from .arm import ArmWorker

            control.arm = ArmWorker(
                control.args.arm_control,
                workspace_margin_m=control.arm_workspace.margin,
                gripper_config=control.arm_config["gripper"]
                if control.args.arm_control and control.args.pico
                else None,
            )
        timeout = time.monotonic() + 5
        while not control.stop and time.monotonic() < timeout:
            if control.link.snapshot()[0] is not None:
                break
            time.sleep(0.01)
        else:
            if not control.stop and not (
                control.controlled and control.args.steps == 0
            ):
                raise RuntimeError("bridge startup timed out")


def close_resources(control):
    if (
        control.link is not None
        and control.last_command_boot is not None
        and control.last_frame is not None
    ):
        # A bounded request window. Bridge watchdog performs the same local shutdown if requests are lost.
        for _ in range(3):
            frame = control.link.snapshot()[0] or control.last_frame
            # A new bridge generation belongs to a future NX session;
            # never send it this process's retired client nonce.
            if frame.boot != control.last_command_boot:
                break
            control.sequence += 1
            zeros = np.zeros((16, 5), dtype=np.float32)
            try:
                control.link.send(
                    wire.pack_command(
                        control.session,
                        control.sequence,
                        time.monotonic_ns(),
                        frame,
                        zeros,
                        mode=wire.SHUTDOWN,
                    )
                )
            except OSError:
                break
            time.sleep(0.005)
    if control.terminal:
        control.terminal.close()
    if control.arm:
        if (
            control.args.arm_control
            and control.return_zero_on_stop
            and control.arm_config["return_zero_on_exit"]
        ):
            try:
                print(
                    "Exit: waiting for body rest before AIRBOT SDK return-zero...",
                    flush=True,
                )
                body_at_rest = control.link is not None and wait_for_body_rest(
                    control.link,
                    control.mapping,
                    cancelled=lambda: control.stop_requests > 1,
                )
                if body_at_rest and control.stop_requests <= 1:
                    print("Exit: requesting AIRBOT SDK return-zero...", flush=True)
                    reached = control.arm.return_zero(
                        cancelled=lambda: control.stop_requests > 1
                    )
                    control.arm_exit_result = (
                        "zero_reached"
                        if reached
                        else (control.arm.return_zero_error or "return_zero_failed")
                    )
                else:
                    control.arm_exit_result = (
                        "return_zero_skipped_body_rest_unconfirmed_or_cancelled"
                    )
                print(f"AIRBOT exit: {control.arm_exit_result}", flush=True)
            except (ValueError, RuntimeError, OSError) as error:
                control.arm_exit_result = f"return_zero_failed: {error}"
                print(f"AIRBOT exit: {control.arm_exit_result}", flush=True)
        control.arm.close()
    if control.pico:
        control.pico.close()
    if control.link:
        control.link.close()
    if control.recorder:
        control.recorder.close()
    for sig, handler in control.old_handlers.items():
        signal.signal(sig, handler)


def wait_for_body_rest(link, mapping, cancelled=lambda: False, timeout=10):
    deadline = time.monotonic() + timeout
    legs = np.arange(16) % 4 != 3
    rest = np.asarray(mapping.config["shutdown_raw_pose"])
    while time.monotonic() < deadline and not cancelled():
        frame, received, _ = link.snapshot()
        if frame is not None and (
            0 <= time.monotonic_ns() - received <= wire.CONTROL_TIMEOUT_NS
            and frame.flags & (wire.MOTORS_VALID | wire.IMU_VALID)
            == wire.MOTORS_VALID | wire.IMU_VALID
            and not frame.flags & (wire.FAULT | wire.DESCENDING)
            and np.all(np.abs(np.asarray(frame.q)[legs] - rest[legs]) <= 0.08)
            and np.all(np.abs(np.asarray(frame.dq)[legs]) <= 0.2)
        ):
            return True
        time.sleep(0.02)
    return False


def main(argv=None):
    control = create_controller(parse_args(argv))
    if control is None:
        return 0

    def on_stop(_signum, _frame):
        control.stop = True
        control.stop_requests += 1
        if _signum == signal.SIGINT:
            control.return_zero_on_stop = True

    try:
        open_resources(control, on_stop)
        return control.run()
    except (ValueError, RuntimeError, OSError) as error:
        if control.fsm:
            control.fsm.fail(error)
        raise
    finally:
        close_resources(control)
        report = deployment_report(control)
        print(json.dumps(report, indent=2))
        if control.args.report:
            control.args.report.write_text(json.dumps(report, indent=2) + "\n")
