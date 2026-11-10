# Adapted from legged-manipulation_unilab, Apache-2.0.
# See models/l1w_rl/SOURCE.md and SOURCE_LICENSE.
"""Policy inference, observation history and action target generation."""

import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import ROOT as ROOT, load_config as load_config
from .mapping import orientation_error, quat_matrix

class Policy:
    """TorchScript or ONNX callable with input/output dimension preflight."""

    def __init__(self, path: str | Path, num_obs: int, num_actions: int, *, device="cpu"):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Exported policy does not exist: {path}")
        if device not in ("cpu", "cuda"):
            raise ValueError("Policy device must be cpu or cuda")
        self.device = device
        self.device_name = "CPU"
        self.num_obs, self.num_actions = num_obs, num_actions
        self.module: Any = None
        self.session: Any = None
        if path.suffix == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ValueError("ONNX inference requires the export extra") from exc
            providers = ["CPUExecutionProvider"]
            if device == "cuda":
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise ValueError(
                        "CUDA policy requested but ONNX Runtime has no CUDAExecutionProvider"
                    )
                providers.insert(0, "CUDAExecutionProvider")
                self.device_name = "ONNX CUDAExecutionProvider"
            try:
                self.session = ort.InferenceSession(
                    str(path), providers=providers
                )
            except Exception as exc:
                raise ValueError(
                    f"Cannot load exported ONNX policy: {path}: {exc}"
                ) from exc
            if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
                raise ValueError("ONNX Runtime failed to initialize CUDAExecutionProvider")
            # Do not silently recreate a failed CUDA session on CPU.
            if device == "cuda":
                self.session.disable_fallback()
            inputs = self.session.get_inputs()
            if len(inputs) != 1:
                raise ValueError("Expected one obs input in exported policy")
            self.input_name = inputs[0].name
        elif path.suffix == ".pt":
            import torch

            if device == "cuda":
                if not torch.cuda.is_available():
                    raise ValueError(
                        "CUDA policy requested but torch.cuda.is_available() is false; "
                        "install the Jetson CUDA-enabled PyTorch runtime or use --device cpu"
                    )
                self.device_name = torch.cuda.get_device_name(0)
            try:
                self.module = torch.jit.load(str(path), map_location=device).eval()
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot load exported TorchScript policy on {device}: {path}: {exc}"
                ) from exc
        else:
            raise ValueError(
                "Policy must be an exported .pt (TorchScript) or .onnx model"
            )
        try:
            self(np.zeros(num_obs, dtype=np.float32))
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Policy input/output does not match {num_obs} -> {num_actions}: {exc}"
            ) from exc

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.num_obs,) or not np.isfinite(obs).all():
            raise ValueError(
                f"Policy obs must be a finite vector of length {self.num_obs}"
            )
        if self.session is not None:
            try:
                result = self.session.run(None, {self.input_name: obs[None, :]})[0]
            except Exception as exc:
                raise ValueError(
                    f"ONNX obs does not match the exported model: {exc}"
                ) from exc
        else:
            import torch

            with torch.inference_mode():
                tensor = torch.from_numpy(obs[None, :])
                if self.device == "cuda":
                    tensor = tensor.to("cuda")
                # Returning to CPU waits for the complete CUDA result. Runtime
                # stage timing therefore includes copies and synchronization.
                result = self.module(tensor).cpu().numpy()
        action = np.asarray(result, dtype=np.float32)
        if action.shape != (1, self.num_actions) or not np.isfinite(action).all():
            raise ValueError(
                f"Policy must return finite actions with shape (1, {self.num_actions})"
            )
        return action[0].copy()

class ObservationHistory:
    """Oldest-to-newest frames, backfilled with the current frame after reset."""

    def __init__(self, length: int, frame_size: int):
        self.frames = np.zeros((length, frame_size), dtype=np.float32)
        self.initialized = False

    def reset(self) -> None:
        self.initialized = False
        self.frames.fill(0)

    def append(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != self.frames.shape[1:] or not np.isfinite(frame).all():
            raise ValueError(
                "Observation frame has an invalid shape or nonfinite values"
            )
        if self.initialized:
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = frame
        else:
            self.frames[:] = frame
            self.initialized = True
        return self.frames.reshape(-1).copy()

class PolicyCore:
    def __init__(self, cfg, policy=None, *, arm_observation_reference=None, device="cpu"):
        self.cfg, self.robot, self.algo = cfg, cfg["robot"], cfg["algorithm"]
        self.default = np.asarray(self.robot["default_dof_pos"], dtype=float)
        self.observation_reference = self.default.copy()
        if arm_observation_reference is not None:
            reference = np.asarray(arm_observation_reference, dtype=float)
            if reference.shape != (6,) or not np.isfinite(reference).all():
                raise ValueError("arm observation reference must be six finite joint angles")
            self.observation_reference[16:22] = reference
        self.kp = np.asarray(self.robot["kp"], dtype=float)
        self.kd = np.asarray(self.robot["kd"], dtype=float)
        # MuJoCo wheel gains stay defined by the XML; preserve those velocity-servo gains.
        self.kd[12:16] = self.robot["wheel_kv"]
        self.supermimic = self.algo.get("name") == "supermimic"
        frame_size = self.algo["num_one_step_obs"]
        history_length = self.algo["num_actor_history"]
        num_inputs = frame_size * history_length + self.algo.get("num_goal_obs", 0)
        self.policy = policy or Policy(self.algo["policy"], num_inputs, 22, device=device)
        self.history = ObservationHistory(history_length, frame_size)
        self.frame = np.zeros(frame_size, dtype=np.float32)
        self.last_action = np.zeros(22, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.phase = np.zeros(4, dtype=np.float32)
        self.gait_phase = 0.0
        # Match the sim runner for every selectable L1-W policy: vx drives the
        # wheels, while only vy/yaw can start the leg gait clock.  Go2-style
        # contracts without wheels retain their original all-axis behavior.
        has_wheels = self.robot["wheel_slice"][1] > self.robot["wheel_slice"][0]
        self.gait_command_axes = slice(1, 3) if has_wheels else slice(0, 3)
        self.gait_command_deadzone = float(
            self.robot.get(
                "gait_command_deadzone",
                0.05 if has_wheels else self.robot["command_deadzone"],
            )
        )
        self.gait_activation_thresholds = (
            np.asarray(self.robot["gait_activation_thresholds"], dtype=float)
            if "gait_activation_thresholds" in self.robot
            else None
        )
        self.ee_target = np.asarray(self.robot["ee_goal"], dtype=np.float32)
        self.ee_quat = np.asarray(self.robot["ee_goal_quaternion"], dtype=float)

    def reset(self):
        self.history.reset()
        self.last_action.fill(0)
        self.phase.fill(0)
        self.gait_phase = 0.0

    def observation(self, state):
        if self.supermimic:
            return self._supermimic_observation(state)
        f = self.frame
        f[0:3] = state["local_ang_vel"]
        f[3:6] = state["projected_gravity"]
        f[6:9] = self.velocity
        f[9:13] = self.phase
        f[13:35] = state["joint_pos"] - self.observation_reference
        f[25:29] = 0  # Ignore only wheel position; retain wheel speed and history.
        f[35:57] = state["joint_vel"]
        f[57:60] = state["ee_position"]
        f[60:63] = self.ee_target
        f[63:66] = state["ee_position"] - self.ee_target
        f[66:88] = self.last_action
        return self.history.append(f)

    def _supermimic_observation(self, state):
        # Source L1WSuperMimicMDP.obs(): joint-relative positions, velocities,
        # gyro, gravity, executed previous action, then the current command.
        # The goal encoder also receives that same current command separately.
        f = self.frame
        f[:22] = state["joint_pos"] - self.observation_reference
        f[12:16] = 0
        f[22:44] = state["joint_vel"]
        f[44:47] = state["local_ang_vel"]
        f[47:50] = state["projected_gravity"]
        f[50:72] = self.last_action
        f[72:75] = self.velocity
        f[75:79] = self.phase
        f[79:82] = self.ee_target
        return np.concatenate((self.history.append(f), f[72:82]))

    def arm_delta(self, state, jac_pos, jac_rot):
        rotation = quat_matrix(state["arm_ref_quaternion"]).T
        jac = rotation @ jac_pos
        error = self.ee_target - state["ee_position"]
        ik = self.robot["ik"]
        if ik["use_orientation"]:
            jac = np.concatenate([jac, rotation @ jac_rot])
            error = np.concatenate(
                [error, orientation_error(self.ee_quat, state["ee_quaternion"])]
            )
        delta = jac.T @ np.linalg.solve(
            jac @ jac.T + np.eye(jac.shape[0]) * ik["damping"] ** 2, error
        )
        if ik["dq_clip"] > 0:
            delta = np.clip(delta, -ik["dq_clip"], ik["dq_clip"])
        return ik["gain"] * delta

    def update_ee_goal(self, cart, quat, *, constrain_workspace=True):
        cart = np.asarray(cart, dtype=float)
        quat = np.asarray(quat, dtype=float)
        if cart.shape != (3,) or not np.isfinite(cart).all():
            raise ValueError("invalid EE position")
        if (
            quat.shape != (4,)
            or not np.isfinite(quat).all()
            or np.linalg.norm(quat) < 1e-8
        ):
            raise ValueError("invalid EE quaternion")
        if not constrain_workspace:
            # Physical PICO tracking follows the original AIRBOT workflow:
            # preserve Cartesian input and let service-side IK assess it.
            self.ee_target[:] = cart
            self.ee_quat[:] = quat / np.linalg.norm(quat)
            return
        workspace = self.robot["ee_workspace"]
        length = max(float(np.linalg.norm(cart)), 1e-6)
        phi = math.atan2(cart[2], cart[0])
        theta = math.asin(np.clip(cart[1] / length, -1, 1))
        length = float(np.clip(length, *workspace["length"]))
        phi = float(np.clip(phi, *workspace["phi"]))
        theta = float(np.clip(theta, *workspace["theta"]))
        clipped = length * np.array(
            [
                math.cos(phi) * math.cos(theta),
                math.sin(theta),
                math.sin(phi) * math.cos(theta),
            ]
        )
        inside = np.all(clipped > workspace["collision_lower"]) and np.all(
            clipped < workspace["collision_upper"]
        )
        if inside or clipped[2] < workspace["underground_limit"]:
            return
        self.ee_target[:] = clipped
        self.ee_quat[:] = quat / np.linalg.norm(quat)

    def targets(self, action, state, arm_delta=None):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (22,) or not np.isfinite(action).all():
            raise ValueError("policy must produce 22 finite actions")
        if self.supermimic:
            # Match the source bridge projection BEFORE storing action history:
            # wheel commands saturate at +/-2.8 and Stage 2 arm motion is IK-only.
            action = action.copy()
            action[12:16] = np.clip(action[12:16], -2.8, 2.8)
            action[16:22] = 0
        q, dq = self.default.copy(), np.zeros(22, dtype=float)
        q[:12] += action[:12] * self.algo["action_scale"]
        dq[12:16] = action[12:16] * self.algo["wheel_action_scale"]
        if arm_delta is None:
            if self.algo["arm_action_scale"] != 0:
                raise ValueError("AIRBOT Cartesian IK requires arm_action_scale=0")
            q[16:] = state["joint_pos"][16:]
        else:  # Offline reference comparison retains the original DLS calculation.
            q[16:] = (
                state["joint_pos"][16:]
                + arm_delta
                + action[16:] * self.algo["arm_action_scale"]
            )
        limits = np.asarray(self.robot["ctrl_ranges"])
        q[:12] = np.clip(q[:12], limits[:12, 0], limits[:12, 1])
        if arm_delta is not None:
            q[16:] = np.clip(q[16:], limits[16:, 0], limits[16:, 1])
        dq[12:16] = np.clip(dq[12:16], limits[12:16, 0], limits[12:16, 1])
        self.last_action[:] = (
            action  # Preserve training's raw action history, not motor targets.
        )
        gait_active = (
            np.any(np.abs(self.velocity) > self.gait_activation_thresholds)
            if self.gait_activation_thresholds is not None
            else np.any(
                np.abs(self.velocity[self.gait_command_axes])
                > self.gait_command_deadzone
            )
        )
        if gait_active:
            increment = self.robot["ctrl_dt"] * self.robot["gait_frequency"]
            if self.supermimic:
                # Real SuperMimic playback accumulates/wraps phase in float32.
                # Match it before feeding phase into both actor inputs.
                self.gait_phase = float(np.fmod(
                    np.float32(self.gait_phase) + np.float32(increment),
                    np.float32(1.0),
                ))
                self.phase[:] = np.fmod(
                    np.float32(self.gait_phase)
                    + np.asarray(self.robot["feet_phase_offsets"], dtype=np.float32),
                    np.float32(1.0),
                )
            else:
                self.gait_phase = (self.gait_phase + increment) % 1
                self.phase[:] = (
                    self.gait_phase + np.asarray(self.robot["feet_phase_offsets"])
                ) % 1
        else:
            self.gait_phase = 0
            self.phase.fill(0)
        return q, dq, self.kp.copy(), self.kd.copy()
