#!/usr/bin/env python3
"""Import a sim2sim policy, with its extracted runtime contract.

Development only: the source runner and MuJoCo are used while importing, never
on the NX. This command reads the source project and writes only local assets.
"""

import argparse
import copy
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance_path(path, project):
    """Describe source provenance without publishing a workstation path."""
    path, project = Path(path), Path(project)
    if not path.is_absolute():
        return path.as_posix()
    if path.is_relative_to(project):
        return path.relative_to(project).as_posix()
    if "logs" in path.parts:
        return Path(*path.parts[path.parts.index("logs"):]).as_posix()
    return (Path("exports") / path.name).as_posix()


def import_policy(source, output_directory, policy=None, algo="ppo", stage="stage2"):
    source = Path(source).resolve()
    sys.path.insert(0, str(source / "src"))
    if algo == "supermimic":
        return import_supermimic(source, output_directory, policy, stage=stage)
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from legged_manipulation_unilab.simulation.runner import (
        SimulationRunner,
        load_config,
    )

    if algo not in ("ppo", "him_ppo"):
        raise ValueError("import supports only ppo and him_ppo")
    if policy is not None:
        policy = Path(policy).expanduser()
        if not policy.is_absolute():
            policy = source / policy
        cfg = load_config(
            "l1_w_arm_manip_loco", algo, "mujoco", policy=str(policy.resolve())
        )
    else:
        cfg = load_config("l1_w_arm_manip_loco", algo, "mujoco")
    model = Path(cfg["algorithm"]["policy"])
    metadata = json.loads(model.with_suffix(".json").read_text())
    exported_algo = metadata.get("algo", "ppo")
    if exported_algo != algo:
        raise ValueError(
            f"sim2sim export algorithm {exported_algo!r} does not match requested {algo!r}"
        )
    if digest(model) != metadata["policy_sha256"]:
        raise ValueError("sim2sim policy hash differs from its export metadata")
    if (metadata["num_obs"], metadata["num_actions"]) != (440, 22):
        raise ValueError("import requires the 440 -> 22 policy contract")
    embedded_normalization = metadata.get("embedded_normalization")
    if not isinstance(embedded_normalization, bool):
        raise ValueError(  # noqa: TRY004 - Invalid export metadata value.
            "sim2sim export must declare embedded_normalization"
        )
    if algo == "ppo" and not embedded_normalization:
        raise ValueError("PPO import requires embedded observation normalization")
    if algo == "him_ppo" and embedded_normalization:
        raise ValueError("HIM-PPO export must consume the raw observation history")
    checkpoint = Path(metadata["checkpoint"])
    if checkpoint.is_file() and digest(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("source checkpoint hash differs from export metadata")
    reference = SimulationRunner(cfg)
    try:
        robot = copy.deepcopy(cfg["robot"])
        kp, kd = robot.pop("kp"), robot.pop("kd")
        # Deployment passes operator commands through; simulator UI ranges do
        # not become additional deployment limits when a policy is imported.
        robot.pop("velocity_limits", None)
        robot["default_dof_pos"] = reference.default_dof_pos.tolist()
        robot["ctrl_ranges"] = reference.backend.actuator_ctrl_ranges.tolist()
        state = reference.backend.read_state()
        algorithm = {
            key: cfg["algorithm"][key]
            for key in (
                "num_one_step_obs",
                "num_actor_history",
                "num_actions",
                "action_scale",
                "wheel_action_scale",
                "arm_action_scale",
            )
        }
        algorithm.update(
            name=algo,
            embedded_normalization=embedded_normalization,
        )
        if (
            algorithm["num_one_step_obs"],
            algorithm["num_actor_history"],
            algorithm["num_actions"],
            float(robot["ctrl_dt"]),
            float(algorithm["arm_action_scale"]),
        ) != (88, 5, 22, 0.02, 0.0):
            raise ValueError(
                "selected sim2sim contract is incompatible with deployment"
            )
        metadata["algo"] = algo
        metadata["runtime_contract"] = {
            "robot": robot,
            "arm_kp": kp[16:],
            "arm_kd": kd[16:],
            "algorithm": algorithm,
            "warmup": {
                "ee_position": state["ee_position"].tolist(),
                "ee_quaternion": state["ee_quaternion"].tolist(),
            },
        }
        document = json.loads((ROOT / "configs/rl/deploy.json").read_text())
        baseline_path = (ROOT / document["model"]).with_suffix(".json")
        baseline = json.loads(baseline_path.read_text())
        metadata["body_reference"] = baseline["body_reference"]
        snapshot = (
            Path("src/legged_manipulation_unilab/simulation/runner.py"),
            Path("src/legged_manipulation_unilab/simulation/backends/mujoco.py"),
            Path("src/legged_manipulation_unilab/simulation/config/l1_w_arm/config.yaml"),
            Path(cfg["backend"]["scene"]).relative_to(source),
        )
        source_hashes = {str(path): digest(source / path) for path in snapshot}
        model_provenance = provenance_path(model, source)
        source_hashes[model_provenance] = digest(model)
        metadata["sim2sim_source"] = {
            "project": source.name,
            "policy": model_provenance,
            "snapshot_sha256": source_hashes,
        }
        run_config = checkpoint.parent / "run_config.json"
        if run_config.is_file():
            training_env = json.loads(run_config.read_text())["config"]["env"]
            command = training_env.get("command_config", {})
            gait_keys = (
                "gait_x_threshold",
                "gait_lateral_threshold",
                "gait_yaw_threshold",
            )
            if all(key in command for key in gait_keys):
                metadata["runtime_contract"]["robot"]["gait_activation_thresholds"] = [
                    1e30 if command[key] is None else float(command[key])
                    for key in gait_keys
                ]
            metadata["training_run"] = {
                "config_sha256": digest(run_config),
                "control_config": training_env.get("control_config"),
                "command_config": training_env.get("command_config"),
                "ik": training_env.get("ik"),
                "arm_stage": training_env.get("arm_stage"),
            }
        output_directory = Path(output_directory).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        target = output_directory / model.name
        # Refuse a silent replacement of another policy at a versioned path.
        if target.exists() and digest(target) != metadata["policy_sha256"]:
            raise ValueError(f"different model already exists at {target}")
        shutil.copyfile(model, target)
        loaded = torch.jit.load(str(target), map_location="cpu").eval()
        rng = np.random.default_rng(20261021)
        error = 0.0
        for _ in range(24):
            obs = rng.normal(0, 0.2, 440).astype(np.float32)
            expected = reference.compute_action(obs)
            with torch.inference_mode():
                actual = loaded(torch.from_numpy(obs[None])).numpy()[0]
            if actual.shape != (22,) or not np.isfinite(actual).all():
                raise ValueError("imported policy returned invalid actions")
            error = max(error, float(np.max(np.abs(expected - actual))))
        if error > 2e-6:
            raise ValueError(f"imported model differs from sim2sim: {error}")
        validation = {"samples": 24, "max_action_error": error}
        # Print local validation results without publishing experiment records.
        metadata.pop("import_validation", None)
        metadata.pop("export_max_error", None)
        metadata["checkpoint"] = provenance_path(checkpoint, source)
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "model": str(target),
                    "sha256": digest(target),
                    "validation": validation,
                },
                indent=2,
            )
        )
    finally:
        reference.close()


def import_supermimic(source, output_directory, checkpoint, *, stage="stage2"):
    """Export only the actor; collect its contract from the source MuJoCo task."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from legged_manipulation_unilab.algos.supermimic.networks import Policy
    from legged_manipulation_unilab.cli import compose_config
    from legged_manipulation_unilab.tasks.l1_w_arm.supermimic import wrap_l1w_env
    from unilab.base.config_adapter import BackendAdapter, create_env
    from unilab.training import ensure_registries

    if checkpoint is None:
        raise ValueError("SuperMimic import requires --policy CHECKPOINT_OR_RUN_DIRECTORY")
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = source / checkpoint
    if checkpoint.is_dir():
        candidates = sorted(checkpoint.glob("ckpt_*.pt"))
        checkpoint = candidates[-1] if candidates else checkpoint / "last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # args.json can be overwritten by playback. The checkpoint owns actor shape.
    args = saved["args"]
    if args["hist"] != 10 or args["goal_frames"] != 1:
        raise ValueError("SuperMimic deployment supports hist=10 and goal_frames=1")
    torch.set_num_threads(1)
    actor = Policy(
        obs_dim=82, goal_dim=10, hist=args["hist"], z_dim=args["z_dim"],
        num_joints=22, hidden=args["policy_hidden"],
    ).eval()
    actor.load_state_dict(saved["pi"], strict=True)

    class ExportActor(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, inputs):
            return self.actor(inputs[:, :820].reshape(-1, 10, 82), inputs[:, 820:830])

    # Keep source assets and cache materialization out of the training checkout.
    os.environ.setdefault("LEGGED_MANIPULATION_ASSET_CACHE", str(ROOT / "build/supermimic-assets"))
    ensure_registries()
    cfg = compose_config(
        "supermimic", "mujoco", [f"training.stage={stage}"],
        task="l1_w_arm_manip_loco",
    )
    if stage == "stage1":
        OmegaConf.update(cfg, "env.arm_stage.fixed_arm_target", True, merge=True)
        OmegaConf.update(cfg, "env.arm_stage.disable_ee_goal_trajectory", True, merge=True)
    else:
        OmegaConf.update(cfg, "env.arm_stage.fixed_arm_target", False, merge=True)
        OmegaConf.update(cfg, "env.control_config.arm_action_scale", float(cfg.training.stage2_arm_action_scale), merge=True)
    override = BackendAdapter(cfg, root_dir=source, algo_name="supermimic").build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=override)
    try:
        bridge = wrap_l1w_env(env)
        bridge.reset()
        mdp = bridge.supermimic_mdp
        command = env.command_manager.get_term("task")
        task = command._task_cfg
        control = task.control_config
        if control.arm_action_scale != 0 or control.simulate_action_latency:
            raise ValueError("SuperMimic deployment requires IK-only arm and no simulated action latency")
        if control.wheel_action_scale != 10 or task.ctrl_dt != 0.02:
            raise ValueError("SuperMimic requires wheel scale=10 and ctrl_dt=0.02")
        # Common robot geometry/settings are shared with the existing contract;
        # read every control-sensitive quantity from this source task itself.
        document = json.loads((ROOT / "configs/rl/deploy.json").read_text())
        baseline = json.loads((ROOT / document["model"]).with_suffix(".json").read_text())
        contract = copy.deepcopy(baseline["runtime_contract"])
        robot = contract["robot"]
        robot.update(
            joint_names=list(env._backend.get_actuator_joint_names()),
            default_dof_pos=mdp.default_q.tolist(),
            ctrl_ranges=np.asarray(env.scene["robot"].data.actuator_ctrl_range).tolist(),
            ctrl_dt=float(task.ctrl_dt), gait_frequency=mdp.gait_frequency,
            # The real NumPy command term compares in float64 against these
            # original values. MDP float32 thresholds change the decision at
            # keyboard yaw +/-0.2 after velocity itself is cast to float32.
            gait_activation_thresholds=[
                1e30 if task.command_config.gait_x_threshold is None
                else float(task.command_config.gait_x_threshold),
                float(task.command_config.gait_lateral_threshold),
                float(task.command_config.gait_yaw_threshold),
            ],
            feet_phase_offsets=[0.0, 0.5, 0.5, 0.0],
            ee_goal=list(task.goal_ee.init_ee_cart if stage == "stage2" else task.arm_stage.fixed_ee_goal_cart),
            ik={key: getattr(task.ik, key) for key in ("damping", "gain", "dq_clip", "use_orientation")},
        )
        goal = task.goal_ee
        robot["ee_workspace"] = {
            "length": list(goal.sphere_l_range), "phi": list(goal.sphere_phi_range),
            "theta": list(goal.sphere_theta_range),
            "collision_lower": list(goal.collision_lower_limits),
            "collision_upper": list(goal.collision_upper_limits),
            "underground_limit": goal.underground_limit,
        }
        sensors = task.sensor
        contract.update(
            arm_kp=list(control.arm_kp), arm_kd=list(control.arm_kd),
            algorithm={
                "name": "supermimic", "embedded_normalization": True,
                "num_one_step_obs": 82, "num_actor_history": 10,
                "num_goal_obs": 10, "num_actions": 22,
                "observation_layout": "supermimic_l1w_v1",
                "action_scale": control.action_scale,
                "wheel_action_scale": control.wheel_action_scale,
                "arm_action_scale": control.arm_action_scale,
            },
            warmup={
                "ee_position": env.scene.bind_sensor_data((sensors.ee_local_pos,)).read()[0].tolist(),
                "ee_quaternion": robot["ee_goal_quaternion"],
            },
        )
        output_directory = Path(output_directory).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        run = checkpoint.parent.name
        stamp = run.split("_")[0].replace("-", "")
        target = output_directory / f"policy_supermimic_{stamp}_{saved['iteration']}.pt"
        wrapper = ExportActor(actor).eval()
        with torch.inference_mode():
            exported = torch.jit.trace(wrapper, torch.zeros(1, 830))
        buffer = io.BytesIO()
        torch.jit.save(exported, buffer)
        # No workstation paths in the published TorchScript debug records.
        clean = io.BytesIO()
        with zipfile.ZipFile(buffer) as original, zipfile.ZipFile(clean, "w") as archive:
            for entry in original.infolist():
                if not entry.filename.endswith(".debug_pkl"):
                    archive.writestr(entry, original.read(entry.filename))
        payload = clean.getvalue()
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"different model already exists at {target}")
        target.write_bytes(payload)
        relative_sources = (
            "algos/supermimic/networks.py", "tasks/l1_w_arm/supermimic.py",
            "tasks/go2_arm/manager_env.py", "tasks/go2_arm/base.py",
            "tasks/l1_w_arm/manip_loco.py", "training/supermimic.py",
            "conf/supermimic/config.yaml",
            "conf/supermimic/task/l1_w_arm_manip_loco/mujoco.yaml",
            "assets/robots/l1_w_arm/l1_w_with_arm.xml",
            "assets/robots/l1_w_arm/scene_flat.xml",
        )
        hashes = {
            f"src/legged_manipulation_unilab/{name}": digest(source / "src/legged_manipulation_unilab" / name)
            for name in relative_sources
        }
        metadata = {
            "algo": "supermimic", "embedded_normalization": True,
            "num_obs": 830, "num_actions": 22,
            "policy_sha256": digest(target),
            "checkpoint": provenance_path(checkpoint, source),
            "checkpoint_sha256": digest(checkpoint),
            "iteration": saved["iteration"], "stage": stage,
            "runtime_contract": contract, "body_reference": baseline["body_reference"],
            "sim2sim_source": {"project": source.name, "snapshot_sha256": hashes},
            "training_run": {
                "learner_args": args,
                "environment_config_source": "current source defaults; full environment config not stored in checkpoint",
                "control_config": OmegaConf.to_container(cfg.env.control_config),
                "command_config": OmegaConf.to_container(cfg.env.command_config),
                "arm_stage": OmegaConf.to_container(cfg.env.arm_stage),
            },
            "input_layout": "10 oldest-to-newest frames of [q_rel(22), qd(22), gyro(3), gravity(3), projected_previous_action(22), command(10)], followed by current command(10)",
        }
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        sys.path.insert(0, str(ROOT))
        from l1w_teleop.rl.config import load_config
        from l1w_teleop.rl.policy import PolicyCore

        core = PolicyCore(load_config(policy=target))
        validation = validate_supermimic(core, actor, mdp, command)
        report = ROOT / "build/supermimic-import-validation.json"
        report.write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps({"model": str(target), "sha256": digest(target), "validation": validation}, indent=2))
    finally:
        env.close()


def validate_supermimic(core, actor, mdp, command):
    """Compare observations, history, goals, actions and projection to training."""
    import numpy as np
    import torch

    rng = np.random.default_rng(5)
    history = []
    previous = torch.zeros(1, 22)
    max_obs = max_action = max_projection = max_phase = 0.0
    for step in range(128):
        if step in (0, 64):
            core.reset()
            history.clear()
            previous.zero_()
            command.phase.fill(0)
            command.feet_phase.fill(0)
        q = (mdp.default_q + torch.from_numpy(rng.normal(0, .1, 22).astype(np.float32)))[None]
        qd = torch.from_numpy(rng.normal(0, .2, (1, 22)).astype(np.float32))
        gyro = torch.from_numpy(rng.normal(0, .1, (1, 3)).astype(np.float32))
        rotation = torch.eye(3)[None]
        angle = float(rng.uniform(-.3, .3))
        rotation[0, 1:, 1:] = torch.tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        velocities = (
            (0, 0, 0), (.6, 0, 0), (1.2, 0, 0), (0, .2, 0),
            (0, 0, .3), (0, 0, .2), (0, 0, -.2),
            (0, .08, 0), (0, -.08, 0), (0, 0, .199), (0, 0, .201),
        )
        velocity = velocities[(step // 8) % len(velocities)]
        core.velocity[:] = velocity
        command.vel_command_b[:] = core.velocity
        core.ee_target[:] = rng.uniform((.15, -.1, .15), (.4, .1, .35))
        # Source frames must use the source command's phase, rather than
        # copying deployment phase and hiding a phase-update mismatch.
        frames = torch.tensor([[
            *command.vel_command_b[0], *command.feet_phase[0], *core.ee_target,
        ]])
        state_tuple = (torch.zeros(1, 3), rotation, torch.zeros(1, 3), gyro, q, qd)
        obs = mdp.obs(state_tuple, previous, frames, frames)
        history = ([obs] * 10) if not history else history[1:] + [obs]
        expected = torch.cat((torch.stack(history, 1).flatten(1), mdp.goal(frames, 1)), 1)
        state = {
            "joint_pos": q[0].numpy(), "joint_vel": qd[0].numpy(),
            "local_ang_vel": (rotation.transpose(-1, -2) @ gyro.unsqueeze(-1)).squeeze(-1)[0].numpy(),
            "projected_gravity": -rotation[0, 2].numpy(),
        }
        actual = core.observation(state)
        max_obs = max(max_obs, float(np.max(np.abs(actual - expected[0].numpy()))))
        with torch.inference_mode():
            action = actor(torch.stack(history, 1), frames)
        deployed = core.policy(actual)
        max_action = max(max_action, float(np.max(np.abs(deployed - action[0].numpy()))))
        core.targets(deployed, state)
        previous = mdp.project_action(action)
        max_projection = max(max_projection, float(np.max(np.abs(core.last_action - previous[0].numpy()))))
        # Compare with real task/playback, whose inactive feet phases are all
        # zero. The source MDP's imagined advance_goal leaves diagonal .5s.
        command._update_command(None)
        max_phase = max(max_phase, float(np.max(np.abs(core.phase - command.feet_phase[0]))))
    if max_obs > 1e-6 or max_action > 1e-4 or max_projection > 1e-4 or max_phase > 1e-6:
        raise ValueError(f"SuperMimic source parity failed: {max_obs}, {max_action}, {max_projection}, {max_phase}")
    return {"steps": 128, "resets": 2, "max_observation_error": max_obs,
            "max_action_error": max_action, "max_projection_error": max_projection,
            "max_phase_error": max_phase, "hardware_commands_sent": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        help="Source export, or SuperMimic checkpoint/run directory; relative to --source",
    )
    parser.add_argument(
        "--algo",
        choices=("ppo", "him_ppo", "supermimic"),
        default="ppo",
        help="Select the matching sim2sim algorithm contract",
    )
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage2", help="SuperMimic task stage")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "models/l1w_rl")
    args = parser.parse_args()
    import_policy(
        args.source,
        args.output_directory,
        policy=args.policy,
        algo=args.algo,
        stage=args.stage,
    )
