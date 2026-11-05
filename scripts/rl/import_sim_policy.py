#!/usr/bin/env python3
"""Import a sim2sim policy, with its extracted runtime contract.

Development only: the source runner and MuJoCo are used while importing, never
on the NX. This command reads the source project and writes only local assets.
"""

import argparse
import copy
import hashlib
import json
import shutil
import sys
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


def import_policy(source, output_directory, policy=None, algo="ppo"):
    source = Path(source).resolve()
    sys.path.insert(0, str(source / "src"))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        help="Import this source export instead of sim2sim's default; relative to --source",
    )
    parser.add_argument(
        "--algo",
        choices=("ppo", "him_ppo"),
        default="ppo",
        help="Select the matching sim2sim algorithm contract",
    )
    parser.add_argument("--output-directory", type=Path, default=ROOT / "models/l1w_rl")
    args = parser.parse_args()
    import_policy(
        args.source,
        args.output_directory,
        policy=args.policy,
        algo=args.algo,
    )
