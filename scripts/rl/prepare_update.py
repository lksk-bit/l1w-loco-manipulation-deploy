#!/usr/bin/env python3
"""Build a self-contained source update with matching NX/3588 trial profiles."""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def package_sources():
    """Repository source and its destination in the installable package."""
    development_only = {
        Path("models/l1w_rl/kinematics.xml"),
        Path("models/l1w_rl/source_snapshot.json"),
    }
    # Ship the flat deployment runtime together with its configuration and models.
    yield Path("l1w_teleop/__init__.py"), Path("nx/l1w_teleop/__init__.py")
    for directory_name in ("l1w_teleop/rl", "configs/rl", "models/l1w_rl"):
        for path in sorted((ROOT / directory_name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                relative = path.relative_to(ROOT)
                if relative not in development_only:
                    yield relative, Path("nx") / relative
    # Preserve project-relative model selection on NX, including models stored
    # outside the usual model directory. Absolute host paths are not portable.
    document = json.loads((ROOT / "configs/rl/deploy.json").read_text())
    model = Path(document.get("model", "models/l1w_rl/policy_ppo_1028_19700.pt"))
    if model.is_absolute() or not (ROOT / model).resolve().is_relative_to(ROOT):
        raise ValueError("packaged model must use a project-relative path")
    model = (ROOT / model).resolve().relative_to(ROOT)
    for relative in (model, model.with_suffix(".json")):
        if not (ROOT / relative).is_file():
            raise ValueError(f"selected model asset missing: {relative}")
        if relative.parts[:2] != ("models", "l1w_rl"):
            yield relative, Path("nx") / relative
    yield Path("requirements-rl.txt"), Path("nx/requirements-rl.txt")
    for name in (
        "README.md", "README.zh-CN.md", "README.deploy.md",
        "README.deploy.zh-CN.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
    ):
        yield Path(name), Path("nx") / name
    for name in ("lowlevel_bridge.cpp", "rl_protocol.h"):
        yield Path("src") / name, Path("controller/src") / name
    if not (ROOT / "include/lowlevel/lowlevel.h").is_file():
        raise ValueError(
            "Vendor LowLevel header missing: obtain lowlevel.h from the vendor "
            "and place it at include/lowlevel/lowlevel.h; see include/lowlevel/README.md"
        )
    yield (
        Path("include/lowlevel/lowlevel.h"),
        Path("controller/include/lowlevel/lowlevel.h"),
    )
    yield (
        Path("scripts/rl/controller_control.sh"),
        Path("controller/scripts/controller_control.sh"),
    )
    yield Path("scripts/rl/install_update.sh"), Path("install.sh")
    yield Path("README.deploy.md"), Path("DEPLOYMENT.md")


def build_source(output):
    """Stage sources with stdlib only; NX builds the validated package using NumPy."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {source for source, _ in package_sources()}
    sources.update(
        {Path("scripts/rl/prepare_update.py"), Path("l1w_teleop/__init__.py")}
    )
    with tarfile.open(output, "w:gz") as archive:
        for source in sorted(sources):
            archive.add(ROOT / source, arcname=str(Path("l1w-rl-source") / source))
    print(f"Source bundle: {output}")


def build(output):
    from l1w_teleop.rl.body import BodyProfile
    from l1w_teleop.rl.policy import load_config
    from l1w_teleop.rl.mapping import PROFILE_VERSION, HardwareMapping
    from l1w_teleop.rl.bridge import CONTROL_TIMEOUT_US

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping = HardwareMapping()
    body = BodyProfile(ROOT / mapping.config["body_config"])
    body.apply(mapping)
    mapping.require_calibration(allow_trial=True)
    body.validate(mapping)
    profile = mapping.profile_bytes()
    cfg = load_config()
    states = cfg["states"]
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        package = Path(directory) / "l1w-rl-update"
        for source, relative in package_sources():
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, target)
        (package / "controller/l1-trial-profile.bin").write_bytes(profile)
        release = {
            "kind": "l1-offset-candidate-source-update",
            "config": "configs/rl/deploy.json",
            "model": Path(cfg["algorithm"]["policy"]).relative_to(ROOT).as_posix(),
            "policy_algorithm": cfg["algorithm"]["name"],
            "policy_sha256": cfg["algorithm"]["sha256"],
            "mapping_sha256": mapping.sha256,
            "imu_quaternion_order": mapping.imu_quaternion_order,
            "body_sha256": body.sha256,
            "bridge_profile_crc32": zlib.crc32(profile),
            "bridge_profile_version": PROFILE_VERSION,
            "bridge_profile_bytes": len(profile),
            "cycle_timeout_ms": None,
            "ack_timeout_ms": None,
            "resident_control": True,
            "timeout_action": "hold_without_descent",
            "arm": cfg.get("arm"),
            "command_ttl_us": CONTROL_TIMEOUT_US,
            "state_reference_lag_ticks": 100,
            "stand_settling": {
                "required": states["require_stand_settling"],
                "position_tolerance_rad": states["settle_position_tolerance_rad"],
                "velocity_tolerance_rad_s": states["settle_velocity_tolerance_rad_s"],
                "stable_seconds": states["settle_seconds"],
            },
            "stand_trajectory": {
                "curve": "two_stage_cubic",
                "hip_return_seconds": body.hip_seconds,
                "rise_seconds": body.seconds - body.hip_seconds,
                "total_seconds": body.seconds,
            },
            "starts_hardware": False,
        }
        (package / "release.json").write_text(json.dumps(release, indent=2) + "\n")
        files = sorted(p for p in package.rglob("*") if p.is_file())
        (package / "SHA256SUMS").write_text(
            "".join(
                f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(package).as_posix()}\n"
                for p in files
            )
        )
        with tarfile.open(output, "w:gz") as archive:
            archive.add(package, arcname=package.name)
    print(f"Update bundle: {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"SHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}")
    print(f"Matching NX/3588 profile CRC32: {release['bridge_profile_crc32']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Stdlib-only source staging for an NX-side package build",
    )
    args = parser.parse_args()
    output = args.output or ROOT / "build" / (
        "l1w-rl-source.tar.gz" if args.source_only else "l1w-rl-update.tar.gz"
    )
    (build_source if args.source_only else build)(output)
