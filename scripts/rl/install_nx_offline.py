#!/usr/bin/env python3
"""Install bundle wheels into NX's project .venv-rl, without network or sudo.

GPU system packages are installed separately by install_nx_gpu_runtime.sh.
"""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

GPU_LIBRARIES = (
    "libopenblas.so.0",
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcusparse.so.12",
    "libcusparseLt.so.0",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcudnn.so.9",
    "libcupti.so.12",
    "libcufile.so.0",
)


def check_gpu_libraries():
    missing = []
    for name in GPU_LIBRARIES:
        try:
            ctypes.CDLL(name)
        except OSError as error:
            missing.append(f"{name}: {error}")
    if missing:
        raise ValueError(
            "GPU runtime dependencies are not ready:\n"
            + "\n".join(missing)
            + "\nFor cuSPARSELt, run: bash ~/l1w-nx-torch28-cu126/install_nx_gpu_runtime.sh"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.home() / "ws/l1w-loco-manipulation-deploy")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("This bundle is for Linux aarch64 NX, not the development PC")
    if sys.version_info[:2] != (3, 10):
        raise ValueError("Run this installer with NX's Python 3.10")
    project = args.project.expanduser().resolve(strict=True)
    if not (project / "l1w_teleop/rl/__main__.py").is_file():
        raise ValueError(f"RL project not found: {project}")
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["python"] != "3.10" or manifest["machine"] != "aarch64":
        raise ValueError("Unexpected bundle target")
    torch_device = manifest.get("torch_device", "cuda")
    if manifest.get("torch_included") and torch_device not in {"cpu", "cuda"}:
        raise ValueError("Unexpected torch device in bundle")
    for name, info in manifest["files"].items():
        path = (bundle / name).resolve(strict=True)
        if not path.is_relative_to(bundle):
            raise ValueError(f"Unexpected bundle path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}; copy the bundle again")
    environment = project / ".venv-rl"
    if environment.is_symlink():
        raise ValueError(".venv-rl must be a directory inside the project")
    config = environment / "pyvenv.cfg"
    if config.exists():
        for line in config.read_text().splitlines():
            if line.startswith("version = ") and not line.startswith("version = 3.10."):
                raise ValueError("Existing .venv-rl uses another Python version")
    # The CLI uses symlinks on Linux, but EnvBuilder defaults to copies. Copying
    # over bin/python3 left by the CLI follows its symlink to the source Python
    # and raises SameFileError. Match the CLI so a partial venv can be repaired.
    venv.EnvBuilder(system_site_packages=True, symlinks=True, with_pip=False).create(
        environment
    )
    python = str(environment / "bin/python")
    process_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PIP_")
        and key not in {"PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"}
    }
    process_env["PIP_CONFIG_FILE"] = os.devnull

    def run(*arguments):
        subprocess.run([python, *arguments], env=process_env, check=True)

    run(
        "-c",
        "import sys; assert sys.prefix != sys.base_prefix; print('venv:', sys.prefix)",
    )
    local_options = [
        "--no-index",
        "--find-links",
        str(bundle / "wheels"),
        "--no-user",
        "--ignore-installed",
        "--no-cache-dir",
    ]
    run(
        str(bundle / "get-pip.py"),
        "--isolated",
        *local_options,
        "pip==25.2",
        "setuptools==80.9.0",
        "wheel==0.45.1",
    )
    run(
        "-m",
        "pip",
        "--isolated",
        "install",
        *local_options,
        "numpy==1.26.4",
    )
    run(
        "-c",
        "import pathlib, sys, numpy, pip; "
        "assert numpy.__version__ == '1.26.4'; "
        "assert pathlib.Path(numpy.__file__).is_relative_to(sys.prefix); "
        "assert pathlib.Path(pip.__file__).is_relative_to(sys.prefix); "
        "print('numpy:', numpy.__version__, numpy.__file__); "
        "print('pip:', pip.__file__)",
    )
    print(f"Base environment ready: {environment}", flush=True)
    if manifest.get("torch_included"):
        if torch_device == "cuda":
            check_gpu_libraries()
        run(
            "-m",
            "pip",
            "--isolated",
            "install",
            *local_options,
            *manifest["torch_requirements"],
        )
        if torch_device == "cuda":
            run(
                "-c",
                "import pathlib, sys, torch; "
                "assert pathlib.Path(torch.__file__).is_relative_to(sys.prefix); "
                "print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); "
                "assert torch.version.cuda == '12.6', 'Expected CUDA 12.6 torch'; "
                "assert torch.cuda.is_available(), 'CUDA GPU unavailable'; "
                "print('GPU:', torch.cuda.get_device_name(0)); "
                "assert torch.ones(4, device='cuda').sum().item() == 4; "
                "print('GPU smoke check passed')",
            )
        else:
            run(
                "-c",
                "import pathlib, sys, torch; "
                "assert pathlib.Path(torch.__file__).is_relative_to(sys.prefix); "
                "assert torch.__version__ == '2.8.0+cpu'; "
                "assert torch.version.cuda is None, 'Expected CPU-only torch'; "
                "assert torch.ones(4).sum().item() == 4; "
                "print('torch:', torch.__version__, 'CPU smoke check passed')",
            )
    else:
        print("Torch is not included. Install the matching Jetson torch wheel next.")
    print(f"Environment ready. Activate: source {environment}/bin/activate")
    print(
        "Check the existing AIRBOT SDK: python -c 'import arm_sdk; print(arm_sdk.__file__)'"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Offline installation failed: {error}", file=sys.stderr)
        sys.exit(1)
