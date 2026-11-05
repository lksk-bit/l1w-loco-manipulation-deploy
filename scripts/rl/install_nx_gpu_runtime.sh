#!/usr/bin/env bash
# Install only the three pinned NVIDIA cuSPARSELt CUDA 12 packages, offline.
# Python packages remain in the project venv. No robot services are accessed.
set -euo pipefail
TASK_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 - "$TASK_BUNDLE" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path

assert platform.system() == "Linux" and platform.machine() == "aarch64", "NX only"
bundle = Path(sys.argv[1])
manifest = json.loads((bundle / "manifest.json").read_text())
assert manifest["cuda"] == "12.6"
for name in manifest["system_debs"]:
    path = (bundle / name).resolve(strict=True)
    assert path.is_relative_to(bundle)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][name]["sha256"], name
print("Verified NVIDIA cuSPARSELt 0.8.1.1 packages for CUDA 12.", flush=True)
PY
sudo dpkg -i \
    "$TASK_BUNDLE/nvidia/libcusparselt0-cuda-12_0.8.1.1-1_arm64.deb" \
    "$TASK_BUNDLE/nvidia/libcusparselt0-dev-cuda-12_0.8.1.1-1_arm64.deb" \
    "$TASK_BUNDLE/nvidia/cusparselt-cuda-12_0.8.1.1-1_arm64.deb"
