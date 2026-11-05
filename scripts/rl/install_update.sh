#!/usr/bin/env bash
# Run from an extracted update bundle. Updates files/builds only; no motor startup.
set -euo pipefail
TASK_CHECK_ONLY=false
if [[ "${1:-}" == --check ]]; then TASK_CHECK_ONLY=true; shift; fi
TASK_KIND="${1:-}"
case "$TASK_KIND" in
    nx) TASK_TARGET="${2:-$HOME/ws/l1w-loco-manipulation-deploy}" ;;
    controller) TASK_TARGET="${2:-$HOME/l1w-loco-manipulation-deploy-controller}" ;;
    *) echo 'Usage: bash install.sh [--check] nx|controller [TARGET_DIRECTORY]' >&2; exit 1 ;;
esac
TASK_TARGET="$(realpath -m -- "$TASK_TARGET")"
if [[ "$TASK_KIND" == controller ]]; then
    if pgrep -x lowlevel_bridge >/dev/null; then
        echo 'lowlevel_bridge is running; end that session before updating its files.' >&2
        exit 1
    fi
    command -v g++ >/dev/null || { echo 'g++ is required on the controller.' >&2; exit 1; }
else
    python3 - <<'PY'
import sys
from pathlib import Path
for process in Path('/proc').glob('[0-9]*'):
    try:
        arguments = (process / 'cmdline').read_bytes().split(b'\0')
    except OSError:
        continue
    if any(a == b'-m' and b == b'l1w_teleop.rl' for a, b in zip(arguments, arguments[1:])):
        print(f'NX RL is running (PID {process.name}); exit that session before updating.', file=sys.stderr)
        raise SystemExit(1)
PY
fi
if "$TASK_CHECK_ONLY"; then
    echo "Update preflight passed: $TASK_KIND"
    exit 0
fi
TASK_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_PACKAGE"
sha256sum --check --quiet SHA256SUMS
if [[ "$TASK_KIND" == controller ]]; then
    mkdir -p "$TASK_TARGET"
    TASK_BINARY="$(mktemp "$TASK_TARGET/.lowlevel_bridge.update.XXXXXX")"
    trap 'rm -f -- "$TASK_BINARY"' EXIT
    g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
        -I "$TASK_PACKAGE/controller/include" \
        "$TASK_PACKAGE/controller/src/lowlevel_bridge.cpp" \
        -pthread -lrt -o "$TASK_BINARY"
    chmod 755 "$TASK_BINARY"
    "$TASK_BINARY" --validate-profile "$TASK_PACKAGE/controller/l1-trial-profile.bin"
    cp -a "$TASK_PACKAGE/controller/." "$TASK_TARGET/"
    mv -f -- "$TASK_BINARY" "$TASK_TARGET/lowlevel_bridge"
    echo "Controller files/profile updated: $TASK_TARGET"
else
    cp -a "$TASK_PACKAGE/nx/." "$TASK_TARGET/"
    # Retire only the modules merged into the flat deployment package.
    # Keep any unrelated local files and the existing Python environment.
    python3 - "$TASK_TARGET" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "l1w_teleop"
obsolete = (
    "rl/arm_process.py", "rl/arm_sdk.py", "rl/gripper.py", "rl/input.py",
    "rl/lifecycle.py", "rl/posture.py", "rl/protocol.py", "rl/transport.py",
    "arm/__init__.py", "arm/airbot.py", "arm/interface.py", "arm/mock.py",
    "control/__init__.py", "control/mapping.py", "control/models.py", "control/safety.py",
    "input/__init__.py", "input/models.py", "input/monitor.py",
)
for relative in obsolete:
    source = root / relative
    source.unlink(missing_ok=True)
    for cached in (source.parent / "__pycache__").glob(source.stem + ".*.py[co]"):
        cached.unlink()
for name in ("arm", "control", "input"):
    for directory in (root / name / "__pycache__", root / name):
        try:
            directory.rmdir()
        except OSError:
            pass  # Absent or containing unrelated local files.
PY
    # Remove only known superseded files, after the unified settings were copied.
    if [[ -f "$TASK_TARGET/configs/rl/deploy.json" ]]; then
        for TASK_OBSOLETE in policy.json hardware.l1_trial.json body.l1_trial.json \
            official_body.json hardware.template.json offline_state.json \
            imu_l1w_recording.json l1_trial_recording.json l1-trial-profile.bin; do
            rm -f -- "$TASK_TARGET/configs/rl/$TASK_OBSOLETE"
        done
    fi
    echo "NX RL files and deploy.json updated: $TASK_TARGET"
fi
echo 'Update complete. No motor program was started, no vendor process was stopped, and no Python dependency was installed.'
