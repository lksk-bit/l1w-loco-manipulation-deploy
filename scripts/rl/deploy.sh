#!/usr/bin/env bash
# One-command source/profile update. Never starts motors or stops vendor services.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_CONTROLLER="${RL_CONTROLLER:-firefly@192.168.234.1}"
TASK_NX="${RL_NX:-robot@192.168.234.234}"
TASK_TARGET="${1:-all}"
case "$TASK_TARGET" in
    controller|nx|all) ;;
    -h|--help|help)
        echo 'Usage: bash scripts/rl/deploy.sh [all|nx|controller]'
        echo 'Default: update both hosts, including the matching driver profile.'
        echo 'Builds with Python 3.10+ and NumPy: local first, then NX .venv-rl/system Python.'
        echo 'Override the build interpreter with RL_PYTHON (local) or RL_NX_PYTHON (NX).'
        exit 0 ;;
    *) echo 'Expected all, nx or controller.' >&2; exit 2 ;;
esac
TASK_LOCAL_TEMP="$(mktemp -d)"
TASK_STAGE=".cache/l1w-rl-update-$(date -u +%Y%m%dT%H%M%S)-$$"
TASK_SSH_OPTIONS=(
    -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=3
    -o ControlMaster=auto -o ControlPersist=60 -o "ControlPath=$TASK_LOCAL_TEMP/ssh-%C"
)
TASK_SSH=(ssh "${TASK_SSH_OPTIONS[@]}")
TASK_TRANSFER_TIMEOUT=60
cleanup() {
    for TASK_HOST in "$TASK_NX" "$TASK_CONTROLLER"; do
        "${TASK_SSH[@]}" -O exit "$TASK_HOST" >/dev/null 2>&1 || true
    done
    rm -rf -- "$TASK_LOCAL_TEMP"
}
trap cleanup EXIT
mkdir -p "$TASK_ROOT/build"
TASK_BUNDLE="$TASK_ROOT/build/l1w-rl-update.tar.gz"

remote_bash() {
    local TASK_HOST="$1" TASK_COMMAND
    shift
    # SSH joins its command arguments before the remote shell parses them.
    # Quote each argument so empty values and interpreter paths survive intact.
    printf -v TASK_COMMAND '%q ' bash -s -- "$@"
    "${TASK_SSH[@]}" "$TASK_HOST" "$TASK_COMMAND"
}

transfer_run() {
    local TASK_STATUS
    if timeout --foreground --kill-after=5s "${TASK_TRANSFER_TIMEOUT}s" "$@"; then
        return 0
    else
        TASK_STATUS=$?
    fi
    if [[ "$TASK_STATUS" == 124 || "$TASK_STATUS" == 137 ]]; then
        echo "Transfer timed out after ${TASK_TRANSFER_TIMEOUT}s; update stopped before installing this file. Check the robot network, then rerun." >&2
    else
        echo "Transfer failed (exit $TASK_STATUS); update stopped before installing this file." >&2
    fi
    return "$TASK_STATUS"
}
upload() {
    local TASK_SOURCE="$1" TASK_HOST="$2" TASK_PATH TASK_COMMAND TASK_DIGEST TASK_BYTES
    TASK_PATH="$TASK_STAGE/$(basename -- "$TASK_SOURCE")"
    read -r TASK_DIGEST _ < <(sha256sum -- "$TASK_SOURCE")
    TASK_BYTES="$(wc -c < "$TASK_SOURCE")"
    echo "Uploading via SSH: $TASK_HOST:$TASK_PATH ($TASK_BYTES bytes; timeout ${TASK_TRANSFER_TIMEOUT}s)"
    # Stream stdin over SSH instead of relying on SCP/SFTP. Keep partial files
    # in staging, and verify their hash before publishing the archive name.
    printf -v TASK_COMMAND 'umask 077; cat > %q' "$TASK_PATH.part"
    transfer_run "${TASK_SSH[@]}" "$TASK_HOST" "$TASK_COMMAND" < "$TASK_SOURCE"
    remote_bash "$TASK_HOST" "$TASK_PATH" "$TASK_DIGEST" <<'REMOTE'
set -euo pipefail
read -r TASK_ACTUAL _ < <(sha256sum -- "$HOME/$1.part")
if [[ "$TASK_ACTUAL" != "$2" ]]; then
    echo "Upload SHA-256 mismatch: $1; installation not started for this file." >&2
    exit 1
fi
mv -- "$HOME/$1.part" "$HOME/$1"
echo "Upload verified: $1"
REMOTE
}
download() {
    local TASK_HOST="$1" TASK_PATH="$2" TASK_TARGET="$3" TASK_COMMAND TASK_DIGEST TASK_ACTUAL
    read -r TASK_DIGEST _ < <("${TASK_SSH[@]}" "$TASK_HOST" sha256sum -- "$TASK_PATH")
    echo "Downloading via SSH: $TASK_HOST:$TASK_PATH (timeout ${TASK_TRANSFER_TIMEOUT}s)"
    printf -v TASK_COMMAND 'cat -- %q' "$TASK_PATH"
    transfer_run "${TASK_SSH[@]}" "$TASK_HOST" "$TASK_COMMAND" > "$TASK_LOCAL_TEMP/download.part"
    read -r TASK_ACTUAL _ < <(sha256sum -- "$TASK_LOCAL_TEMP/download.part")
    if [[ "$TASK_ACTUAL" != "$TASK_DIGEST" ]]; then
        echo "Download SHA-256 mismatch: $TASK_PATH; update stopped." >&2
        return 1
    fi
    mv -- "$TASK_LOCAL_TEMP/download.part" "$TASK_TARGET"
    echo "Download verified: $TASK_PATH"
}
preflight() {
    echo "Checking update target: $2 ($1)"
    remote_bash "$1" --check "$2" < "$TASK_ROOT/scripts/rl/install_update.sh"
}
stage_directory() {
    remote_bash "$1" "$TASK_STAGE" <<'REMOTE'
set -euo pipefail
mkdir -p -- "$HOME/$1"
REMOTE
}
install_remote() {
    remote_bash "$1" "$TASK_STAGE" "$2" <<'REMOTE'
set -euo pipefail
cd -- "$HOME/$1"
tar -xzf l1w-rl-update.tar.gz
bash l1w-rl-update/install.sh "$2"
REMOTE
}
remove_stage() {
    remote_bash "$1" "$TASK_STAGE" <<'REMOTE'
set -euo pipefail
case "$1" in
    .cache/l1w-rl-update-*) rm -rf -- "$HOME/$1" ;;
    *) exit 2 ;;
esac
REMOTE
}

# Refuse before uploading/installing anything if a selected runtime is active.
if [[ "$TASK_TARGET" == all || "$TASK_TARGET" == nx ]]; then preflight "$TASK_NX" nx; fi
if [[ "$TASK_TARGET" == all || "$TASK_TARGET" == controller ]]; then preflight "$TASK_CONTROLLER" controller; fi

TASK_PYTHON=""
TASK_PYTHON_PROBE='import sys; assert sys.version_info >= (3, 10); import numpy'
if [[ -n "${RL_PYTHON:-}" ]]; then
    if ! "$RL_PYTHON" -c "$TASK_PYTHON_PROBE" >/dev/null 2>&1; then
        echo 'RL_PYTHON must name Python 3.10+ with NumPy.' >&2
        exit 1
    fi
    TASK_PYTHON="$RL_PYTHON"
else
    TASK_CANDIDATES=("$TASK_ROOT/.venv-rl/bin/python" "$TASK_ROOT/.venv/bin/python")
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then TASK_CANDIDATES+=("$VIRTUAL_ENV/bin/python"); fi
    TASK_CANDIDATES+=(python3 python)
    for TASK_CANDIDATE in "${TASK_CANDIDATES[@]}"; do
        if "$TASK_CANDIDATE" -c "$TASK_PYTHON_PROBE" >/dev/null 2>&1; then
            TASK_PYTHON="$TASK_CANDIDATE"
            break
        fi
    done
fi

TASK_BUILT_ON_NX=false
if [[ -n "$TASK_PYTHON" ]]; then
    echo "Building source/profile package locally: $TASK_PYTHON"
    "$TASK_PYTHON" -B "$TASK_ROOT/scripts/rl/prepare_update.py" --output "$TASK_BUNDLE"
else
    # Source staging uses only Python's standard library. No PC dependency install.
    if [[ "$TASK_TARGET" == controller ]]; then preflight "$TASK_NX" nx; fi
    # Probe before transferring the source bundle. A first installation may not
    # have the project's venv yet; a separate NumPy interpreter can build it.
    TASK_NX_BUILD_PYTHON="$(remote_bash "$TASK_NX" "${RL_NX_PYTHON:-}" <<'REMOTE'
set -euo pipefail
TASK_PROBE='import sys; assert sys.version_info >= (3, 10); import numpy'
if [[ -n "${1:-}" ]]; then
    TASK_CANDIDATES=("$1")
else
    TASK_CANDIDATES=("$HOME/ws/l1w-loco-manipulation-deploy/.venv-rl/bin/python" python3 python)
fi
for TASK_CANDIDATE in "${TASK_CANDIDATES[@]}"; do
    case "$TASK_CANDIDATE" in '~/'*) TASK_CANDIDATE="$HOME/${TASK_CANDIDATE#\~/}" ;; esac
    if "$TASK_CANDIDATE" -c "$TASK_PROBE" >/dev/null 2>&1; then
        printf '%s\n' "$TASK_CANDIDATE"
        exit 0
    fi
done
echo 'No usable NX build Python: Python 3.10+ with NumPy is required.' >&2
echo 'Set RL_NX_PYTHON to an existing NX interpreter, or prepare a local build environment:' >&2
echo '  python3 -m venv .venv-rl' >&2
echo '  .venv-rl/bin/python -m pip install "numpy>=1.26,<3"' >&2
echo 'Then rerun bash scripts/rl/deploy.sh all. No update was installed.' >&2
exit 1
REMOTE
)"
    echo "Local NumPy unavailable; building the validated package with NX $TASK_NX_BUILD_PYTHON."
    python3 -B "$TASK_ROOT/scripts/rl/prepare_update.py" --source-only \
        --output "$TASK_LOCAL_TEMP/l1w-rl-source.tar.gz"
    stage_directory "$TASK_NX"
    upload "$TASK_LOCAL_TEMP/l1w-rl-source.tar.gz" "$TASK_NX"
    remote_bash "$TASK_NX" "$TASK_STAGE" "$TASK_NX_BUILD_PYTHON" <<'REMOTE'
set -euo pipefail
cd -- "$HOME/$1"
tar -xzf l1w-rl-source.tar.gz
TASK_PYTHON="$2"
"$TASK_PYTHON" -B l1w-rl-source/scripts/rl/prepare_update.py --output "$PWD/l1w-rl-update.tar.gz"
REMOTE
    download "$TASK_NX" "$TASK_STAGE/l1w-rl-update.tar.gz" "$TASK_BUNDLE"
    TASK_BUILT_ON_NX=true
fi

# Install the controller first; a compile/profile failure leaves NX unchanged.
if [[ "$TASK_TARGET" == all || "$TASK_TARGET" == controller ]]; then
    stage_directory "$TASK_CONTROLLER"
    upload "$TASK_BUNDLE" "$TASK_CONTROLLER"
    install_remote "$TASK_CONTROLLER" controller
fi
if [[ "$TASK_TARGET" == all || "$TASK_TARGET" == nx ]]; then
    if ! "$TASK_BUILT_ON_NX"; then
        stage_directory "$TASK_NX"
        upload "$TASK_BUNDLE" "$TASK_NX"
    fi
    install_remote "$TASK_NX" nx
fi
if [[ "$TASK_TARGET" == all || "$TASK_TARGET" == controller ]]; then remove_stage "$TASK_CONTROLLER"; fi
if [[ "$TASK_TARGET" == all || "$TASK_TARGET" == nx || "$TASK_BUILT_ON_NX" == true ]]; then remove_stage "$TASK_NX"; fi
echo "Update complete: $TASK_TARGET. Matching profile included; NX .venv-rl preserved."
echo 'No motor runtime was started and no vendor service was stopped.'
