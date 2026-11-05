#!/usr/bin/env bash
# Controller-local entry. Preflight checks data integrity, not a rest-pose/speed gate.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 ]]; then
    echo 'Usage: bash scripts/controller_control.sh PROFILE.bin CPU [NX_IP]'
    exit 1
fi
TASK_PROFILE="$(realpath -- "$1")"
TASK_CPU="$2"
TASK_PEER="${3:-192.168.234.234}"
# Offline validation must succeed before changing vendor process state.
"$TASK_DIR/lowlevel_bridge" --validate-profile "$TASK_PROFILE"
sudo "$TASK_DIR/lowlevel_bridge" --preflight-control "$TASK_PROFILE"
pgrep -x spline_daemon >/dev/null || { echo 'SPLINE not running' >&2; exit 1; }
robot-launch stop motion_control
for ((TASK_ATTEMPT=0; TASK_ATTEMPT<40; TASK_ATTEMPT++)); do
    if ! pgrep -x mc_ctrl >/dev/null; then break; fi
    sleep 0.1
done
if pgrep -x mc_ctrl >/dev/null; then echo 'mc_ctrl still running; refusing control' >&2; exit 1; fi
# Supervisor stop must remain effective; do not fight a respawning controller.
for ((TASK_CHECK=0; TASK_CHECK<20; TASK_CHECK++)); do
    if pgrep -x mc_ctrl >/dev/null; then echo 'mc_ctrl respawned after supervisor stop; refusing control' >&2; exit 1; fi
    sleep 0.1
done
pgrep -x spline_daemon >/dev/null || { echo 'SPLINE not running' >&2; exit 1; }
TASK_REPORT_DIR="$TASK_DIR/reports"
mkdir -p -- "$TASK_REPORT_DIR"
TASK_REPORT="$TASK_REPORT_DIR/bridge-$(date +%Y%m%d-%H%M%S)-$$.json"
echo 'Exclusive LowLevel session: passive/stand/RL/descent stay in this bridge. Vendor motion_control stays stopped.'
echo 'Resident bridge: session shutdown/disconnect returns to waiting; Ctrl+C ends the process.'
if sudo "$TASK_DIR/lowlevel_bridge" --control --profile "$TASK_PROFILE" --cpu "$TASK_CPU" --peer "$TASK_PEER" --report "$TASK_REPORT"; then
    TASK_STATUS=0
else
    TASK_STATUS=$?
fi
echo "Bridge report: $TASK_REPORT"
echo 'Bridge exited. Vendor motion_control remains stopped; recovery is a separate manual operation.'
exit "$TASK_STATUS"
