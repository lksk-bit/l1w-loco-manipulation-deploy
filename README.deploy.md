# Deployment and operation

English | [简体中文](README.deploy.zh-CN.md) · [Project home](README.md)

NX runs policy inference, PICO input and the AIRBOT client. RK3588 runs the LowLevel bridge. Startup is passive; standing and RL require explicit operator requests.

## 1. Hosts and directories

| Host | SSH target | Project directory |
| --- | --- | --- |
| Development PC | Local | `~/Projects/l1w-loco-manipulation-deploy` |
| NX | `robot@192.168.234.234` | `~/ws/l1w-loco-manipulation-deploy` |
| RK3588 | `firefly@192.168.234.1` | `~/l1w-loco-manipulation-deploy-controller` |

Run commands from the stated directory on the stated host. Configuration and relative model paths resolve from the project root. NX runtime logs go in `reports/` when requested.

Connect the PICO XRoboToolkit app to NX at `192.168.234.234`. Chassis commands use UDP 15051; NX receives feedback on UDP 15052. AIRBOT listens locally on TCP 50051.

## 2. Update code from the development PC

For a first checkout:

```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/lksk-bit/l1w-loco-manipulation-deploy.git
cd l1w-loco-manipulation-deploy
```

Obtain the vendor LowLevel header and place it at `include/lowlevel/lowlevel.h`, following [the dependency instructions](include/lowlevel/README.md). Git excludes this header. Your local copy is included in the private device update bundle for RK3588 compilation.

Before updating, press Ctrl+C in NX RL and wait for shutdown; then exit the RK3588 bridge. Both processes are resident, so passive mode does not mean the process has exited.

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
bash scripts/rl/deploy.sh all
```

| Change | Update target |
| --- | --- |
| Model, PICO mapping, NX inference or gripper settings | `nx` |
| RK3588 bridge source | `controller` |
| Stand/rest poses, chassis gains, mapping or soft limits | `all` |

```bash
bash scripts/rl/deploy.sh nx
```

```bash
bash scripts/rl/deploy.sh controller
```

The script packages code, models and a matching profile, verifies transfers, and compiles the bridge on RK3588. It preserves the NX environment and unrelated local files. It does not install Python dependencies/SDKs or start control.

Packaging requires Python 3.10+ and NumPy, without PyTorch. The script tries the development PC first; if unavailable, it checks NX's project `.venv-rl/bin/python`, then `python3` and `python`, before uploading sources. Use `RL_PYTHON` to select a local build interpreter. `RL_NX_PYTHON` selects the remote build interpreter only when no usable local interpreter was found.

Reuse an existing local environment by replacing the example path:

```bash
RL_PYTHON="$HOME/path/to/existing-venv/bin/python" bash scripts/rl/deploy.sh all
```

If no compatible build environment exists, create one on the development PC:

```bash
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install 'numpy>=1.26,<3'
bash scripts/rl/deploy.sh all
```

For offline inference, install all [runtime requirements](requirements-rl.txt). NX's runtime environment is separate from the build interpreter; prepare it before first startup using [section 12](#12-environment-setup-and-model-import).

Override SSH targets when the robot network changes:

```bash
RL_NX=robot@192.168.234.234 \
RL_CONTROLLER=firefly@192.168.234.1 \
bash scripts/rl/deploy.sh all
```

These variables only affect deployment SSH connections. At runtime, use NX's `--host` for the RK3588 address, the bridge script's third argument for NX's address, and the actual NX address in the PICO app.

## 3. NX startup checks

```bash
ssh robot@192.168.234.234
cd ~/ws/l1w-loco-manipulation-deploy
systemctl is-active airbot-can1 xrobo-pc-service
ip -brief link show can1
.venv-rl/bin/python -c 'import numpy, torch, arm_sdk, xrobotoolkit_sdk; print("imports OK")'
```

The service names above refer to the device's configured services. Open XRoboToolkit in the headset, connect to NX, and keep the app in the foreground with controllers tracked. Keyboard mode does not require PICO; hardware RL still reads real AIRBOT feedback.

## 4. Start AIRBOT on NX

Use a separate NX terminal:

```bash
ssh robot@192.168.234.234
sudo airbot-arm --address 127.0.0.1:50051 -i can1 -t airbot_play_g2 --no-return
```

Wait for `gRPC server listening on 127.0.0.1:50051`. Reuse an existing healthy service rather than starting a second instance. `--no-return` controls the AIRBOT service's own exit behavior; RL client return-zero is requested separately.

## 5. Start the RK3588 bridge

```bash
ssh firefly@192.168.234.1
cd ~/l1w-loco-manipulation-deploy-controller
bash scripts/controller_control.sh l1-trial-profile.bin 3
```

`3` is the CPU index used by the bridge motor thread. Append the actual NX IP if it changes:

```bash
bash scripts/controller_control.sh l1-trial-profile.bin 3 192.168.234.234
```

The script validates the profile, stops vendor `motion_control`, and takes exclusive LowLevel ownership. Keep the terminal running. Bridge exit does not automatically restore vendor motion control.

## 6. Start RL on NX

### PICO chassis, arm and gripper control

```bash
ssh robot@192.168.234.234
cd ~/ws/l1w-loco-manipulation-deploy
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode hardware --pico --arm-control
```

Select the bundled model explicitly and record the session:

```bash
cd ~/ws/l1w-loco-manipulation-deploy
mkdir -p reports
TASK_RUN="pico-$(date +%Y%m%d-%H%M%S)-$$"
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode hardware --pico --arm-control \
  --policy models/l1w_rl/policy_ppo_1028_19700.pt \
  --log "reports/$TASK_RUN.jsonl" --report "reports/$TASK_RUN.json"
```

`--log` records cycle data; `--report` saves the shutdown summary. Create the directory first and use a unique name for each session: existing log files are rejected. Startup prints the selected model and SHA-256.

### Keyboard chassis control with arm feedback only

```bash
cd ~/ws/l1w-loco-manipulation-deploy
mkdir -p reports
TASK_RUN="keyboard-$(date +%Y%m%d-%H%M%S)-$$"
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode hardware --trial --keyboard \
  --log "reports/$TASK_RUN.jsonl" --report "reports/$TASK_RUN.json"
```

`--trial` disables arm commands and cannot be combined with `--arm-control`. `--pico` and `--keyboard` are mutually exclusive. Add `--host ACTUAL_RK3588_IP` if needed.

## 7. PICO controls

Press left X to stand before using other controls. The configured stand trajectory takes 2 seconds: 1 second of abad alignment and 1 second of rise.

| Control | Behavior |
| --- | --- |
| Left X | Request stand; release and press again after hold/fault recovery |
| Left Trigger, front index-finger trigger | Hold for RL; release to stand and zero wheel-speed targets |
| Left stick forward/backward and left/right | RL chassis forward and lateral velocity |
| Right stick left/right | RL yaw velocity |
| Right Grip, side middle-finger grip | Hold and move the right controller to translate the arm; release to retain its target |
| Right Trigger, front index-finger trigger | Hold to open the gripper; release to retain measured opening |
| Right stick click, press down on the stick | Hold to close the gripper; release to retain measured opening |
| Right B | Request descent to passive; keep the process resident |

Stand and RL both support arm/gripper control. Passive/hold do not send arm/gripper motion targets. Gripper buttons do not require right Grip; close takes precedence if both open and close are held. Right Grip does not pause locomotion. Right stick click closes the gripper; it does not request arm zero.

## 8. Keyboard controls

| Key | Behavior |
| --- | --- |
| `1` | Descend to passive without exiting |
| `2` | Request stand |
| `3` | Request RL after standing completes |
| `W / S` | Increase/decrease forward velocity |
| `A / D` | Increase/decrease lateral velocity |
| `Q / E` | Increase/decrease yaw velocity |
| Space | Zero all velocity commands |
| Esc / Ctrl+C | Request descent and exit NX RL |

Velocity keys change the command by 0.2 per press. Releasing a key does not reset velocity; use Space. State changes clear previous velocity commands. In recovery waiting, `3` requests stand only; request RL again afterward.

## 9. Models and configuration

Daily settings live in [configs/rl/deploy.json](configs/rl/deploy.json).

| Setting | Purpose |
| --- | --- |
| `model` | Default policy path |
| `rl_gains` | RL leg Kp/Kd and wheel Kd; optional `by_algorithm` overrides |
| `stand` | Stand pose, gains and two-stage durations |
| `rest` | Rest pose and passive damping |
| `mapping` | Joint signs, offsets and IMU mounting rotation |
| `arm.gripper` | Opening limits and open/close speed |
| `arm.return_zero_on_exit` | Request AIRBOT SDK zero on client shutdown |

Only the following model is bundled and selected by default:

```text
models/l1w_rl/policy_ppo_1028_19700.pt
models/l1w_rl/policy_ppo_1028_19700.json
```

This export comes from the October 28 run's `model_19700.pt`. Training leg Kp/Kd was 50/1; hardware configuration remains 40/1. The model metadata activates gait when `|vx| > 0.5`, `|vy| > 0.08` or `|yaw| > 0.12`.

`--policy` overrides the default for that launch. Compatible PPO / HIM-PPO exports remain supported through a 440-input, 22-action TorchScript interface and same-stem JSON contract. Additional models must be supplied separately. Relative model paths resolve from the project root; restart NX RL to change models.

Gripper speed is `arm.gripper.speed_m_s`, currently 0.05 m/s in both directions. Opening ranges from 0 to 0.072 m. Application-level current/contact detection does not stop gripper motion; vendor driver and firmware protections remain external.

## 10. Shutdown and restart

1. Press Ctrl+C in NX RL and wait for body descent and cleanup.
2. With `--arm-control` and return-zero enabled, the client requests SDK zero after confirming body rest. `AIRBOT exit: zero_reached` indicates arrival; failures/skips print their reason. A second Ctrl+C cancels remaining waits.
3. To stop the full stack, exit the RK3588 bridge, then stop the AIRBOT service.

RK3588 can stay resident while NX RL restarts. Stop AIRBOT after the NX client has exited. Vendor `motion_control` requires separate manual recovery after the LowLevel bridge has ended.

## 11. Logs and troubleshooting

NX session logs, when enabled:

```bash
cd ~/ws/l1w-loco-manipulation-deploy
ls -lt reports/ | head -20
```

RK3588 bridge reports:

```bash
ls -lt ~/l1w-loco-manipulation-deploy-controller/reports/ | head -20
```

NX service and device diagnostics:

```bash
systemctl status xrobo-pc-service airbot-can1 --no-pager
ss -lntp
ip -details -statistics link show can1
journalctl -k -n 100 --no-pager
```

| Symptom | Check |
| --- | --- |
| `waiting_bridge`, cannot stand | Bridge process, both IPs, UDP ports and profile compatibility |
| `arm=unavailable` | AIRBOT service, `can1`, SDK feedback and USB kernel messages |
| `arm_command=waiting_grip` | Hold the side Grip, not the front Trigger |
| PICO stale input enters hold | Foreground app, headset sleep, controller tracking; release and press left X after recovery |
| Gripper does not close | Hold right stick click; inspect `right_stick_click=True` and `state=closing` |
| RK3588 hard fault | Latest report's `fault_reason` and `first_fault` |
| Update refused | Exit the resident control process on each host |

Policy/commands run at 50 Hz; state feedback targets 500 Hz; the bridge motor loop runs at 1000 Hz. Cycle and ACK timeout exits are disabled. Stale PICO/arm feedback and command-stream interruptions follow the existing hold behavior and require a new stand request after recovery. Implementation: [NX controller](l1w_teleop/rl/controller.py), [body state machine](l1w_teleop/rl/body.py), [RK3588 bridge](src/lowlevel_bridge.cpp).

## 12. Environment setup and model import

### NX runtime environment

Code deployment does not create `.venv-rl` or install its dependencies. Prepare it before first startup. Alternatively, run the new project's CLI using the full Python path of an existing compatible environment; code, configuration and models still come from the new checkout.

If compatible dependencies are installed in NX's system/user Python packages:

```bash
cd ~/ws/l1w-loco-manipulation-deploy
python3.10 -m venv --system-site-packages .venv-rl
.venv-rl/bin/python -c 'import numpy, torch, arm_sdk, xrobotoolkit_sdk; print("imports OK")'
```

For a new Python 3.10 / Linux aarch64 CPU environment, prepare the offline bundle on an internet-connected development PC:

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
python3 scripts/rl/prepare_nx_offline.py --cpu
scp build/l1w-nx-torch28-cpu.tar.gz robot@192.168.234.234:~/
```

Install on NX:

```bash
cd ~
tar -xzf l1w-nx-torch28-cpu.tar.gz
python3.10 ~/l1w-nx-torch28-cpu/install.py \
  --project ~/ws/l1w-loco-manipulation-deploy
```

The installer supplies Python dependencies, not AIRBOT / XRoboToolkit SDKs, CAN setup or device services. Check all imports before starting hardware control. Daily code updates preserve the environment.

### Offline checks and manual packaging

On the development PC, with the vendor header supplied:

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
mkdir -p reports
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode benchmark --steps 100 --report reports/benchmark.json
.venv-rl/bin/python scripts/rl/prepare_update.py
```

Copy `build/l1w-rl-update.tar.gz` to each target and extract it. Enter the extracted `l1w-rl-update` directory, then run the appropriate command:

```bash
bash install.sh nx ~/ws/l1w-loco-manipulation-deploy
```

```bash
bash install.sh controller ~/l1w-loco-manipulation-deploy-controller
```

The bundle's `install.sh` comes from [scripts/rl/install_update.sh](scripts/rl/install_update.sh). Device bundles contain your locally supplied vendor header and are excluded from public Git publication.

### Optional sim2sim import

This development step needs the source sim2sim project and its dependencies. Runtime deployment does not read that project. Replace both the source root and exported-policy path with actual locations:

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
SIM2SIM_ROOT="/path/to/sim2sim"
python3 scripts/rl/import_sim_policy.py \
  --source "$SIM2SIM_ROOT" --policy exports/policy_ppo_1028_19700.pt
```

For a compatible HIM-PPO export, add `--algo him_ppo`. The importer consumes an exported policy and matching JSON, checks hashes and the `440 → 22` contract, and writes them to `models/l1w_rl/` with portable provenance paths. Export raw training checkpoints in the source project first. Update NX and select the imported model with `--policy`.
