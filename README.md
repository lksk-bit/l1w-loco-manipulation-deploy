# l1w-loco-manipulation-deploy

English | [简体中文](README.zh-CN.md)

Standalone deployment for L1-W locomotion and AIRBOT manipulation, with RL policy inference, PICO teleoperation, and an NX–RK3588 control pipeline.

**50 Hz policy inference · PPO / HIM-PPO / SuperMimic · PICO / keyboard input · Arm and gripper control**

[Quick start](#quick-start) · [Deployment guide](README.deploy.md) · [Configuration](configs/rl/deploy.json) · [License](LICENSE)

## Features

- Run exported TorchScript policies with matching JSON inference contracts; select a compatible model with `--policy`.
- Control chassis velocity with PICO or a keyboard, with concurrent arm and gripper control in stand and RL modes.
- Use a two-stage standing trajectory and a persistent LowLevel control session.
- Package matching NX code and RK3588 profiles, and record feedback, targets, observations and actions.

The repository includes the default October 28 PPO export and November 5 / November 10 SuperMimic Stage 2 exports, each with a matching JSON runtime contract. HIM-PPO loading remains supported; no HIM model is bundled. Runtime operation does not import the training or sim2sim project. Device SDKs, services and the vendor LowLevel header are external dependencies.

## Architecture

```text
PICO / keyboard -> NX: observations, policy inference and control -> UDP -> RK3588: LowLevel bridge -> L1-W
                                  |
                                  +-> AIRBOT SDK subprocess -> AIRBOT service -> arm and gripper
```

PICO and AIRBOT SDKs run in separate subprocesses and communicate with the NX main process through local pipes. Policy inference and command publication run at 50 Hz, UDP state feedback targets 500 Hz, and the RK3588 motor loop runs at 1000 Hz.

## Working directories

Examples use these project locations and robot network addresses:

| Host | SSH target | Project directory |
| --- | --- | --- |
| Development PC | Local machine | `~/Projects/l1w-loco-manipulation-deploy` |
| NX | `robot@192.168.234.234` | `~/ws/l1w-loco-manipulation-deploy` |
| RK3588 | `firefly@192.168.234.1` | `~/l1w-loco-manipulation-deploy-controller` |

Configuration, models and relative `--policy` paths resolve from the project root.

## Quick start

Clone the project and run an offline inference check on the development PC. Use a Python version supported by PyTorch 2.8; the NX offline installer specifically targets Python 3.10 / Linux aarch64. Package requirements are in [requirements-rl.txt](requirements-rl.txt).

```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/lksk-bit/l1w-loco-manipulation-deploy.git
cd l1w-loco-manipulation-deploy
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install -r requirements-rl.txt
mkdir -p reports
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode benchmark --device cpu --steps 100 --report reports/benchmark.json
```

The report should contain `steps_recorded: 100` and `writes: 0`. This synthetic check exercises inference without connecting to a robot; it does not measure physical tracking performance. If `venv` reports a missing `ensurepip`, install the venv package matching your chosen Python interpreter. An existing compatible environment can also be used by its full Python path.

The CLI defaults to CUDA GPU inference. The offline development check explicitly uses `--device cpu`; NX requires CUDA-enabled PyTorch. See the [deployment guide](README.deploy.md) for installation.

Horizontal arm translation follows headset yaw with smoothing and a deadband. Turning the head does not move the target of a stationary hand. Chassis joystick mapping is unchanged.

## Deploy to the robot

Prepare the vendor header at `include/lowlevel/lowlevel.h` using [these instructions](include/lowlevel/README.md). It is excluded from Git. NX needs compatible PyTorch and AIRBOT / XRoboToolkit SDK installations; RK3588 needs `g++`, SPLINE and the vendor LowLevel interface. See the [deployment guide](README.deploy.md) for environment setup, startup order and controls.

**Development PC: update both hosts.** Exit NX RL and the RK3588 bridge before updating.

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
bash scripts/rl/deploy.sh all
```

The script checks processes, builds and verifies the package, and preserves NX's `.venv-rl`. It does not install Python dependencies or SDKs, start motor control, or stop vendor services. Packaging needs only Python 3.10+ and NumPy. To reuse an existing local interpreter, set `RL_PYTHON`; NX's runtime environment must be prepared separately.

**NX terminal 1: start AIRBOT.** Reuse an existing healthy service.

```bash
ssh robot@192.168.234.234
sudo airbot-arm --address 127.0.0.1:50051 -i can1 -t airbot_play_g2 --no-return
```

**RK3588 terminal: start the chassis bridge.** This takes exclusive control and stops vendor `motion_control`; exiting does not restart it.

```bash
ssh firefly@192.168.234.1
cd ~/l1w-loco-manipulation-deploy-controller
bash scripts/controller_control.sh l1-trial-profile.bin 3
```

**NX terminal 2: start PICO teleoperation.** Connect the headset's XRoboToolkit app to NX.

```bash
ssh robot@192.168.234.234
cd ~/ws/l1w-loco-manipulation-deploy
mkdir -p reports
TASK_RUN="pico-$(date +%Y%m%d-%H%M%S)-$$"
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode hardware --pico --arm-control \
  --log "reports/$TASK_RUN.jsonl" --report "reports/$TASK_RUN.json"
```

Startup is passive. Left X requests stand; holding left Trigger enables RL, and releasing it returns to stand. Right Grip enables arm translation, right Trigger opens the gripper, and holding right stick click closes it. Full semantics are in [PICO controls](README.deploy.md#7-pico-controls) and [shutdown](README.deploy.md#10-shutdown-and-restart).

## Code map

Daily configuration lives in [configs/rl/deploy.json](configs/rl/deploy.json).

| Purpose | File or directory |
| --- | --- |
| Policy, gains, stand/rest poses, offsets, gripper speed | [deploy.json](configs/rl/deploy.json) |
| Observations, history, inference and actions | [policy.py](l1w_teleop/rl/policy.py) |
| Configuration and model contracts | [config.py](l1w_teleop/rl/config.py), [models/l1w_rl/](models/l1w_rl/) |
| Initialization and shutdown | [runtime.py](l1w_teleop/rl/runtime.py) |
| Ordered input, feedback, FSM, inference and command cycle | [controller.py](l1w_teleop/rl/controller.py) |
| Stand/rest trajectories and state transitions | [body.py](l1w_teleop/rl/body.py) |
| Joint/frame mappings and arm workspace | [mapping.py](l1w_teleop/rl/mapping.py) |
| Input mapping and PICO subprocess | [teleop.py](l1w_teleop/rl/teleop.py), [pico_sdk.py](l1w_teleop/rl/pico_sdk.py) |
| Arm/gripper IPC and AIRBOT control | [arm.py](l1w_teleop/rl/arm.py), [airbot_sdk.py](l1w_teleop/rl/airbot_sdk.py) |
| NX protocol and diagnostics | [bridge.py](l1w_teleop/rl/bridge.py), [diagnostics.py](l1w_teleop/rl/diagnostics.py) |
| RK3588 motor loop and shared-memory integration | [src/](src/), [LowLevel dependency](include/lowlevel/README.md) |
| Deployment, environment setup and data collection | [scripts/rl/](scripts/rl/) |

Use `bash scripts/rl/deploy.sh nx` for NX-only code/model changes. Update both hosts with `all` when chassis mapping, posture parameters or the bridge profile changes.

## License and attribution

Project code is licensed under [Apache-2.0](LICENSE); attribution is in [NOTICE](NOTICE). Policy provenance, hashes and the inference contract are retained in the [model metadata](models/l1w_rl/policy_ppo_1028_19700.json) and [source notes](models/l1w_rl/SOURCE.md). External SDKs and vendor interfaces are described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Local experiment reports, validation records, historical policies, logs, environment directories and build artifacts are excluded from the public repository.
