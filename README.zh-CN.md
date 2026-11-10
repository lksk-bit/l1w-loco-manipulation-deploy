# l1w-loco-manipulation-deploy

[English](README.md) | 简体中文

L1-W 轮足底盘与 AIRBOT G2 机械臂的 loco-manipulation 部署工程：在 NX 上运行运动策略和 PICO 遥操作，在 RK3588 上执行底盘控制。

**50 Hz 策略推理 · PPO / HIM-PPO / SuperMimic · PICO / 键盘输入 · 机械臂与夹爪控制**

[快速开始](#快速开始) · [双端部署](README.deploy.zh-CN.md) · [配置](configs/rl/deploy.json) · [第三方来源](THIRD_PARTY_NOTICES.md)

## 提供什么

- 加载随项目提供的 TorchScript 策略，或通过 `--policy` 选择兼容模型及同名 JSON 元数据。
- 使用 PICO 请求站立、切换 RL 和控制底盘速度；在 stand / RL 下同时操作机械臂和夹爪。
- 使用键盘进行底盘调试，记录关节反馈、控制目标、观测和动作。
- 一条命令打包并更新 NX / RK3588，生成配套的底盘 profile。

部署运行不依赖训练或 sim2sim 工程。仓库附带默认的 2026 年 10 月 28 日 PPO 模型，以及 11 月 5 日、11 月 10 日 SuperMimic Stage 2 模型及同名 JSON 运行契约；HIM-PPO 加载能力保留，未附带 HIM 模型。设备 SDK、SPLINE 服务和厂商 LowLevel 头文件需自行准备。

## 系统组成

```text
PICO / 键盘 ──► NX：观测、策略推理与控制调度 ──UDP──► RK3588：LowLevel 桥接 ──► L1-W
                         │
                         └── AIRBOT SDK 子进程 ──► AIRBOT 服务 ──► 机械臂与夹爪
```

PICO SDK 和 AIRBOT SDK 各运行在独立子进程，通过本地管道向 NX 主进程交换输入、反馈和目标。
底盘策略 / 指令为 50 Hz，UDP 状态反馈目标为 500 Hz，RK3588 电机循环为 1000 Hz。

## 工作目录

将项目文件夹放在开发机的 `~/Projects/` 下。本文统一使用以下工作目录：

| 位置 | 默认登录地址 | 工作目录 |
| --- | --- | --- |
| 开发机 | 本机 | `~/Projects/l1w-loco-manipulation-deploy` |
| NX | `robot@192.168.234.234` | `~/ws/l1w-loco-manipulation-deploy` |
| RK3588 | `firefly@192.168.234.1` | `~/l1w-loco-manipulation-deploy-controller` |

项目内部的 `configs/`、`models/`、`scripts/` 均相对工程根目录使用。`--policy` 的相对路径也以工程根目录解析。

## 快速开始

先在开发机完成一次离线推理，不需要连接机器人、PICO 或 AIRBOT 服务。
需要与 PyTorch 2.8 兼容的 Python；NX 离线安装包针对 Python 3.10 / Linux aarch64。NumPy / PyTorch 版本范围见 [requirements-rl.txt](requirements-rl.txt)。

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install -r requirements-rl.txt

mkdir -p reports
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode benchmark --device cpu --steps 100 \
  --report reports/benchmark.json
```

成功时会输出模型路径及 SHA-256，报告中的 `steps_recorded` 为 `100`、`writes` 为 `0`。
`benchmark` 使用合成状态检查推理链路，不代表仿真运动效果或实机性能。

CLI 默认使用 CUDA GPU 推理。开发机离线检查显式使用 `--device cpu`；NX 需要支持 CUDA 的 PyTorch，安装方法见[部署说明](README.deploy.zh-CN.md)。

机械臂水平平移轴跟随头显 yaw，采用平滑和死区处理；转头不会移动静止手柄对应的末端目标。底盘摇杆映射保持不变。

## 部署到机器人

NX 需要可导入的 `arm_sdk`、`xrobotoolkit_sdk` 和兼容的 PyTorch 环境；RK3588 需要 `g++`、SPLINE 服务及厂商 LowLevel 接口。
先按 [LowLevel 依赖说明](include/lowlevel/README.md) 将厂商头文件放到 `include/lowlevel/lowlevel.h`；该文件不随 Git 仓库公开。依赖准备、按键和完整启动顺序见 [部署与操作说明](README.deploy.zh-CN.md)。

**开发机：更新两端。** 更新前退出 NX RL 和 RK3588 桥接程序。

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
bash scripts/rl/deploy.sh all
```

脚本执行预检、打包、传输校验和安装，保留 NX 的 `.venv-rl`；不会自动启动控制程序或安装设备 SDK。

**NX 终端 1：启动 AIRBOT 服务。** 已有服务正常运行时直接复用。

```bash
ssh robot@192.168.234.234
sudo airbot-arm --address 127.0.0.1:50051 -i can1 -t airbot_play_g2 --no-return
```

**RK3588 终端：启动底盘桥接。** 该命令会停止厂商 `motion_control` 并取得底盘控制权；退出后不会自动恢复厂商运控。

```bash
ssh firefly@192.168.234.1
cd ~/l1w-loco-manipulation-deploy-controller
bash scripts/controller_control.sh l1-trial-profile.bin 3
```

**NX 终端 2：启动 PICO 遥操作并记录数据。** 头显内 XRoboToolkit App 连接 NX 地址。

```bash
ssh robot@192.168.234.234
cd ~/ws/l1w-loco-manipulation-deploy
mkdir -p reports
TASK_RUN="pico-$(date +%Y%m%d-%H%M%S)-$$"
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode hardware --pico --arm-control \
  --log "reports/$TASK_RUN.jsonl" \
  --report "reports/$TASK_RUN.json"
```

启动后处于 passive。左 X 请求站立，按住左 Trigger 进入 RL，松开返回 stand；右 Grip 控制机械臂末端平移，右 Trigger 打开夹爪，按住右摇杆按键闭合夹爪。
完整控制语义和 Ctrl+C 退出顺序见 [PICO 按键](README.deploy.zh-CN.md#7-pico-按键) 与 [正常退出](README.deploy.zh-CN.md#10-正常退出与重新启动)。

## 修改与扩展

日常参数集中在 [configs/rl/deploy.json](configs/rl/deploy.json)，修改后通过部署脚本更新设备。

| 想修改或检查的内容 | 文件 |
| --- | --- |
| 默认模型、RL 增益、站卧姿态和时长、offset、夹爪速度 | [configs/rl/deploy.json](configs/rl/deploy.json) |
| 模型加载、观测历史、归一化和动作处理 | [policy.py](l1w_teleop/rl/policy.py) |
| 配置校验和模型元数据 | [config.py](l1w_teleop/rl/config.py)、[models/l1w_rl/](models/l1w_rl/) |
| 生命周期、资源初始化与退出 | [runtime.py](l1w_teleop/rl/runtime.py) |
| 输入、反馈、FSM、推理和发送的周期顺序 | [controller.py](l1w_teleop/rl/controller.py) |
| 站卧流程与状态转移 | [body.py](l1w_teleop/rl/body.py) |
| 关节、IMU 与机械臂坐标映射、末端范围 | [mapping.py](l1w_teleop/rl/mapping.py) |
| PICO 按键与指令映射、输入采集和子进程 | [teleop.py](l1w_teleop/rl/teleop.py)、[pico_sdk.py](l1w_teleop/rl/pico_sdk.py) |
| 机械臂 / 夹爪进程通信与 SDK 控制 | [arm.py](l1w_teleop/rl/arm.py)、[airbot_sdk.py](l1w_teleop/rl/airbot_sdk.py) |
| NX 通信协议、日志和运行报告 | [bridge.py](l1w_teleop/rl/bridge.py)、[diagnostics.py](l1w_teleop/rl/diagnostics.py) |
| RK3588 电机循环、共享内存与协议 | [src/](src/)、[include/lowlevel/](include/lowlevel/) |
| 打包、安装、启动及模型导入 | [scripts/rl/](scripts/rl/) |

只更换 NX 模型或输入映射时可执行 `bash scripts/rl/deploy.sh nx`；修改底盘映射、站卧参数或 profile 时使用 `all` 同步两端。

## 来源与许可

策略运行和状态机的部分实现改编自 `legged-manipulation_unilab`，其 Apache-2.0 许可证见 [SOURCE_LICENSE](models/l1w_rl/SOURCE_LICENSE)。
项目代码采用 [Apache-2.0](LICENSE)，署名见 [NOTICE](NOTICE)。
模型来源、SHA-256 和运行契约保存在同名 JSON；本机实验、日志与历史快照不属于公开发布内容。
第三方 SDK、LowLevel 头文件的来源与许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
