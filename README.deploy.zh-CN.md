# l1w-loco-manipulation-deploy 部署与操作

[English](README.deploy.md) | 简体中文 · [项目首页](README.zh-CN.md)

本文以 `l1w-loco-manipulation-deploy` 为工程根目录。NX 运行策略、PICO 输入和 AIRBOT 客户端，RK3588 运行 LowLevel 桥接。
默认启动后为 passive，需手动请求站立和 RL。

## 1. 设备与目录

| 位置 | 登录地址 | 工程目录 | 用途 |
| --- | --- | --- | --- |
| 开发机 | 本机 | `~/Projects/l1w-loco-manipulation-deploy` | 修改、打包、更新 |
| NX | `robot@192.168.234.234` | `~/ws/l1w-loco-manipulation-deploy` | 推理、PICO、机械臂和夹爪 |
| 3588 | `firefly@192.168.234.1` | `~/l1w-loco-manipulation-deploy-controller` | 底盘 LowLevel 桥接 |

开发机按表中目录放置项目；以下命令均从相应设备的工作目录执行。模型和配置路径相对工程根目录解析，NX 运行日志统一写入工程内的 `reports/`。

PICO 的 XRoboToolkit App 连接 **NX 的 `192.168.234.234`**。
RL 指令使用 UDP 15051，NX 接收状态使用 UDP 15052；AIRBOT 服务为 NX 本机 TCP 50051。

## 2. 更新代码（开发机）

首次从 GitHub 获取代码：

```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/lksk-bit/l1w-loco-manipulation-deploy.git
cd l1w-loco-manipulation-deploy
```

打包前，按 [LowLevel 依赖说明](include/lowlevel/README.md) 自行取得厂商头文件，放入 `include/lowlevel/lowlevel.h`。该文件由 Git 忽略，不在公开仓库中；本机部署包会携带你提供的副本供 RK3588 编译。

更新前，先在 NX 的 RL 终端按 Ctrl+C，等退出动作结束；再在 3588 桥接终端按 Ctrl+C。
两个程序默认常驻，仅降到 passive 不代表进程已经退出。

更新两端：

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
bash scripts/rl/deploy.sh all
```

只更新一端时，选择下面一条执行：

```bash
bash scripts/rl/deploy.sh nx
```

```bash
bash scripts/rl/deploy.sh controller
```

| 修改内容 | 更新选择 |
| --- | --- |
| 模型、PICO 映射、NX 推理代码、夹爪参数 | `nx` |
| 3588 桥接代码 | `controller` |
| 站姿、卧姿、底盘增益、映射、底盘软限位 | `all`，同步匹配的 profile |
| 不确定影响范围 | `all` |

脚本打包本工程的代码、配置和模型，生成匹配 profile，并在 RK3588 编译桥接。安装时清理已合并的模块，保留其他本地文件及 NX 的 `.venv-rl`。
它不会自动启动控制，也不会安装 Python 依赖或设备 SDK。打包只需 Python 3.10+ 和 NumPy，无需 PyTorch。脚本优先在开发机打包；开发机缺少 NumPy 时，依次检查 NX 新工程的 `.venv-rl/bin/python`、系统 `python3` 和 `python`，找到可用解释器后才上传源码。也可通过 `RL_PYTHON` 指定开发机打包解释器，或通过 `RL_NX_PYTHON` 指定 NX 打包解释器；后者仅在开发机无可用环境时使用。

若开发机已有兼容环境，可直接指定解释器，无需重复安装：

```bash
RL_PYTHON="$HOME/path/to/existing-venv/bin/python" bash scripts/rl/deploy.sh all
```

把示例路径改成已有 Python 的路径。首次部署若两端均没有可用打包环境，在开发机执行：

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install 'numpy>=1.26,<3'
bash scripts/rl/deploy.sh all
```

仅打包部署不必安装 PyTorch；要运行离线推理，再按[快速开始](README.zh-CN.md#快速开始)安装完整依赖。打包解释器与 NX 运行环境分开：代码部署成功后，首次启动前仍需准备 NX 新工程的 `.venv-rl`，见[首次环境准备](#12-首次环境准备与模型维护)。

地址变化时，在开发机覆盖 SSH 目标：

```bash
RL_NX=robot@192.168.234.234 \
RL_CONTROLLER=firefly@192.168.234.1 \
bash scripts/rl/deploy.sh all
```

这里的变量只改变部署 SSH 目标。运行时底盘地址还需使用 NX 的 `--host`，
3588 回传 NX 地址使用启动脚本第三个参数；PICO App 也需要填写实际 NX 地址。

## 3. 启动前检查（NX）

```bash
ssh robot@192.168.234.234
cd ~/ws/l1w-loco-manipulation-deploy
systemctl is-active airbot-can1 xrobo-pc-service
ip -brief link show can1
.venv-rl/bin/python -c 'import numpy, torch, arm_sdk, xrobotoolkit_sdk; print("imports OK")'
```

在头显打开 XRoboToolkit App，连接 NX，保持前台运行及手柄跟踪。
键盘模式不需要 PICO；当前硬件 RL 仍读取 AIRBOT 真实反馈。

## 4. 启动 AIRBOT 服务（NX，独立终端）

```bash
ssh robot@192.168.234.234
sudo airbot-arm --address 127.0.0.1:50051 -i can1 -t airbot_play_g2 --no-return
```

保持该终端运行，看到 `gRPC server listening on 127.0.0.1:50051` 后继续。
已有服务正常运行时复用原服务，不要再启动第二份。
`--no-return` 是 AIRBOT 服务自己的退出选项，NX RL 的 Ctrl+C 回零由 RL 客户端另行请求。

## 5. 启动底盘桥接（3588，独立终端）

```bash
ssh firefly@192.168.234.1
cd ~/l1w-loco-manipulation-deploy-controller
bash scripts/controller_control.sh l1-trial-profile.bin 3
```

`3` 是桥接电机线程绑定的 CPU 编号，沿用当前部署值。
若 NX 地址变化，追加实际 NX IP：

```bash
bash scripts/controller_control.sh l1-trial-profile.bin 3 192.168.234.234
```

脚本会校验 profile、停止原厂 `motion_control`，再进入独占 LowLevel 会话。
保持该终端运行；桥接退出后不会自动恢复原厂运控。

## 6. 启动 RL（NX，另一个终端）

### PICO 控制底盘、机械臂和夹爪

```bash
ssh robot@192.168.234.234
cd ~/ws/l1w-loco-manipulation-deploy
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode hardware --pico --arm-control
```

指定模型并保存本次日志：

```bash
cd ~/ws/l1w-loco-manipulation-deploy
mkdir -p reports
TASK_RUN="pico-$(date +%Y%m%d-%H%M%S)-$$"
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode hardware --pico --arm-control \
  --policy models/l1w_rl/policy_ppo_1028_19700.pt \
  --log "reports/$TASK_RUN.jsonl" \
  --report "reports/$TASK_RUN.json"
```

`--log` 保存逐周期数据，`--report` 保存退出汇总；这里都写入 NX 工程的 `reports/`。先创建目录，并为每次测试使用不同的 `TASK_RUN`；日志文件以独占创建方式打开，已有同名文件会被拒绝。建议实机测试始终带这两个参数。
启动日志中的 `Policy:` 显示实际模型路径和 SHA-256。

### 键盘控制底盘，机械臂只读反馈

```bash
cd ~/ws/l1w-loco-manipulation-deploy
mkdir -p reports
TASK_RUN="keyboard-$(date +%Y%m%d-%H%M%S)-$$"
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode hardware --trial --keyboard \
  --log "reports/$TASK_RUN.jsonl" \
  --report "reports/$TASK_RUN.json"
```

`--trial` 不发送机械臂控制命令，不能与 `--arm-control` 同用。
`--pico` 和 `--keyboard` 也不能同用。
底盘地址变化时，在上述 NX 命令追加 `--host 实际3588地址`。

## 7. PICO 按键

先按左 X 完成站立，再操作其他控制。当前配置的两阶段站立共2秒：外展归位1秒、站起1秒。 RL 返回 stand 默认使用2秒单阶段三次插值，从最后下发目标直接过渡到站姿；`stand.rl_return_single_stage=false` 可恢复两阶段。passive 起立和 hold 恢复仍采用两阶段。

| 按键 | 功能 |
| --- | --- |
| 左 X | 请求站立；hold/故障恢复后需松开再按，重新接管 |
| 左 Trigger（前方食指扳机） | 按住进入 RL；松开返回 stand，并清零轮速目标 |
| 左摇杆前后、左右 | RL 下控制底盘前后、侧向速度 |
| 右摇杆左右 | RL 下控制底盘转向 |
| 右 Grip（侧面中指握把） | 按住并移动右手柄，控制机械臂末端平移；松开保持目标 |
| 右 Trigger（前方食指扳机） | 按住打开夹爪；松开保持实测开度 |
| 右摇杆按键（向下按压摇杆） | 按住闭合夹爪；松开保持实测开度 |
| 右 B | 请求降身到 passive，程序继续驻留 |

stand 和 RL 都支持机械臂、夹爪控制；passive/hold 不发送臂爪运动目标。
夹爪按键不需要同时按住右 Grip；开合同时按下时闭合优先。
按住右 Grip 不会暂停底盘，RL 下可同时操作底盘与机械臂。
右摇杆按键用于闭合夹爪，不用于机械臂回零。

机械臂水平操作轴跟随头显 yaw；上下方向与锁定末端姿态保持不变。仅累计新增手柄位移，转头不会旋转已累计目标。yaw 使用100 ms平滑时间常数和0.5°死区，跨越±180°时沿最短角更新；接近竖直视线时保持最后有效水平轴。头显跟踪无效时暂停跟随，恢复后重新建立参考，不补发中断位移。需在 XRoboToolkit 启用头部跟踪；底盘摇杆映射保持不变。

## 8. 键盘按键

| 按键 | 功能 |
| --- | --- |
| `1` | 降身到 passive，保持进程运行 |
| `2` | 请求 stand |
| `3` | 请求 RL，未完成起立时先等待站立完成 |
| `W / S` | 前后速度指令增减 |
| `A / D` | 侧向速度指令增减 |
| `Q / E` | 转向速度指令增减 |
| 空格 | 三个速度指令归零 |
| Esc / Ctrl+C | 请求降身并退出 NX 程序 |

速度键每次增减0.2；松键不会自动归零，使用空格停止。
状态切换会清空旧速度。恢复等待时按 `3` 也只请求重新站立，之后需再请求 RL。

## 9. 选择模型与调参

唯一日常配置：[configs/rl/deploy.json](configs/rl/deploy.json)。

| 配置项 | 内容 |
| --- | --- |
| `model` | 默认模型路径 |
| `rl_gains` | RL 腿部 Kp/Kd、轮子 Kd；可在 `by_algorithm` 下按 PPO/HIM-PPO 覆盖 |
| `stand` | 站姿、站立增益、两阶段时长 |
| `rest` | 卧姿及 passive 阻尼 |
| `mapping` | 关节方向、offset、IMU 安装旋转 |
| `arm.gripper` | 最大开度、闭合位置、开闭速度 |
| `arm.return_zero_on_exit` | Ctrl+C 后是否请求机械臂 SDK 回零 |

`--policy` 优先于配置中的 `model`，只影响本次启动。
默认使用以下 PPO 模型及同名元数据：

```text
models/l1w_rl/policy_ppo_1028_19700.pt
models/l1w_rl/policy_ppo_1028_19700.json
```

该模型来自 2026 年 10 月 28 日训练运行的 `model_19700.pt`。训练腿部 Kp/Kd 为 50/1；当前实机配置仍为 40/1。步态触发阈值 `|vx| > 0.5`、`|vy| > 0.08` 或 `|yaw| > 0.12` 随模型元数据加载。
PPO / HIM-PPO 加载能力保留；其他兼容模型需要自行提供 `440 → 22` TorchScript 文件及同名 JSON。公开仓库不附带历史模型、本机实验报告和来源快照。

相对模型路径以工程根目录解析；更换模型后重启 NX RL，不在运行中热切换。
### SuperMimic Stage 2

仓库附带 `policy_supermimic_20261105_20000.pt`、`policy_supermimic_20261110_20000.pt` 及同名 JSON。使用 `--policy` 选择，默认 PPO 模型不变：

```bash
cd ~/ws/l1w-loco-manipulation-deploy
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode hardware --pico --arm-control \
  --policy models/l1w_rl/policy_supermimic_20261110_20000.pt
```

仅测试底盘时，将 `--arm-control` 替换为 `--trial`。SuperMimic 输入为82维×10帧历史加10维当前目标，共830维，输出22维动作；内置观测与目标归一化。部署仅加载 actor，机械臂仍由 AIRBOT Cartesian/IK 控制。当前 SuperMimic 腿部 Kp/Kd 为40/1，轮子 Kd 为0.7；参数由 `rl_gains.by_algorithm.supermimic` 选择。模型来源与运行契约见[来源说明](models/l1w_rl/SOURCE.md)。

夹爪 `speed_m_s` 单位 m/s，开闭使用相同速度；当前部署配置为0.05，开度范围0～0.072m。
程序不根据夹爪电流/接触状态停止开合，AIRBOT 原生驱动保护仍由服务端和固件负责。

## 10. 正常退出与重新启动

1. 在 NX RL 终端按 Ctrl+C，等待身体降身及退出流程完成。
2. 启用 `--arm-control` 且配置允许时，程序在确认卧姿后请求机械臂 SDK 零位。
   `AIRBOT exit: zero_reached` 表示反馈到位；失败或跳过时会打印原因。
   再按一次 Ctrl+C 会跳过/取消剩余等待。
3. 需要关闭整个控制栈时，在3588桥接终端按 Ctrl+C，然后结束 AIRBOT 服务。

仅重启 NX RL 时，3588桥接可继续驻留。停止 AIRBOT 服务应在 NX 客户端退出之后进行。
原厂 `motion_control` 不会自动重启；恢复原厂运控需在结束 LowLevel 桥接后单独按原厂流程操作。

## 11. 日志与常用检查

NX 日志（需启动时指定 `--log` / `--report`）：

```bash
cd ~/ws/l1w-loco-manipulation-deploy
ls -lt reports/ | head -20
```

3588 桥接报告：

```bash
ls -lt ~/l1w-loco-manipulation-deploy-controller/reports/ | head -20
```

NX 检查输入服务和机械臂底层：

```bash
systemctl status xrobo-pc-service airbot-can1 --no-pager
ss -lntp
ip -details -statistics link show can1
journalctl -k -n 100 --no-pager
```

| 现象 | 先检查 |
| --- | --- |
| `waiting_bridge` 无法起立 | 3588桥接是否运行、双方IP、UDP端口及profile是否匹配 |
| `arm=unavailable` | AIRBOT服务、`can1`、SDK反馈及USB内核日志 |
| `arm_command=waiting_grip` | 是否按住侧面右Grip；它不是前方Trigger |
| PICO输入过期进入hold | App前台、头显休眠、手柄跟踪；恢复后松开再按左X |
| 夹爪不闭合 | 按住右摇杆按键，观察 `right_stick_click=True`、`state=closing` |
| 3588硬故障 | 最新报告的 `fault_reason` 和 `first_fault` |
| 更新被拒绝 | 两端常驻控制进程是否已各自退出 |

当前底盘周期：策略/指令50Hz，UDP状态反馈目标500Hz，桥接电机循环1000Hz。
周期与 ACK 超时退出已关闭。PICO 输入过期和桥接命令断流进入 hold，恢复后需重新请求站立。机械臂掉线、反馈过期或启用臂控时实测关节超限，仅暂停臂爪控制：底盘 RL 保留最后有效臂部观测，臂速置零、末端目标冻结，底盘关节和 IMU 继续实时更新。启动仍需至少一帧有效臂部反馈；恢复后需松开再握住右 Grip。限位数值保持不变。实现分别位于 [NX 控制循环](l1w_teleop/rl/controller.py)、[姿态状态机](l1w_teleop/rl/body.py) 和 [RK3588 桥接](src/lowlevel_bridge.cpp)。

## 12. 首次环境准备与模型维护

### NX Python 环境

代码安装与 Python 环境安装是两个步骤。先用开发机的兼容 Python 环境打包部署代码，再为 NX 工程准备 `.venv-rl`。
NX 的 Python 需能访问设备上的 AIRBOT / XRoboToolkit SDK，PyTorch 包需匹配设备架构和系统。

NX 工程的 `.venv-rl` 不由代码部署脚本创建。若已有兼容环境，可在新工程目录使用该环境的完整 Python 路径运行 CLI；代码、配置和模型仍从新工程读取。

若设备已有系统依赖，可创建能读取系统 Python 包的环境并检查：

```bash
cd ~/ws/l1w-loco-manipulation-deploy
python3.10 -m venv --system-site-packages .venv-rl
.venv-rl/bin/python -c 'import numpy, torch, arm_sdk, xrobotoolkit_sdk; print("imports OK")'
```

NumPy / PyTorch 版本范围见 [requirements-rl.txt](requirements-rl.txt)。缺少依赖时，项目提供 Python 3.10 / Linux aarch64 的 CPU 离线安装包工具。
在联网开发机执行：

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
python3 scripts/rl/prepare_nx_offline.py --cpu
```

将生成的 `build/l1w-nx-torch28-cpu.tar.gz` 复制到 NX 的主目录，然后在 NX 执行：

```bash
cd ~
tar -xzf l1w-nx-torch28-cpu.tar.gz
python3.10 ~/l1w-nx-torch28-cpu/install.py \
  --project ~/ws/l1w-loco-manipulation-deploy
```

该安装包安装 Python 依赖，不提供 AIRBOT / XRoboToolkit SDK，不配置 CAN、SPLINE 或设备服务。日常代码更新保留 `.venv-rl`，无需重复安装。

### GPU 推理

CLI 默认 `--device cuda`，也可显式指定 `--device cpu`。CUDA 不可用时直接报错，不静默切回 CPU；TorchScript 保持 float32。JetPack 6.2 / CUDA 12.6 的离线环境包在开发机准备：

```bash
python3 scripts/rl/prepare_nx_offline.py --jetson-cu126
scp build/l1w-nx-torch28-cu126.tar.gz robot@192.168.234.234:~/
```

在 NX 安装：

```bash
cd ~
tar -xzf l1w-nx-torch28-cu126.tar.gz
bash ~/l1w-nx-torch28-cu126/install_nx_gpu_runtime.sh
python3.10 ~/l1w-nx-torch28-cu126/install.py \
  --project ~/ws/l1w-loco-manipulation-deploy
cd ~/ws/l1w-loco-manipulation-deploy
.venv-rl/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

CPU 离线环境也可使用，但启动时需增加 `--device cpu`。ONNX GPU 推理另需提供 `CUDAExecutionProvider` 的 ONNX Runtime；部分不支持的算子仍可能分配给 CPU。

### 离线检查与手工打包

在开发机执行，不连接机器人：

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
mkdir -p reports
.venv-rl/bin/python -m l1w_teleop.rl \
  --mode benchmark --device cpu --steps 100 --report reports/benchmark.json
.venv-rl/bin/python scripts/rl/prepare_update.py
```

更新包为 `build/l1w-rl-update.tar.gz`。需要手工安装时，将包复制到相应设备并解压，在解压目录运行：

```bash
# NX
bash install.sh nx ~/ws/l1w-loco-manipulation-deploy
```

```bash
# RK3588
bash install.sh controller ~/l1w-loco-manipulation-deploy-controller
```

安装脚本的文件名是解压后的 `install.sh`；源码对应 [scripts/rl/install_update.sh](scripts/rl/install_update.sh)。

### 从 sim2sim 导入模型

这是可选开发步骤，需要源 sim2sim 工程及其依赖，部署运行不读取源工程。
将 `SIM2SIM_ROOT` 改为源工程的绝对路径；`--policy` 示例相对该源工程解析，并需改成实际导出文件的位置：

```bash
cd ~/Projects/l1w-loco-manipulation-deploy
SIM2SIM_ROOT="/path/to/sim2sim"
python3 scripts/rl/import_sim_policy.py \
  --source "$SIM2SIM_ROOT" \
  --policy exports/policy_ppo_1028_19700.pt
```

导入 HIM-PPO 时增加 `--algo him_ppo` 并指定对应的 `policy_him_ppo_*.pt`。
工具接收已导出的策略及同名 JSON，检查哈希和 `440 → 22` 运行契约，并写入 `models/l1w_rl/`；训练 checkpoint 需先在源工程完成导出。
生成模型及配套 JSON 后，更新 NX 并用 `--policy` 选择。

SuperMimic 可直接从 checkpoint 或运行目录导出。使用安装了训练工程依赖的 Python 环境：

```bash
python3 scripts/rl/import_sim_policy.py --algo supermimic --stage stage2 \
  --source /path/to/supermimic-training \
  --policy logs/path/to/run/ckpt_020000.pt
```

导入器只导出 actor，生成同名 JSON 并核对源端与部署端契约；部署运行不依赖训练工程。已有且内容不同的导出不会被覆盖。
