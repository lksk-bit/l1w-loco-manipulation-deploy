# Bundled policies

The original PPO export remains the default:

- File: `policy_ppo_1028_19700.pt`
- Algorithm: PPO, with embedded observation normalization.
- Training run: `2026-10-28_18-17-25_mujoco`, checkpoint `model_19700.pt`.
- Source export SHA-256: `6b7649c5904e38135b58b4e3a2ab1bf6a3bc756f46a2306edb09f6261e2da208`.
- Published artifact SHA-256: `1538f67c0ed3bfb53120289d46926a4b6b2e41a59275a08ba6313f9f555fdd8c`.
- Input: 440 values, formed from five 88-value observations.
- Output: 22 actions; the runtime contract sets the arm action scale to zero.

The matching JSON retains checkpoint and export hashes, source-file hashes, training configuration and the runtime contract. Provenance paths use logical project-relative names rather than local machine paths. TorchScript debug source records were removed to avoid publishing workstation paths. Tensor values and inference graphs are unchanged; both the source and release hashes are retained.

The model was exported using the `legged-manipulation_unilab-l1w` sim2sim workflow. Policy runtime code and reference assets adapted from that project retain its Apache-2.0 license in [SOURCE_LICENSE](SOURCE_LICENSE). Project attribution and external dependency scope are described in [NOTICE](../../NOTICE) and [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

The deployment loads the exported policy directly, without importing the training or sim2sim project. Physical arm control uses AIRBOT service-side IK. The JSON's training settings describe the source model; hardware gains are selected separately by `configs/rl/deploy.json`.

Historical policies, local experiment reports, validation traces and extraction snapshots are not part of the public release.

## SuperMimic Stage 2

- File: `policy_supermimic_20261105_20000.pt`, with same-stem JSON.
- Source project: `legged-manipulation_unilab-training-supermimic`.
- Run: `2026-11-05_17-58-22_mujoco`, checkpoint `ckpt_020000.pt`, iteration 20000, Stage 2 (operator-confirmed).
- Checkpoint SHA-256: `56ac24db532e08301ba93c64a3bb5bd206661b32aa78889ca866035f0b099ea4`.
- Export SHA-256: `63fd65af2d1dca3932320bdb18c909f5213a63aae8db24f3234c841c93cd4d44`.
- Input: 10 frames × 82 values, followed by 10 current command/goal values (830 total).
- Each frame: relative joint positions (22, wheel positions masked), joint velocities (22), body angular velocity (3), projected gravity (3), projected previous actions (22), command (10).
- Command: body vx/vy/yaw (3), feet phase in native sensor order (4), arm-base EE goal xyz (3). Joint order remains FBL/FAR/RAR/RBL, while phases use FBL/FAR/RBL/RAR.
- Output: 22 actor actions. Like the training bridge, runtime clips wheel actions to ±2.8 and zeros arm actions before storing previous actions. Legs use default + action × 0.25; wheels use action × 10 rad/s, limited to ±28 rad/s. AIRBOT runs Cartesian/IK arm control.
- Normalization: actor observation and goal running mean/variance are embedded, including the source epsilon (0.01). Only actor weights are deployed; no world model or critic is needed.
- History: oldest-to-newest, first-frame backfill after every policy reset.
- Gains: this algorithm now selects leg 40/1 and wheel Kd 0.7 from deployment settings. PPO/HIM settings, mapping, protocol, posture and timeout behavior are unchanged.

The checkpoint saves the learner configuration and actor weights, but **does not save a full environment configuration**. `args.json` was subsequently overwritten (12000 iterations versus 20000 in the checkpoint), so actor construction uses checkpoint arguments. The JSON records hashes of the current source files and the extracted Stage 2 environment contract; it does not claim an unavailable historical environment snapshot. Any training-time environment overrides must be reconciled before using a different run.

### Source phase discrepancy

The real task/playback zeros all four feet phases when gait is inactive. The source `L1WSuperMimicMDP.advance_goal()` used for imagined rollouts instead produces `[0, 0.5, 0.5, 0]` when inactive. Deployment follows the real task/playback. This existing source difference was identified during parity checks; training code was not changed as part of this deployment migration.

The exported gait thresholds retain the real command term's original Python values (`1.0`, `0.08`, `0.2`), rather than the MDP's float32 copies. Real playback promotes its float32 velocity to float64 for comparison: a keyboard yaw nudge stored as float32 `0.2` is slightly greater than the original `0.2` threshold and activates gait. Rounding the threshold to float32 during export incorrectly prevented that activation. SuperMimic phase accumulation and wrapping also use float32 to match real playback. Boundary validation now builds reference observations from the source command's own phases, independently of deployment phases; actor weights are unchanged.

SuperMimic attribution is retained in [SUPERMIMIC_LICENSE](SUPERMIMIC_LICENSE). The exporter strips TorchScript debug source records containing workstation paths; actor tensors and the exported computation remain intact. The same-stem JSON is the authoritative artifact hash and runtime contract. Offline parity does not validate NX scheduling or physical robot behavior.

## SuperMimic Stage 2 — November 10

- Files: `policy_supermimic_20261110_20000.pt` and same-stem JSON; the default model is unchanged.
- Source run: `logs/supermimic-l1w-stage2/L1WArmManipLoco/2026-11-10_00-09-55_mujoco/ckpt_020000.pt`, iteration 20000, imported as Stage 2.
- Checkpoint SHA-256: `d9dafc9039a8c28d6d4fa68b56ac70bf7a2c82e26bc29b008842321b1e2b8fb9`.
- Export SHA-256: `61deecebcecd97eb51d123155731653a37b20674e244a336f04ee55d6be1a445`.
- Actor input/output: 830 → 22, with embedded observation/goal normalization; no world model or critic.
- Current source defaults specify leg Kp/Kd 40/1; deployment was subsequently adjusted to 40/1 for SuperMimic. The import itself did not modify hardware gains. The checkpoint lacks a complete environment snapshot, so source defaults do not prove historical training overrides.
- Model construction uses saved learner arguments. Source hashes and the current Stage 2 runtime contract are retained in the JSON. Attribution and license remain as described above.

## 中文说明

公开仓库附带默认的 2026 年 10 月 28 日 PPO 模型，以及 11 月 5 日和 11 月 10 日 SuperMimic Stage 2 模型，均有同名 JSON。SuperMimic 使用 82×10+10 输入，当前选择腿部 40/1 增益；现有 PPO/HIM 控制不变。源 checkpoint 不含完整环境快照，环境参数取自当前训练工程的 Stage 2 配置；发现源工程真实环境与想象 rollout 在不启动步态时相位不一致，部署跟随真实环境，未修改训练代码。模型调试记录中的本机路径已移除，来源和运行契约保留；本机验证记录只放在被忽略的 build 中。

另已导入 11 月 10 日 Stage 2 的 20000 步模型，默认模型不变。当前源码腿部增益为 40/1，导入后部署 SuperMimic 已另行调整为 40/1；未执行硬件运动。
