# Bundled policy

The public repository contains one policy export:

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

## 中文说明

公开仓库仅附带 2026 年 10 月 28 日的 `policy_ppo_1028_19700.pt` 及同名 JSON。已移除 TorchScript 中含本机路径的调试记录，权重与推理图保持不变；元数据中的本机绝对路径已替换为逻辑相对路径；训练来源、哈希与运行契约保留。训练、仿真工程不参与部署运行，实机增益由部署配置单独指定。
