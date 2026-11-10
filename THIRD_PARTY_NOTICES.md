# Third-party components / 第三方组件

## Project code

Original project contributions are licensed under [Apache-2.0](LICENSE). Attribution is recorded in [NOTICE](NOTICE).

## Policy runtime and reference assets

Portions of the policy runtime, posture logic and reference assets are adapted from `legged-manipulation_unilab`, supplied under Apache-2.0. The upstream license is retained at [models/l1w_rl/SOURCE_LICENSE](models/l1w_rl/SOURCE_LICENSE). Existing attribution comments remain in the source files.

The October 28 PPO and November 5 SuperMimic Stage 2 exports are described in [models/l1w_rl/SOURCE.md](models/l1w_rl/SOURCE.md). Their JSON files preserve model/checkpoint hashes and inference contracts. SuperMimic's network attribution and MIT terms are retained in [SUPERMIMIC_LICENSE](models/l1w_rl/SUPERMIMIC_LICENSE). The project license does not grant rights in external components beyond those held by their respective contributors.

## Vendor LowLevel interface

The vendor `lowlevel.h` is **not distributed in this Git repository**. The supplied local copy has no public redistribution license attached. Obtain it from the robot vendor and follow [the dependency instructions](include/lowlevel/README.md). A local copy may be included in a private device update bundle; that bundle is not a public release artifact.

## External services and SDKs

`arm_sdk`, `xrobotoolkit_sdk`, AIRBOT services and SPLINE are supplied separately by their respective vendors. Their code and binaries are not included or relicensed by this repository.

## 中文说明

本项目自有代码采用 Apache-2.0，保留上游代码的署名及许可证。模型来源、校验值与运行契约见模型目录的说明及 JSON。厂商 LowLevel 头文件不随 Git 仓库公开；AIRBOT、XRoboToolkit 和 SPLINE 由用户按各自条款准备。本项目许可证不替代外部组件的许可。
