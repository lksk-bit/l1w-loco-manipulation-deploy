# Vendor LowLevel dependency / 厂商 LowLevel 依赖

## English

Obtain the L1-W `lowlevel.h` header from the robot vendor and place it here:

```text
include/lowlevel/lowlevel.h
```

The vendor header is required to compile the RK3588 bridge. It is excluded from Git because the supplied copy has no public redistribution license attached. Do not replace it with a fabricated interface or change its shared-memory layout.

Once supplied locally, `scripts/rl/prepare_update.py` includes your copy in the private device update bundle. Do not publish that bundle without the relevant permissions. AIRBOT / XRoboToolkit SDKs and SPLINE are also supplied separately.

## 简体中文

向机器人厂商取得 L1-W 的 `lowlevel.h`，放入上述路径。RK3588 桥接编译需要此文件。

收到的头文件未附公开再分发许可证，因此由 Git 忽略，不随公开仓库提供。请使用厂商接口，保持共享内存布局不变。

本机提供头文件后，打包工具会将其加入设备更新包供编译使用；更新包不属于公开仓库内容。AIRBOT / XRoboToolkit SDK 和 SPLINE 也需自行安装。
