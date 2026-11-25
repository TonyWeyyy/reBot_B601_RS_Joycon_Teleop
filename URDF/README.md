# RS 模型来源与用途

## 文件

- `RS/urdf/ReBot_Arm_RS.urdf`：来自 Seeed 官方 `reBotArm_control_py` 的 B601-RS 模型，保留作关节链、轴向、零位、惯性、限位和末端帧的权威来源。
- `RS/urdf/ReBot_Arm_RS_simple.urdf`：由官方 RS 模型生成的轻量仿真版本，供本项目 FK／IK 和 MuJoCo 回退显示使用；已为每个 link 写入基础几何体颜色。

官方来源仓库：[Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm)。硬件配置中的原始模型路径为 `urdf/RS/urdf/ReBot_Arm_RS.urdf`，末端帧为 `gripper_end`。

## 轻量模型保留了什么

轻量版保留官方六轴链、关节原点与 RPY、轴向、限位、惯性、基座和 `gripper_end` 末端帧。只把依赖大型网格文件的视觉／碰撞几何替换为基础几何体，因此压缩包不需要携带不完整的大型 mesh。

MuJoCo 显示优先使用 `config/default.yaml` 的 `simulation.urdf`（可为 URDF 或 MJCF）；未配置或路径不存在时，尝试官方 SDK 的完整 `ReBot_Arm_RS.urdf`，最后回退本轻量模型。轻量模型适用于：

- RS 模型坐标下的正运动学、雅可比和阻尼最小二乘 IK。
- 显示经过安全层后的目标姿态。
- 无硬件控制逻辑验证。

它不适用于：

- 经认证的真实碰撞检测或精确外形干涉判断。
- 接触、抓取力、电机动力学或高保真动力学仿真。
- 替代真机工作空间和夹爪验证。

原始 URDF 仍可能引用官方仓库中的网格路径；这些网格未随本压缩包复制。若要在其他工具中使用原始模型，请从相同版本的官方仓库补齐网格。默认配置使用轻量版，不依赖这些外部文件。

## 坐标与零偏

所有角度均为 RS 模型坐标。特别是 J2／J3 使用 RS 模型的正向范围，不能套用 DM 项目的负角表示或软件零偏。`config/calibration_snapshot.json` 只是只读反馈快照，不参与模型零位修正。

零位和 HOME 的检查结果仅验证运动学一致性；真机首次运行仍需核对实际关节方向、机械安装和末端负载。
