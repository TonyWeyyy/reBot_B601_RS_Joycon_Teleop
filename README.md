# rebot-joycon-rs

将 [box2ai-robotics/rebot-joycon](https://github.com/box2ai-robotics/rebot-joycon) 的键盘、Joy-Con、仿真、重力补偿和 LeRobot 工作流迁移到 **Seeed reBot Arm B601-RS（RobStride 电机，SocketCAN）**。

本项目不是把 DM 的串口命令替换成 CAN 命令：控制后端、RS 坐标与限位、七路反馈、停止路径、输入看门狗和 LeRobot 设备桥接均已重写。六轴与夹爪始终通过同一个机械臂后端控制，不会为夹爪另开 CAN 主控。

> **真机安全边界**：当前代码已通过 59 项无硬件回归测试、Python 编译检查和 Shell 语法检查；模拟测试通过不等于真机安全认证。首次真机必须按下文顺序清场、固定机械臂、低速验证，并保持物理急停／电源开关可立即触达。

## 1. 功能对应

| 原项目功能 | RS 入口 | 说明 |
|---|---|---|
| 示例 0：标定 | `examples/0_calibrate_rebot.py` | POS_VEL 七路不使能反馈快照；不写硬件零点。MIT 读取需先使能，已默认拒绝 |
| 示例 1：键盘关节仿真 | `examples/1_ReBot_B601_RS_Key_qpos_SIM.py` | Fake 后端＋MuJoCo 显示 |
| 示例 2：键盘笛卡尔仿真 | `examples/2_ReBot_B601_RS_Key_Gpos_SIM.py` | RS URDF、解析雅可比、自适应阻尼 IK |
| 示例 3：Joy-Con 体感仿真 | `examples/3_ReBot_B601_RS_Joycon_Gpos_SIM.py` | 真实 Joy-Con 输入，Fake 机械臂 |
| 示例 4：键盘关节真机 | `examples/4_ReBot_B601_RS_Key_qpos_REAL.py` | MIT 或 POS_VEL 后端 |
| 示例 5：键盘笛卡尔真机 | `examples/5_ReBot_B601_RS_Key_Gpos_REAL.py` | 工作空间投影＋IK＋轨迹限速 |
| 示例 6：Joy-Con 体感真机 | `examples/6_ReBot_B601_RS_Joycon_Gpos_REAL.py` | 右 Joy-Con、独立安全键、IMU 看门狗 |
| 重力补偿 | `gravity_compensation/run_gravity_compensation.py` | 使用 Seeed 官方 `GravityCompensation`，不依赖原项目缺失包 |
| LeRobot 遥操作／录制／训练／推理 | `learning/` | 注册 `rebot_b601_rs` Robot 与 `joycon_rs` Teleoperator |

通用改进：

- 首选官方 MIT 控制器（兼容 `rebot_control.ReBotRSMITController` 和 `rebot.ReBotRSMITController`），不可用时回退 `reBotArm_control_py` POS_VEL。
- 所有键盘、Joy-Con、仿真和真机示例共用同一套安全控制链。
- 官方模型限位乘以 0.7 形成主机软限位（J1 额外采用更严格的 ±150° 再乘 0.7），且 **IK 直接在软限位内求解**，避免后截断引起末端跑偏。
- 默认 TCP 工作空间以肩部中心 `(0, 0, 0.145)` 为球心、半径 `0.58 m`（约 B601-RS 真实肩到末端展长的 70%，上限 0.62 m），`z ∈ [0.04, 0.65] m`。
- 默认 `HOME` 为水平向前的标准工作姿态 `[0°, 50°, 40°, 10°, 0°, 0°]`（`rpy ≈ [0°, 0°, 0°]`，六轴距软限位余量均 ≥ 40°）。
- 六轴等比例协调限速/限加速度 + 笛卡尔超前量限制（防低速 IK 死锁）；暂停、断线恢复和控制卡顿后从实际反馈重新起步。
- 键盘 Space（监听线程活性检查，持续按住不会误超时）／Joy-Con R（按住使能 + 60 Hz 真实 hidraw 数据包心跳看门狗）；断连触发后必须松开再重新按住。
- 原生单 `hidraw` 句柄 Joy-Con 驱动 + SO(3) 四元数相对姿态控制（方案 A 键位），无 90° 偏航反向、无多线程句柄冲突。
- 奇异点附近使用等比例步长缩放的自适应阻尼 IK，并支持位置优先贴边滑行，不会卡死。
- 夹爪力矩异常只允许反向卸载；严格模式默认要求夹爪力矩反馈，缺失时拒绝真机运动。
- 温度反馈允许启动宽限期，默认 75°C 预警、80°C 请求安全停止；宽限期后仍全部缺失会拒绝继续运行。
- 正常退出、异常、Ctrl+C 与 `SIGTERM`/`SIGHUP` 统一走“平滑回零→分轴验证实际反馈到位→失能”。`HOME` 是工作姿态，退出时的“回零”是模型七轴零位，两者不同。

## 2. 目录

```text
rebot-joycon-rs/
├── config/                  # RS 硬件、控制参数和 Joy-Con 映射
├── URDF/RS/urdf/            # 官方 RS 模型与轻量仿真模型
├── rebot_joycon/            # 后端、运动学、安全层、输入和 CLI
├── examples/                # 原示例 0–6 的 RS 入口
├── tools/                   # CAN、HOME、按键映射和测试工具
├── gravity_compensation/    # 官方 RS 重力补偿入口
├── learning/                # LeRobot RS 设备、遥操作和策略启动器
└── tests/                   # 无硬件回归测试
```

## 3. 安装

推荐在已经安装 Seeed 官方 SDK 的 Ubuntu＋conda 环境中运行：

```bash
conda activate <你的环境>
cd rebot-joycon-rs
bash install.sh
```

不在 conda 中时，脚本优先使用 Python 3.12（缺失则回退到 `python3`，要求 3.10+）创建 `.venv`。脚本安装基础依赖、克隆并安装 `joycon-robotics`，最后自动运行无硬件测试。

在 `config/default.yaml` 配置 SDK 路径，使对应包可以被导入。路径必须使 `rebot_sdk_path/rebot` 或 `rebot_sdk_path/rebot_control` 存在，且 `sdk_path/reBotArm_control_py` 存在。

例如当前 LAN-GER/rebot_control 仓库的 Python 包是内层 `rebot`，应这样配置：

```yaml
arm:
  backend: auto
  rebot_sdk_path: "/home/tony/wey/rebot_developer/rebot_control"
  sdk_path: "/home/tony/wey/rebot_developer/reBotArm_control_py"
```

如果你的机器是其他用户名或目录，只需要替换绝对路径，不要照抄 `/home/tony`。

- `auto`：优先 MIT；兼容 `rebot_control.ReBotRSMITController`、新版 `rebot.ReBotRSMITController` 和 `rebot.robot.ReBotRSMITController`；找不到 MIT 时回退 POS_VEL。
- MIT 使用角度制调用 `set_joint_angles()`；新版反馈接口为 `read_joint_angles()`，温度列表为 `last_temperatures`。该 SDK 没有暴露速度／力矩接口，后端会直接读取 `controller.motors[*].get_state()` 的 motorbridge 缓存（`vel`／`torq`），不另发 CAN 请求；启动时最多等待 2 秒让遥测线程填入第一帧夹爪力矩，严格力矩保护保持开启。
- POS_VEL 使用项目内 `config/rebotarm_rs.yaml`，其官方默认控制参数可能不如 MIT 平滑，首次真机更应保持低速。
- MIT 控制器自身的 CAN／硬件配置由官方 SDK 管理，确认真机同样为 `can0`、1 Mbps 和七路 RS 电机。

Joy-Con 在 Linux 上使用 evdev，不依赖 pygame／SDL 枚举 hid-nintendo。若 `bluetoothctl` 在 conda 下发生 D-Bus 崩溃，先 `conda deactivate` 再配对。输入设备权限不足时，配置本机 input 组／udev 权限并重新登录，不要长期用 root 运行控制程序。

## 4. CAN 与只读检查

每次重启或重新插拔 PCAN 后都要重新配置：

```bash
bash tools/setup_can.sh can0
ip -details link show can0
```

确认没有其他程序占用机械臂 CAN 总线。先采集只读快照：

```bash
python examples/0_calibrate_rebot.py --backend pos_vel
# MIT 的当前 SDK 读取流程需要先使能，项目会拒绝该只读请求：
# 不要通过 disable_motors() 在弯曲姿态强行采集。
```

输出为 `config/calibration_snapshot.json`。该文件只记录位置、温度及可用力矩，不是 DM 零偏文件，也不会修改电机零点。缺失力矩或温度写为 `null`。

> 只有 `pos_vel` 快照路径不会使能电机。MIT SDK 的读取需要先使能，因此新版默认拒绝 MIT 只读快照，不再读取后立即失能。

检查默认 HOME 的模型位置：

```bash
python tools/teach_home.py
```

默认 HOME 是 RS 模型坐标中的半展开姿态。需要更换时，可用 `--save-current` 读取当前反馈作为候选（仅 POS_VEL 支持不使能读取；MIT 请求会被拒绝），再人工检查 FK 后合入配置：

```bash
python tools/teach_home.py --save-current --backend pos_vel
```

## 5. 仿真与无真机链路测试

```bash
python examples/1_ReBot_B601_RS_Key_qpos_SIM.py
python examples/2_ReBot_B601_RS_Key_Gpos_SIM.py
python examples/3_ReBot_B601_RS_Joycon_Gpos_SIM.py
```

前两个使用键盘；第三个需要真实右 Joy-Con，但机械臂为 Fake 后端。可加 `--headless --duration 10` 做限时无窗口检查。仿真窗口显示的是经过安全层后实际发送的目标，不是另一套独立控制逻辑。

MuJoCo 默认使用 `config/default.yaml` 中 `simulation.urdf` 指定的模型；当前配置指向本机 RS 数字孪生场景（带网格、灯光和桌面）。该路径不存在时会自动回退官方完整 URDF，再回退 `ReBot_Arm_RS_simple.urdf` 的彩色基础几何体。

运行回归测试：

```bash
python tools/smoke_test.py
```

## 6. 键位与安全输入

### 键盘

运动必须全程按住 **Space**。启动后约有 1 秒保护时间。

| 功能 | 键位 |
|---|---|
| 关节 1–6 正向 | `1 2 3 4 5 6` |
| 关节 1–6 负向 | `q w e r t y` |
| 笛卡尔 XYZ 正／负 | `w/s`、`a/d`、`r/f` |
| 笛卡尔姿态三轴正／负 | `e/q`、`g/t`、`z/c` |
| 夹爪闭合／张开 | `7`／`u` |
| 持续平滑回 HOME | 按一次 `0`，仍需按住 Space |
| 安全回零并退出 | Esc |

夹爪使用独立键位；关节模式的 `1`／`3` 不会连带夹爪运动。

### 右 Joy-Con（方案 A，原生单 hidraw 驱动）

默认映射位于 `config/joycon_map.json`（均使用右 Joy-Con 真实物理按键）：

| 功能 | 默认键 | 说明 |
|---|---|---|
| 按住使能（离合器） | `R` (`r`) | 每次按住自动以当前机械臂实际姿态为锚点，松开保持姿态 |
| 水平前后 / 左右平移 | 右摇杆 | 世界系 `+X/-X`（前/后）、`+Y/-Y`（左/右） |
| 末端纯 Z 轴上升 / 下降 | `X` (`x`) / `B` (`b`) | 世界系纯 `+Z / -Z` 升降 |
| 末端三维姿态 | 手腕体感旋转 | 纯 SO(3) 四元数相对增量，无 90° 反向与万向节锁 |
| 夹爪开合切换 | `ZR` (`zr`) | 按住 `R` 时点按切换张开 / 闭合 |
| 平滑回 HOME | `+` (`plus`) | 按住 `R` 时点按平滑展开到水平 HOME |
| 安全回零并退出 | 长按 `Home` (0.6s) | 防误触；`R + 按下摇杆` (`r_stick`) 为立即安全停止 |
| LeRobot 提前结束／重录 | `A` (`a`) / `Y` (`y`) | 录制数据控制事件 |

实时诊断按键、摇杆和体感姿态，或交互重映射：

```bash
python tools/diagnose.py --joycon
python tools/map_joycon.py --print
python tools/map_joycon.py
```

## 7. 首次真机：必须按此顺序

1. 固定底座，移除工具和障碍物，人员离开工作区；手放在物理急停／电源开关旁。不要在弯曲姿态直接断电或直接失能。
2. 完成 CAN、只读快照、HOME FK 和输入映射检查。
3. 先运行对应 SIM／Fake 示例，确认按键方向、使能、夹爪和退出流程。
4. 从**关节空间**开始，六轴限速到 0.05 rad/s：

   ```bash
   python examples/4_ReBot_B601_RS_Key_qpos_REAL.py --backend auto --max-vel 0.05 --headless
   ```

   `--max-vel 0.05` 仅 2.9°/s，适合第一次点动验证；从折叠零位回 HOME 可能需要 20 秒以上。完成零位附近启动和停止验证后，可用 0.1–0.15 rad/s（约 5.7–8.6°/s）缩短 HOME 时间，但仍需清场并有人值守。

5. 程序默认读取并保持当前姿态，不会自动跳 HOME。输入大写 `RS` 后，轻点单个关节键验证方向；方向不符时只修改对应输入映射／符号，不要修改电机零点。
6. 松开 Space，确认目标保持；Esc 退出时确认平滑回零且反馈到位。若回零超时，程序会保持使能并报错，此时扶稳机械臂、排除阻挡，不要直接断电。
7. 关节空间验证完成后，再分别验证笛卡尔和 Joy-Con：

   ```bash
   python examples/5_ReBot_B601_RS_Key_Gpos_REAL.py --backend auto --max-vel 0.05 --headless
   python examples/6_ReBot_B601_RS_Joycon_Gpos_REAL.py --backend auto --max-vel 0.05 --headless
   ```

8. 只有在低速、清场条件下才验证 HOME、看门狗恢复和夹爪。不要为“测试急停”让机械臂带负载高速运动。

> 夹爪端点来自 `config/default.yaml` 的 `gripper_limits_rad`，不是从当前真机自动标定。首次验证夹爪必须空载、短行程点动，确认开合方向和端点；若方向或端点不符，先修正配置（或重新示教）再运行自动流程。

如需“启动后先到 HOME，再用 Joy-Con 操作”，显式加 `--home-on-start`。启动后程序会打印 `[等待] 未按住使能...`，此时机械臂保持当前姿态；**任何时候第一次按住使能**（Joy-Con R／键盘 Space）都会开始平滑回 HOME，请求不会被启动保护吞掉。回 HOME 途中松开使能会立即暂停；HOME 到位后需完全松开再重新按住，手柄输入才会生效。未确认当前姿态和 HOME 路径前不要使用。

若当前反馈超出软限位，程序会拒绝直接 clamp 当前位置造成跳变。请用 Seeed 官方低速恢复工具向安全区内移动，再运行本项目。唯一例外是七路统一允许 1° 反馈容差：RS 全零折叠位压在 J2/J3 限位下边界，回零后 J2/J3 可能出现约 -0.1° 的编码器负偏，该容差只放行这种零位附近误差，运动命令仍严格 clamp 在软限位内，并通过限速轨迹平滑进入 HOME。超过 1° 仍必须用官方工具恢复，并检查零位标定。

## 8. 重力补偿

需要 `arm.sdk_path` 指向官方 `reBotArm_control_py` 所在父目录：

```bash
python gravity_compensation/run_gravity_compensation.py
```

清场并扶稳后输入 `GC`，再按住 Space 启动。松开 Space 或超出六轴软限位会结束补偿并执行官方安全回零；程序检查六轴实际反馈到位后才断开，夹爪保持当前位置。该模式是重力前馈，不是位置遥操作，首次使用仍需单独低速验证。

## 9. LeRobot

RS 设备注册、遥操作、数据集录制、ACT 训练和策略推理见 [`learning/README.md`](learning/README.md)。策略输出同样经过七轴限位、单步变化限制、速度／加速度限制、TCP 工作空间检查和温度保护；不能把策略网络直接连接到电机 SDK。

## 10. 日志与故障排查

运行事件写入 `logs/`。常见问题：

- **找不到 SDK**：检查 `rebot_sdk_path`／`sdk_path` 是否是包的父目录，并在同一 Python 环境验证导入。
- **Pinocchio 二进制不兼容**：如果 `reBotArm_control_py` 报 `undefined symbol` 且路径来自 `~/.local`，说明用户目录的 Pinocchio 覆盖了 conda/venv 版本。项目现在会在导入 POS_VEL/重力补偿前优先当前环境的 `site-packages`；若仍失败，确认当前环境已安装兼容 `pinocchio`，并用 `python -s` 或 `PYTHONNOUSERSITE=1` 运行。
- **CAN 无反馈**：重新执行 `tools/setup_can.sh`，检查 1 Mbps、接线和其他 CAN 主控进程。
- **POS_VEL 只有位置、没有速度/力矩/温度**：部分 RS 固件不填充 `motorbridge` 的 feedback 缓存，但 `mechPos(0x7019)` 可读。程序会自动用参数读取兜底并打印提示；此时需要温度或夹爪力矩保护的场景应改用 MIT 后端。
- **Joy-Con 不存在**：检查右柄配对、hid-nintendo、`/dev/input/event*` 权限；用映射工具 `--print` 诊断。
- **Joy-Con 报 hidapi/`libhidapi` 找不到**：Linux 安装 `libhidapi-hidraw0`，或 `conda install -c conda-forge hidapi`。项目会优先使用 `hidraw` 后端以看到蓝牙 Joy-Con。
- **已配对但仍报 `joycon connect failed`**：上游 `joycon-robotics` 只接受 `9c:54`/`98:b6` 开头的序列号；本项目已在运行时放宽为非空序列号。若仍失败，检查 `/dev/hidraw*` 权限和是否被其他程序占用。
- **Joy-Con 转同一方向，腕部突然反向**：这是上游用 `sin(航向)` 当 yaw 导致的已知问题，已在运行时改为连续航向；若仍出现，检查是否在姿态限位附近，并降低 `joycon.imu_yaw_scale` 或转动幅度。
- **一直提示心跳超时**：安全心跳现在优先取 `joycon-robotics` 陀螺读取器的真实 hidraw 输入报告回调，手柄静止时也有持续数据包，不需要通过晃动手柄维持使能；evdev IMU 变化事件只作回退。若仍超时，说明 hidraw 数据包确实停止，检查右柄连接、电量、hidraw 权限和是否被其他程序占用；不要为了继续运动而关闭看门狗。
- **MIT 启动提示夹爪力矩反馈不可用**：当前 SDK 不暴露力矩接口，项目已从 `controller.motors[*].get_state()` 的 motorbridge 缓存读取，并在启动时等待最多 2 秒。若仍失败，说明电机反馈缓存未填充：检查 CAN 通信、SDK 遥测线程和固件反馈，不要为了启动而关闭 `require_gripper_torque_feedback`。
- **看门狗已锁存**：完全松开使能，再重新按住。
- **启动提示夹爪或 J2/J3 略超零位**：七路反馈统一允许 1° 零位容差，运动命令仍严格限幅；软限位本身不放宽。全零折叠位压在 J2/J3 下限，回零后约 -0.1° 属该容差范围。若偏差超过 1°，仍须用官方工具恢复并检查零位标定。
- **零位附近运动慢**：这是自适应阻尼在避免奇异位形跳变；先从已验证的 HOME 开始，不要提高 IK 阻尼阈值来硬锁死。
- **温度过高**：停止任务并冷却，不要提高 80°C 主机停止阈值来绕过保护。
- **安全回零未收敛**：保持机械臂支撑和使能状态，排除阻挡／通信故障；禁止直接 `disable_all` 或断电。
- **反复提示控制周期过长(dt>0.25s)**：当前 SDK 的 `read_joint_angles()` 每帧做 7 次 CAN 参数读取，可能让 30 Hz 主循环超时。后端现在优先使用 motorbridge 位置／速度／力矩缓存，启动时会与 SDK 精确读数比对并打印 `[arm] 使用 motorbridge 高速位置缓存`；若打印"回退 SDK 参数读取"，说明缓存不可信，需要检查电机反馈链路，不要直接放宽卡顿保护。
- **回零途中再次 Ctrl+C**：MIT 后端把它视为明确的立即中止请求，会调用 SDK `stop(return_to_zero=False)` 真正失能后退出，并打印 `[紧急中止]`；不会再出现程序退出但电机仍使能的情况。项目已独立确认零位后，SDK 正常停止不再二次回零。正常停止请只按一次 Ctrl+C、Esc 或手柄 Home。

## 11. 验证状态

已在本交付环境完成：

- 已通过 59 项无硬件回归测试，覆盖原生 Joy-Con `0x30` 协议解析、方案 A 键位、SO(3) 连续偏航、上电零包过滤与标定态发布、MIT 悬空待机防重力滑落保持与静差 HOME 收敛、边界 IK 容差、仿真 5s 宽限期后正常温度模拟、无 `mujoco` 环境下 `--headless` 跳过、键盘长按 Space 不误超时、低速 `--max-vel 0.05` 不死锁、软限位 IK、直线协调跟踪、MuJoCo 锁安全及重力补偿 CLI 与反馈校验。
- Joy-Con 零姿态在启动、暂停和 HOME 时对齐当前实际 TCP（不再是模型零位），避免首次使能或恢复时腕部突然转向旧目标。
- 全部 Python 文件编译检查与 Shell 脚本语法检查。
- RS 零位及 HOME 的 FK、工作空间和可操作度检查。

仍需在你的机器上完成：

- 官方 SDK 版本与七路真实反馈接口核对。
- MuJoCo 窗口、真实 Joy-Con／IMU、CAN 通信和两种后端的现场运行。
- 真机低速方向、停止、回零、温度、夹爪和工作空间验证。
- LeRobot 实际版本安装、摄像头、录制、训练和策略推理验证。
- 官方重力补偿参数与真实负载验证。

在这些项目完成前，请勿无人值守运行、携带人员附近负载或提高默认速度与限位。
