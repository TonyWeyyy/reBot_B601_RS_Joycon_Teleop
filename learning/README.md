# B601-RS LeRobot 桥接

本目录把原项目的 DM 专用 Example 6 后台 worker 改为标准 LeRobot 设备：

- `rebot_b601_rs`：七轴 Robot，负责所有真实硬件反馈、动作执行和安全停止。
- `joycon_rs`：右 Joy-Con Teleoperator，只生成动作，不直接访问 CAN。
- `run_rebot_policy.py`：策略推理启动器，复用同一个 RS Robot。

动作和观测中的 `*.pos` 使用 **degree**，顺序为：

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper
```

项目核心控制与配置使用 rad，转换只在 LeRobot 边界发生。不要加载 DM 标定文件或 DM 动作零偏。

## 1. 安装原项目配套 LeRobot

先完成根目录 `install.sh` 并激活同一个 conda 或 `.venv` 环境，然后：

```bash
cd /绝对路径/rebot-joycon-rs
python -m pip install uv
bash learning/install_lerobot.sh
export ROOT="$PWD"
export PYTHONPATH="$ROOT/learning:$ROOT:${PYTHONPATH:-}"
python -c "import rebot_rs_lerobot; print('RS LeRobot 设备注册成功')"
```

脚本会浅克隆原 `rebot-joycon` 仓库到 `.deps/rebot-joycon-upstream`，并安装其中 `Learning_Model/lerobot`，从而保持与原项目训练／推理脚本的 API 配套。LeRobot 及其依赖较重；安装成功不代表真机接口已经验证。

每次新开终端运行自定义设备前，都要设置上述 `PYTHONPATH`。`learning/sitecustomize.py` 会自动导入并注册设备。

## 2. 摄像头

下面示例使用一个 OpenCV 前置摄像头：

```json
{
  "front": {
    "type": "opencv",
    "index_or_path": 0,
    "width": 640,
    "height": 480,
    "fps": 30,
    "fourcc": "MJPG"
  }
}
```

按本机情况修改 `index_or_path`。先独立验证摄像头画面和帧率，再开始真机录制。摄像头名称必须与训练和推理保持一致。

## 3. Joy-Con 遥操作与数据集录制

先完成根目录文档中的 CAN、只读反馈、HOME、右 Joy-Con 映射和低速真机验证。录制不是绕过安全层的独立路径：Teleoperator 产生目标，RS Robot 再次执行反馈检查、软限位、单步变化限制、速度／加速度限制、工作空间、温度和夹爪保护。

以下命令采用原项目配套 LeRobot 的常用 CLI；不同上游提交可能调整参数，执行前先用 `python -m lerobot.scripts.lerobot_record --help` 核对：

```bash
cd "$ROOT"
python -m lerobot.scripts.lerobot_record \
  --robot.type=rebot_b601_rs \
  --robot.port=can0 \
  --robot.id=rebot_rs \
  --robot.control_backend=mit \
  --robot.home_on_connect=false \
  --robot.max_relative_target=5.0 \
  --robot.project_config="$ROOT/config/default.yaml" \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
  --teleop.type=joycon_rs \
  --teleop.project_config="$ROOT/config/default.yaml" \
  --dataset.repo_id="$USER/rebot_b601_rs_pick_place" \
  --dataset.single_task="抓取物体并放到目标区域" \
  --dataset.num_episodes=50 \
  --dataset.fps=30 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.video=true
```

- 默认 `home_on_connect=false`，连接时保持当前姿态，避免突然跳 HOME。
- `max_relative_target=5.0` 表示单轴单步最多相对当前反馈变化 5°，之后仍会经过轨迹限速。
- 运动时按住 R（R 自带的上升功能会被屏蔽）；A 提前结束当前 episode，Y 请求重录，- 请求 HOME，Home 请求安全回零退出。
- 先录 1–2 段并回放检查关节顺序、单位、夹爪方向、图像与时间同步，再批量采集。
- 数据量、摄像头视角和训练轮数取决于任务；本项目不虚构固定的“足够数据集”数量。

## 4. ACT 训练

训练本身不需要连接机械臂，可关闭所有 CAN 控制程序。示例：

```bash
cd "$ROOT"
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$USER/rebot_b601_rs_pick_place" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="$ROOT/.deps/rebot-joycon-upstream/Learning_Model/lerobot/outputs/train" \
  --job_name=rebot_smoke \
  --steps=1000
```

先用少量 step 验证数据管线和checkpoint，再按任务增加训练。实际显存、batch size、图像策略和恢复训练参数以所安装 LeRobot 的 `lerobot_train --help` 为准。

## 5. 策略推理

编辑 `learning/run_rebot_policy.py` 顶部：

- `POLICY_PATH`：训练 checkpoint 的 `pretrained_model` 目录。
- `CAMERA_INDEX`：摄像头序号。
- `DEVICE`：`cuda` 或 `cpu`。
- `TASK`、`DURATION`：任务文本与推理时长。

随后：

```bash
cd "$ROOT"
python learning/run_rebot_policy.py
```

启动器会检查 Python 环境、原项目 LeRobot 和必要 checkpoint 文件，再启动 `lerobot_rollout`。默认 `return_to_initial_position=false`，避免推理框架额外发起未经确认的复位。

首次策略推理必须先清场、移除负载，并把 `config/default.yaml` 的六轴速度降至首次验证水平；也可以使用 Robot 配置进一步收紧单步变化。策略输出异常、超出工作空间或反馈失效时，不允许直接连接电机绕过 RS Robot。

## 6. 停止与失败语义

- `RebotRSFollower.disable_torque()` 仅为兼容 LeRobot API 保留名称，实际执行安全回零，不是在弯曲姿态直接失能。
- `disconnect()` 先请求安全回零，再断开摄像头。
- HOME 超时、反馈缺失、非有限动作、超温或安全回零失败都应视为未完成任务；不要通过捕获异常后直接断电来“结束程序”。
- 真机策略验证完成前，不要无人值守运行。

## 7. 尚未替代的现场验证

当前交付环境没有安装完整 LeRobot 依赖、真实摄像头或机械臂，因此未实际执行录制、ACT 训练和策略 rollout。桥接代码已做静态编译检查，但你仍需在锁定的 LeRobot 版本上完成一次无运动注册检查、短数据集录制、checkpoint 加载和低速单段推理。
