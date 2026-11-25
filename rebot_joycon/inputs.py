from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .config import Settings
from .joycon_driver import JoyConSnapshot, NativeJoyConDriver


@dataclass
class InputSample:
    deadman: bool
    heartbeat: float | None
    streaming_heartbeat: bool = True
    joint_direction: np.ndarray = field(default_factory=lambda: np.zeros(6))
    cartesian_direction: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # Joy-Con 速度 + 纯 SO(3) 相对姿态模式（避免绝对位姿跳变和欧拉角万向节锁）。
    cartesian_velocity_m_s: np.ndarray | None = None
    target_rotation: np.ndarray | None = None
    # 兼容外部直接给绝对 6D 位姿的测试或调用方。
    absolute_pose: np.ndarray | None = None
    gripper_absolute: float | None = None
    gripper_direction: float = 0.0
    home: bool = False
    exit: bool = False
    estop: bool = False
    episode_events: list[int] = field(default_factory=list)


class KeyboardInput:
    """事件驱动键盘输入。

    心跳改为监听线程健康状态（streaming_heartbeat=False）：按住 Space 超过 0.2s
    不再因 Linux X11 按键重复延迟误触“输入心跳超时”；监听线程异常退出时仍立即触发看门狗。
    """

    JOINT_POS = tuple("123456")
    JOINT_NEG = tuple("qwerty")
    CART_POS = tuple("w a r e g z".split())
    CART_NEG = tuple("s d f q t c".split())

    def __init__(self, settings: Settings):
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise ImportError("键盘控制需要安装 pynput，并使用 Linux 图形桌面/X11") from exc
        self._keyboard = keyboard
        self.deadman_key = str(settings.section("keyboard").get("deadman_key", "space")).lower()
        self._pressed: set[str] = set()
        self._home_edge = False
        self._exit_edge = False
        self._lock = threading.Lock()
        self._last_event = time.monotonic()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    @property
    def last_event_time(self) -> float:
        with self._lock:
            return float(self._last_event)

    def _key(self, key) -> str | None:
        char = getattr(key, "char", None)
        if char is not None:
            return char.lower()
        name = getattr(key, "name", None)
        if name:
            return str(name).lower()
        text = str(key).replace("Key.", "").lower()
        return text or None

    def _on_press(self, key) -> None:
        name = self._key(key)
        with self._lock:
            self._last_event = time.monotonic()
            if name == "esc":
                self._exit_edge = True
            elif name == "0" and "0" not in self._pressed:
                self._home_edge = True
            if name:
                self._pressed.add(name)

    def _on_release(self, key) -> None:
        name = self._key(key)
        with self._lock:
            self._last_event = time.monotonic()
            self._pressed.discard(name)

    def poll(self) -> InputSample:
        with self._lock:
            pressed = self._pressed.copy()
            home, self._home_edge = self._home_edge, False
            exit_, self._exit_edge = self._exit_edge, False
            running = bool(self._listener.running)
        joint = np.zeros(6)
        for index, (positive, negative) in enumerate(zip(self.JOINT_POS, self.JOINT_NEG, strict=True)):
            joint[index] = float(positive in pressed) - float(negative in pressed)
        cartesian = np.zeros(6)
        for index, (positive, negative) in enumerate(zip(self.CART_POS, self.CART_NEG, strict=True)):
            cartesian[index] = float(positive in pressed) - float(negative in pressed)
        return InputSample(
            deadman=getattr(self, "deadman_key", "space") in pressed,
            heartbeat=time.monotonic() if running else None,
            streaming_heartbeat=False,
            joint_direction=joint,
            cartesian_direction=cartesian,
            # 夹爪独立键位；不能复用 1/3，否则关节模式会同时运动两处。
            gripper_direction=float("7" in pressed) - float("u" in pressed),
            home=home,
            exit=exit_,
        )

    def close(self) -> None:
        self._listener.stop()
        self._listener.join(timeout=1.0)


class JoyConInput:
    """基于单 hidraw 句柄 NativeJoyConDriver 的安全体感遥操作输入（方案 A）。

    右 Joy-Con 默认键位：
    - R（按住）：运动使能（离合器，每次重新按住自动以当前机械臂姿态为基准）
    - 摇杆：水平面 +X/-X（前/后）、+Y/-Y（左/右）平移
    - X / B：纯世界 +Z / -Z 升降
    - 手腕旋转：纯 SO(3) 相对姿态控制（无 90° 反向、无万向节锁）
    - ZR：点按切换夹爪张开 / 闭合
    - +：平滑回 HOME（需按住使能）
    - 长按 Home（默认 0.6s）或 R + 按下摇杆：统一安全回零退出
    - A / Y：LeRobot 提前结束 / 重录 episode
    """

    def __init__(self, settings: Settings, home_xyz: np.ndarray, gripper_open: float,
                 gripper_close: float, home_rpy: np.ndarray | None = None,
                 *, driver=None):
        self.settings = settings
        self.mapping = settings.joycon_map()
        config = settings.section("joycon")
        safety = settings.section("safety")

        self.require_imu_heartbeat = bool(safety.get("require_imu_heartbeat", True))
        self.linear_speed_m_s = float(config.get("linear_speed_m_s", 0.14)) * float(
            config.get("position_scale", 1.0))
        self.vertical_speed_m_s = float(config.get("vertical_speed_m_s", 0.10)) * float(
            config.get("position_scale", 1.0))
        self.rotation_scale = float(config.get("rotation_scale", 1.0))
        self.imu_yaw_scale = float(config.get("imu_yaw_scale", 0.8))
        raw_imu_signs = config.get("imu_axis_signs", [1.0, 1.0, 1.0])
        self.imu_axis_signs = np.asarray(raw_imu_signs, dtype=float).reshape(3)
        raw_stick_signs = config.get("stick_axis_signs", [1.0, 1.0])
        stick_signs = (float(raw_stick_signs[0]), float(raw_stick_signs[1]))
        self.exit_hold_s = float(config.get("exit_hold_s", 0.6))
        self.startup_guard_s = float(self.mapping.get("startup_guard_s", 1.0))

        self.gripper_open = float(gripper_open)
        self.gripper_close = float(gripper_close)
        self.gripper_state = float(gripper_close)

        if driver is not None:
            self.driver = driver
        else:
            self.driver = NativeJoyConDriver(
                device=str(config.get("device", "right")),
                gyro_deadband_rad_s=float(config.get("imu_gyro_deadband_rad_s", 0.03)),
                lowpass_alpha=float(config.get("imu_lowpass_alpha", 0.20)),
                stick_deadzone=float(config.get("stick_deadzone", 0.15)),
                stick_axis_signs=stick_signs,
            )

        self._started = time.monotonic()
        self._prev_buttons: dict[str, bool] = {}
        self._exit_press_started: float | None = None
        # 离合锚点：当前使能段开始时的 (手柄姿态, 机械臂实际末端姿态)。
        self._imu_anchor: np.ndarray | None = None
        self._arm_rot_anchor: np.ndarray = (
            Rotation.from_euler("xyz", np.asarray(home_rpy, dtype=float)).as_matrix()
            if home_rpy is not None else np.eye(3)
        )
        self._arm_pos_anchor: np.ndarray = np.asarray(home_xyz, dtype=float).copy()

    def sync_anchor(self, arm_pose: np.ndarray) -> None:
        """将当前 Joy-Con 姿态与机械臂实际 TCP 位姿重新锚定（无跳变离合）。"""
        pose = np.asarray(arm_pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return
        snap = self.driver.snapshot()
        self._arm_pos_anchor = pose[:3, 3].copy()
        self._arm_rot_anchor = pose[:3, :3].copy()
        self._imu_anchor = snap.rotation.copy()

    def rebase(self, xyz: np.ndarray, rpy: np.ndarray) -> None:
        """兼容接口：等价于 sync_anchor(make_pose(xyz, rpy))。"""
        xyz = np.asarray(xyz, dtype=float)
        rpy = np.asarray(rpy, dtype=float)
        if (xyz.shape != (3,) or rpy.shape != (3,)
                or not np.isfinite(xyz).all() or not np.isfinite(rpy).all()):
            return
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = xyz
        pose[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        self.sync_anchor(pose)

    def _scaled_relative_rotation(self, current_imu_rot: np.ndarray) -> np.ndarray:
        """计算相对锚点的 SO(3) 增量旋转，支持单轴符号与 yaw 缩放。"""
        if self._imu_anchor is None:
            return self._arm_rot_anchor.copy()
        r_rel = current_imu_rot @ self._imu_anchor.T
        rotvec = Rotation.from_matrix(r_rel).as_rotvec()
        scale = np.array([
            self.rotation_scale * self.imu_axis_signs[0],
            self.rotation_scale * self.imu_axis_signs[1],
            self.rotation_scale * self.imu_yaw_scale * self.imu_axis_signs[2],
        ], dtype=float)
        scaled_vec = rotvec * scale
        if float(np.linalg.norm(scaled_vec)) < 1e-12:
            return self._arm_rot_anchor.copy()
        return Rotation.from_rotvec(scaled_vec).as_matrix() @ self._arm_rot_anchor

    def poll(self) -> InputSample:
        snap: JoyConSnapshot = self.driver.snapshot()
        now = time.monotonic()
        buttons = snap.buttons

        def pressed(role: str) -> bool:
            key = self.mapping.get(role)
            return bool(buttons.get(key, False)) if key else False

        def rising(role: str) -> bool:
            key = self.mapping.get(role)
            if not key:
                return False
            curr = bool(buttons.get(key, False))
            prev = bool(self._prev_buttons.get(key, False))
            return curr and not prev

        deadman = pressed("deadman")
        estop = bool(deadman and pressed("estop"))
        past_guard = (now - self._started) >= self.startup_guard_s

        home = bool(past_guard and rising("home"))

        # 长按 Home（默认 0.6s）触发安全回零退出，避免误触；exit_hold_s=0 时点按即退。
        exit_pressed = pressed("exit")
        exit_triggered = False
        if exit_pressed and past_guard:
            if self._exit_press_started is None:
                self._exit_press_started = now
            if (now - self._exit_press_started) >= self.exit_hold_s:
                exit_triggered = True
        else:
            self._exit_press_started = None

        # ZR 上升沿切换夹爪开合（启动保护后且按住使能时生效，避免闲置误触排队动作）。
        if past_guard and deadman and rising("gripper"):
            if abs(self.gripper_state - self.gripper_open) < 1e-6:
                self.gripper_state = self.gripper_close
            else:
                self.gripper_state = self.gripper_open

        events: list[int] = []
        if past_guard and rising("episode_end"):
            events.append(1)
        if past_guard and rising("episode_retry"):
            events.append(-1)

        # 记录本帧按键状态用于下一帧上升沿检测。
        self._prev_buttons = dict(buttons)

        # 世界系平移速度：摇杆驱动 X/Y，X/B 驱动纯 Z 上升/下降。
        vz = (float(pressed("z_up")) - float(pressed("z_down"))) * self.vertical_speed_m_s
        velocity = np.array([
            float(snap.stick_xy[0]) * self.linear_speed_m_s,
            float(snap.stick_xy[1]) * self.linear_speed_m_s,
            vz,
        ], dtype=float)

        if self._imu_anchor is None:
            self._imu_anchor = snap.rotation.copy()
        target_rot = self._scaled_relative_rotation(snap.rotation)

        return InputSample(
            deadman=deadman,
            heartbeat=snap.last_packet_monotonic,
            streaming_heartbeat=self.require_imu_heartbeat,
            cartesian_velocity_m_s=velocity,
            target_rotation=target_rot,
            gripper_absolute=self.gripper_state,
            home=home,
            exit=exit_triggered,
            estop=estop,
            episode_events=events,
        )

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
