from __future__ import annotations

import importlib.util
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .errors import BackendUnavailable, SafetyError
from .safety import pose_reached


def _prefer_environment_packages() -> None:
    """Put the active env's site-packages ahead of ~/.local .pth injections."""
    try:
        import sysconfig
        purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    except Exception:
        return
    if not purelib.is_dir():
        return
    sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != purelib]
    sys.path.insert(0, str(purelib))


def _hardware_feedback_time(source) -> float | None:
    """只接受单调时钟语义明确的时间戳，避免把 Unix 时间误当 monotonic。"""
    for name in ("last_rx_monotonic", "last_update_monotonic", "updated_monotonic",
                 "rx_monotonic", "timestamp_monotonic"):
        value = getattr(source, name, None)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    return None


def _load_mit_controller():
    """兼容 rebot_control 包，以及从 rebot 包直接导出的新版 SDK。"""
    try:
        from rebot_control import ReBotRSMITController
        return ReBotRSMITController
    except ImportError:
        try:
            from rebot import ReBotRSMITController
            return ReBotRSMITController
        except ImportError:
            try:
                from rebot.robot import ReBotRSMITController
                return ReBotRSMITController
            except ImportError as final_error:
                raise BackendUnavailable(
                    "未找到受支持的 MIT 控制器；请配置 rebot_sdk_path，"
                    "支持 import rebot_control、rebot 或 rebot.robot") from final_error


@dataclass
class ArmState:
    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    temperatures: list[float | None]
    timestamp: float | None


class ArmBackend(ABC):
    name = "base"

    def __init__(self) -> None:
        self.enabled = False
        self.connected = False
        self.stopped = False

    @abstractmethod
    def connect(self, *, enable: bool = True) -> ArmState:
        raise NotImplementedError

    @abstractmethod
    def read_state(self) -> ArmState:
        raise NotImplementedError

    @abstractmethod
    def command(self, q: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def safe_stop(self) -> None:
        """唯一正常停止路径：平滑回零、等待、再失能。"""

    def close(self) -> None:
        if self.connected and self.enabled and not self.stopped:
            self.safe_stop()

    def abort_before_enable(self) -> None:
        self.connected = False


class FakeBackend(ArmBackend):
    name = "fake"

    def __init__(self, initial: np.ndarray):
        super().__init__()
        self.initial = np.asarray(initial, dtype=float).copy()
        self.position = self.initial.copy()
        self.target = self.initial.copy()
        self.commands: list[np.ndarray] = []
        self.safe_stop_calls = 0

    def connect(self, *, enable: bool = True) -> ArmState:
        self.connected = True
        self.enabled = enable
        return self.read_state()

    def read_state(self) -> ArmState:
        return ArmState(self.position.copy(), np.zeros(7), np.zeros(7), [25.0] * 7,
                        time.monotonic())

    def command(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        if not self.enabled or self.stopped or q.shape != (7,) or not np.isfinite(q).all():
            raise SafetyError("仿真后端未使能、已停止或命令无效")
        self.target = q.copy()
        self.position = self.target.copy()
        self.commands.append(self.target.copy())

    def safe_stop(self) -> None:
        self.safe_stop_calls += 1
        self.target = np.zeros(7)
        self.position = np.zeros(7)
        self.enabled = False
        self.stopped = True


class PosVelBackend(ArmBackend):
    """官方 reBotArm_control_py 后端；一个 CAN master 包含第七路夹爪。"""

    name = "pos_vel"

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        _prefer_environment_packages()
        sdk_path = settings.section("arm").get("sdk_path")
        if sdk_path:
            path = str(settings.resolve(sdk_path))
            if not Path(path).is_dir():
                raise BackendUnavailable(f"sdk_path 不存在：{path}")
            sys.path.insert(0, path)
        try:
            from reBotArm_control_py.actuator import RebotArm
        except ImportError as exc:
            if "pinocchio" in str(exc).lower():
                raise BackendUnavailable(
                    "reBotArm_control_py 导入失败：当前环境的 Pinocchio 与 Python/"
                    "二进制依赖不兼容。请在 conda 环境安装兼容 pinocchio，"
                    "并用 python -s 或 PYTHONNOUSERSITE=1 避免加载 ~/.local 版本") from exc
            raise BackendUnavailable(
                "未找到 reBotArm_control_py；请在 config/default.yaml 设置 sdk_path") from exc
        self.arm = RebotArm(str(settings.hardware_yaml))
        self._target = np.zeros(7)
        self._lock = threading.RLock()
        self._fallback_warned = False

    def connect(self, *, enable: bool = True) -> ArmState:
        self.arm.connect()
        self.connected = True
        may_be_enabled = False
        try:
            state = self.read_state()
            self._target = state.position.copy()
            if enable:
                if not self.arm.arm.mode_pos_vel():
                    raise SafetyError("六轴 POS_VEL 模式切换未全部确认")
                if self.arm.has_gripper and not self.arm.gripper.mode_pos_vel():
                    raise SafetyError("夹爪 POS_VEL 模式切换未全部确认")
                # enable_all 可能只成功一部分；从第一次调用起就禁止直接断开。
                may_be_enabled = True
                self.arm.enable_all()
                self.enabled = True
                self.arm.start_control_loop(self._control_callback)
                print("[arm] reBotArm_control_py POS_VEL 后端；已启用主机轨迹限速")
            return state
        except BaseException as exc:
            if may_be_enabled:
                self.connected = True
                self.enabled = True
                self.stopped = False
                raise SafetyError(
                    "POS_VEL 在使能阶段失败；已禁止直接 disconnect。请保持机械臂受支撑，"
                    "由上层执行受控回零；若控制线程未启动，请使用官方低速恢复工具") from exc
            try:
                self.arm.disconnect()
            except Exception as disconnect_error:
                print(f"[arm] 未使能连接清理失败：{disconnect_error}；请人工检查 CAN")
            finally:
                self.connected = False
                self.enabled = False
                self.stopped = False
            raise

    def _control_callback(self, arm, dt: float) -> None:
        with self._lock:
            target = self._target.copy()
        arm.arm.send_pos_vel(target[:6])
        if arm.has_gripper:
            arm.gripper.send_pos_vel(target[6:])

    def read_state(self) -> ArmState:
        if not self.connected:
            raise SafetyError("POS_VEL 后端未连接")
        if self.enabled and not self.arm.control_loop_active:
            raise SafetyError("官方 POS_VEL 控制线程已退出；禁止继续发送目标")
        position, velocity, torque = self.arm.get_state()
        position = np.asarray(position, dtype=float).copy()
        velocity = np.asarray(velocity, dtype=float).copy()
        torque = np.asarray(torque, dtype=float).copy()
        # 官方 get_state 对缺失反馈会填 0；不能把 0 当反馈。
        motors = getattr(self.arm, "_motor_map", {})
        temperatures = [None] * len(self.arm.joint_names)
        missing = []
        feedback_times = []
        for index, name in enumerate(self.arm.joint_names):
            motor = motors.get(name)
            state = None if motor is None else motor.get_state()
            if state is not None:
                position[index] = float(state.pos)
                velocity[index] = float(state.vel)
                torque[index] = float(state.torq)
                temperatures[index] = next(
                    (float(getattr(state, attr)) for attr in
                     ("t_mos", "mos_temperature", "temperature", "temp")
                     if getattr(state, attr, None) is not None), None)
                feedback_times.append(_hardware_feedback_time(motor) or
                                      _hardware_feedback_time(state))
                continue

            # 部分 RS 固件不填充 feedback 缓存，但 mechPos 参数读取可用。
            missing.append(name)
            get_param = getattr(motor, "robstride_get_param_f32", None)
            if get_param is None:
                raise SafetyError(f"电机反馈缺失：{name}，且没有位置参数读取兜底")
            try:
                position[index] = float(get_param(0x7019, timeout_ms=200))
                # 同步参数响应本身构成一次新鲜反馈。
                feedback_times.append(time.monotonic())
            except Exception as exc:
                raise SafetyError(
                    f"电机反馈缺失：{name}；mechPos(0x7019) 读取也失败：{exc}") from exc
            velocity[index] = 0.0
            torque[index] = np.nan
        if missing and not getattr(self, "_fallback_warned", False):
            print("[arm] POS_VEL feedback 缓存为空，已改用 mechPos(0x7019) 读取位置；"
                  f"这些电机的速度/力矩/温度不可用：{', '.join(missing)}")
            self._fallback_warned = True
        if (position.shape != (7,) or not np.isfinite(position).all()
                or velocity.shape != (7,) or not np.isfinite(velocity).all()
                or torque.shape != (7,)):
            raise SafetyError("POS_VEL 反馈维度或数值无效")
        timestamp = (min(feedback_times) if len(feedback_times) == 7
                     and all(value is not None for value in feedback_times) else None)
        return ArmState(position, velocity, torque, temperatures, timestamp)

    def command(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        if not self.enabled or self.stopped or q.shape != (7,) or not np.isfinite(q).all():
            raise SafetyError("POS_VEL 命令无效")
        with self._lock:
            self._target = q.copy()

    def safe_stop(self) -> None:
        if not self.connected or not self.enabled or self.stopped:
            return
        safety = self.settings.section("safety")
        timeout = float(safety.get("home_timeout_s", 15.0))
        arm_tol = float(safety.get("home_tolerance_rad", 0.04))
        grip_tol = float(safety.get("gripper_home_tolerance_rad", 0.15))
        start = self.read_state().position
        # smoothstep peak speed is 1.5 * delta / duration; keep it within the
        # same per-joint limits used by the teleoperation trajectory.
        peak_speeds = 1.5 * np.abs(start) / self.settings.max_velocity
        duration = max(3.0, float(np.max(peak_speeds)))
        print(f"[安全停止] POS_VEL 平滑回零（预计 {duration:.1f}s）...", flush=True)
        began = time.monotonic()
        # smoothstep 轨迹，起止速度均为零。
        while time.monotonic() - began < duration:
            ratio = (time.monotonic() - began) / duration
            blend = ratio * ratio * (3.0 - 2.0 * ratio)
            self.command(start * (1.0 - blend))
            time.sleep(0.01)
        self.command(np.zeros(7))
        wait_budget = max(5.0, timeout, 0.5 * duration)
        deadline = time.monotonic() + wait_budget
        while time.monotonic() < deadline:
            pos = self.read_state().position
            if pose_reached(pos, np.zeros(7), arm_tolerance_rad=arm_tol,
                            gripper_tolerance_rad=grip_tol):
                self.arm.stop_control_loop()
                self.arm.disconnect()  # 只有到零并等待后才允许其内部失能。
                self.enabled = False
                self.connected = False
                self.stopped = True
                print("[安全停止] POS_VEL 已回零并失能", flush=True)
                return
            time.sleep(0.03)
        raise SafetyError(
            "安全回零未在超时内收敛；电机仍保持使能。请扶稳机械臂、排除阻挡，"
            "不要直接断电或调用 disable_all/estop。")

    def abort_before_enable(self) -> None:
        try:
            if self.connected and not self.enabled:
                self.arm.disconnect()
        finally:
            self.connected = False
            self.enabled = False


class RebotMITBackend(ArmBackend):
    """官方推荐 rebot_control MIT 后端，内部 200 Hz 平滑及温度保护。"""

    name = "mit"

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self._prefer_cached_position = False
        _prefer_environment_packages()
        sdk_path = settings.section("arm").get("rebot_sdk_path")
        if sdk_path:
            path = str(settings.resolve(sdk_path))
            if not Path(path).is_dir():
                raise BackendUnavailable(f"rebot_sdk_path 不存在：{path}")
            sys.path.insert(0, path)
        controller_class = _load_mit_controller()
        self.controller = controller_class()

    @staticmethod
    def _array(value, *, degrees: bool) -> np.ndarray:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.shape != (7,) or not np.isfinite(array).all():
            raise SafetyError("rebot_control 必须返回七路有限反馈")
        return np.deg2rad(array) if degrees else array

    def _cached_motor_states(self) -> list:
        motors = getattr(self.controller, "motors", None)
        if not motors:
            return []
        states = []
        for motor in list(motors)[:7]:
            try:
                states.append(motor.get_state())
            except Exception:
                states.append(None)
        return states

    @staticmethod
    def _cached_channel(states: list, field: str, *, default: float) -> np.ndarray:
        values = np.full(7, default, dtype=float)
        for index, state in enumerate(states):
            raw = getattr(state, field, None) if state is not None else None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values[index] = value
        return values

    def _sdk_position(self) -> np.ndarray:
        """SDK 的精确位置读取：逐路 CAN 参数读取，可靠但一帧可能超过 250 ms。"""
        if hasattr(self.controller, "get_joint_angles"):
            return self._array(self.controller.get_joint_angles(), degrees=True)
        if hasattr(self.controller, "read_joint_angles"):
            return self._array(self.controller.read_joint_angles(), degrees=True)
        if hasattr(self.controller, "get_positions"):
            return self._array(self.controller.get_positions(), degrees=False)
        raise SafetyError("当前 MIT SDK 缺少受支持的七路位置反馈接口")

    def _calibrate_position_source(self) -> None:
        """启动时用 SDK 精确读数校验高速缓存；不可信则回退，绝不盲用缓存。"""
        cached = self._cached_channel(self._cached_motor_states(), "pos", default=np.nan)
        if not np.isfinite(cached).all():
            self._prefer_cached_position = False
            print("[arm] motorbridge 位置缓存不可用，回退 SDK 参数读取（控制周期可能较慢）",
                  flush=True)
            return
        reference = self._sdk_position()
        error = float(np.max(np.abs(cached - reference)))
        if error > np.deg2rad(2.0):
            self._prefer_cached_position = False
            print(f"[arm] 位置缓存与 SDK 读数偏差 {np.rad2deg(error):.2f}°，回退 SDK 参数读取",
                  flush=True)
            return
        self._prefer_cached_position = True
        print(f"[arm] 使用 motorbridge 高速位置缓存（启动偏差 {np.rad2deg(error):.2f}°）",
              flush=True)

    def _wait_for_motorbridge_cache(self, timeout_s: float = 2.0) -> None:
        """等待 SDK 遥测线程刷新第一帧夹爪力矩缓存（避免启动阶段误读未校准缓存）。"""
        if hasattr(self.controller, "get_torques"):
            return
        torques = self._cached_channel(self._cached_motor_states(), "torq", default=np.nan)
        if np.isfinite(torques[6]):
            return
        print(f"[arm] 等待 motorbridge 夹爪力矩反馈（最多 {timeout_s:.1f} s）...", flush=True)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.05)
            torques = self._cached_channel(self._cached_motor_states(), "torq", default=np.nan)
            if np.isfinite(torques[6]):
                return

    def connect(self, *, enable: bool = True) -> ArmState:
        if enable:
            # 项目层负责 Esc/Ctrl+C 与安全停止，避免 SDK 重复安装信号处理。
            try:
                try:
                    self.controller.start(enable_esc=False, install_signal_handlers=False)
                except TypeError:
                    self.controller.start(enable_esc=False)
            except BaseException:
                # start() 可能已在使能后失败；先标记状态，再执行项目侧受控回零。
                self.connected = True
                self.enabled = True
                try:
                    self.safe_stop()
                except Exception as stop_error:
                    print(f"[arm] MIT 启动失败后的安全停止也失败：{stop_error}；"
                          "请人工扶稳机械臂，禁止直接失能")
                raise
            self.connected = True
            self.enabled = True
            if hasattr(self.controller, "set_max_speeds"):
                self.controller.set_max_speeds(np.rad2deg(self.settings.max_velocity).tolist())
            print("[arm] rebot_control MIT 后端（内部平滑，七路同一 CAN master）", flush=True)
            # 先等待缓存填充并完成与 SDK 精确读数的比对，再返回首次 ArmState。
            self._wait_for_motorbridge_cache(timeout_s=2.0)
            self._calibrate_position_source()
            return self.read_state()

        raise SafetyError(
            "当前 MIT SDK 的读取流程必须先使能电机，不能提供安全只读快照；"
            "已拒绝连接，避免读取后在弯曲姿态直接失能。请改用 --backend pos_vel")

    def read_state(self) -> ArmState:
        error = getattr(self.controller, "last_error", None)
        if error is not None:
            raise SafetyError(f"MIT 控制线程异常：{error}")
        if getattr(self, "enabled", False) and not getattr(self.controller, "started", True):
            raise SafetyError("MIT 控制线程已停止；禁止继续发送目标")
        # LAN-GER/rebot_control 的 read_joint_angles() 每帧做 7 次 CAN 参数读取，
        # 实测会让 30 Hz 主循环频繁超过 250 ms 而不断暂停。位置/速度/力矩优先
        # 使用 motorbridge 反馈缓存；启动时已与 SDK 精确读数比对，不可信则回退。
        cached = self._cached_motor_states()
        cached_position = self._cached_channel(cached, "pos", default=np.nan)
        if (getattr(self, "_prefer_cached_position", False)
                and np.isfinite(cached_position).all()):
            position = cached_position
        else:
            position = self._sdk_position()
        if hasattr(self.controller, "get_velocities"):
            velocity = self._array(self.controller.get_velocities(), degrees=False)
        else:
            velocity = self._cached_channel(cached, "vel", default=0.0)
        if hasattr(self.controller, "get_torques"):
            raw_torque = np.asarray(self.controller.get_torques(), dtype=float).reshape(-1)
            if raw_torque.shape != (7,):
                raise SafetyError("rebot_control 的力矩反馈必须为七路")
            # 部分固件不提供力矩；保留 NaN，由夹爪保护视为不可用。
            torque = np.where(np.isfinite(raw_torque), raw_torque, np.nan)
        else:
            torque = self._cached_channel(cached, "torq", default=np.nan)
        if hasattr(self.controller, "get_temperatures"):
            raw_temperatures = self.controller.get_temperatures()
        elif hasattr(self.controller, "get_temperature"):
            raw_temperatures = self.controller.get_temperature()
        elif hasattr(self.controller, "last_temperatures"):
            # LAN-GER/rebot_control 当前由遥测线程维护该列表。
            raw_temperatures = self.controller.last_temperatures
        else:
            raw_temperatures = [None] * 7
        temperatures = list(raw_temperatures)[:7] if raw_temperatures is not None else []
        temperatures += [None] * (7 - len(temperatures))
        return ArmState(position, velocity, torque, temperatures,
                        _hardware_feedback_time(self.controller))

    def command(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        if not self.enabled or self.stopped or q.shape != (7,) or not np.isfinite(q).all():
            raise SafetyError("MIT 命令无效")
        self.controller.set_joint_angles(np.rad2deg(q).tolist())

    def safe_stop(self) -> None:
        if not self.connected or not self.enabled or self.stopped:
            return
        safety = self.settings.section("safety")
        timeout = float(safety.get("home_timeout_s", 15.0))
        arm_tol = float(safety.get("home_tolerance_rad", 0.04))
        grip_tol = float(safety.get("gripper_home_tolerance_rad", 0.15))
        start = self.read_state().position.copy()
        peak_speeds = 1.5 * np.abs(start) / self.settings.max_velocity
        duration = max(3.0, float(np.max(peak_speeds)))
        print(f"[安全停止] MIT 平滑回零（预计 {duration:.1f}s）...", flush=True)
        began = time.monotonic()
        try:
            # 不把 SDK stop() 当作实际到位证明；项目先独立生成回零轨迹并检查反馈。
            while time.monotonic() - began < duration:
                ratio = (time.monotonic() - began) / duration
                blend = ratio * ratio * (3.0 - 2.0 * ratio)
                self.command(start * (1.0 - blend))
                time.sleep(0.01)
            self.command(np.zeros(7))
            wait_budget = max(5.0, timeout, 0.5 * duration)
            deadline = time.monotonic() + wait_budget
            while time.monotonic() < deadline:
                pos = self.read_state().position
                if pose_reached(pos, np.zeros(7), arm_tolerance_rad=arm_tol,
                                gripper_tolerance_rad=grip_tol):
                    break
                time.sleep(0.03)
            else:
                pos = self.read_state().position
                raise SafetyError(
                    "MIT 实际反馈未在超时内回到零位；未调用 SDK 失能。"
                    f"当前误差(deg)={np.rad2deg(np.abs(pos)).round(2).tolist()}。"
                    "请扶稳机械臂并排除阻挡")
        except KeyboardInterrupt:
            self._abort_mit_now()
            return
        try:
            # 项目已独立生成回零轨迹并确认实际到位；SDK 不再二次回零，
            # 只关闭控制线程并失能，缩短停止窗口。
            self.controller.stop(return_to_zero=False, wait=True)
        except KeyboardInterrupt:
            self._abort_mit_now()
            return
        except Exception as exc:
            raise SafetyError(
                "机械臂已实际回到零位，但 MIT SDK 停止失败；请检查 SDK/CAN 状态") from exc
        self.enabled = False
        self.connected = False
        self.stopped = True
        print("[安全停止] MIT 已回零并失能", flush=True)

    def _abort_mit_now(self) -> None:
        # 回零/停止期间再次 Ctrl+C 是明确的立即中止请求；必须真正失能，
        # 不能让程序带着电机使能直接退出（与官方 SDK 的二次中断语义一致）。
        print("\n[紧急中止] 停止被再次 Ctrl+C 中断：立即失能，请扶稳机械臂", flush=True)
        self.controller.stop(return_to_zero=False, wait=True)
        self.enabled = False
        self.connected = False
        self.stopped = True

    def abort_before_enable(self) -> None:
        if not self.connected:
            return
        if self.enabled and not self.stopped:
            self.safe_stop()
            return
        self.controller.stop(return_to_zero=False, wait=True)
        self.connected = False
        self.enabled = False


def create_backend(settings: Settings, requested: str | None = None,
                   *, initial: np.ndarray | None = None) -> ArmBackend:
    name = requested or settings.section("arm").get("backend", "auto")
    if name == "fake":
        return FakeBackend(settings.home if initial is None else initial)
    if name == "pos_vel":
        return PosVelBackend(settings)
    if name == "mit":
        return RebotMITBackend(settings)
    if name != "auto":
        raise BackendUnavailable(f"未知后端：{name}")
    sdk_path = settings.section("arm").get("rebot_sdk_path")
    if sdk_path:
        sys.path.insert(0, str(settings.resolve(sdk_path)))
    if (importlib.util.find_spec("rebot_control") is not None
            or importlib.util.find_spec("rebot") is not None):
        try:
            return RebotMITBackend(settings)
        except BackendUnavailable as exc:
            print(f"[arm] MIT 控制器不可用，回退 POS_VEL：{exc}")
    else:
        print("[arm] 未找到 rebot_control/rebot MIT SDK，回退 POS_VEL")
    return PosVelBackend(settings)
