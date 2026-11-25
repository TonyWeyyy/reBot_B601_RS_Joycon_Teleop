from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .errors import SafetyError


@dataclass
class GateStatus:
    allowed: bool
    reason: str = ""


class SafetyGate:
    """按住使能 + 心跳看门狗；断线后必须松开再重新按住才能恢复。

    streaming_heartbeat=True 用于 Joy-Con IMU 等连续数据流，要求心跳年龄 <= watchdog_s；
    streaming_heartbeat=False 用于键盘等事件驱动输入，只要监听线程存活（heartbeat is not None）
    即视为在线，避免按住 Space 期间无新按键事件导致 0.2s 误触发超时。
    """

    def __init__(self, watchdog_s: float = 0.25, startup_guard_s: float = 1.0,
                 require_deadman: bool = True, *, streaming_heartbeat: bool = True):
        self.watchdog_s = watchdog_s
        self.startup_guard_s = startup_guard_s
        self.require_deadman = require_deadman
        self.streaming_heartbeat = streaming_heartbeat
        self.started = time.monotonic()
        self.fault_latched = False

    def update(self, deadman: bool, heartbeat_s: float | None,
               *, now: float | None = None,
               streaming: bool | None = None) -> GateStatus:
        now = time.monotonic() if now is None else now
        if now - self.started < self.startup_guard_s:
            return GateStatus(False, "启动保护")
        check_fresh = self.streaming_heartbeat if streaming is None else bool(streaming)
        if check_fresh:
            alive = (heartbeat_s is not None
                     and math.isfinite(float(heartbeat_s))
                     and -1e-3 <= now - float(heartbeat_s) <= self.watchdog_s)
        else:
            alive = heartbeat_s is not None and math.isfinite(float(heartbeat_s))
        if not alive:
            self.fault_latched = True
            return GateStatus(False, "输入心跳超时" if check_fresh else "输入监听线程异常")
        if self.fault_latched:
            if not deadman:
                self.fault_latched = False
            return GateStatus(False, "看门狗已锁存：松开后重新按住")
        if self.require_deadman and not deadman:
            return GateStatus(False, "未按住使能")
        return GateStatus(True)


class JointLimiter:
    def __init__(self, model_limits: np.ndarray, gripper_limits: np.ndarray,
                 scale: float = 0.7):
        model_limits = np.asarray(model_limits, dtype=float)
        if model_limits.shape != (2, 6):
            raise ValueError("模型限位形状必须为 2×6")
        self.hard = np.concatenate([model_limits, np.asarray(gripper_limits)[:, None]], axis=1)
        # 官方教程的 J1 为 ±150°，比当前 URDF 的 ±2.8 rad 更严格。
        self.hard[:, 0] = np.deg2rad([-150.0, 150.0])
        self.soft = self.hard * float(scale)
        if not np.isfinite(self.soft).all() or np.any(self.soft[0] >= self.soft[1]):
            raise ValueError("软件关节限位无效")

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.soft[0], self.soft[1])

    def validate(self, q: np.ndarray, *, soft: bool = True) -> None:
        q = np.asarray(q, dtype=float)
        limits = self.soft if soft else self.hard
        if q.shape != (7,) or not np.isfinite(q).all():
            raise SafetyError("关节反馈必须是七个有限数值")
        # RS 全零折叠位压在 J2/J3 限位下边界，夹爪机械零位也有编码器偏差；
        # 七路统一允许 1° 反馈容差，避免回零后的微小负偏无法再次启动。
        # 这只放宽“当前反馈”判定，命令目标仍在 rate.step()/clamp() 中严格限幅。
        tolerance = np.full(7, np.deg2rad(1.0))
        if np.any(q < limits[0] - tolerance) or np.any(q > limits[1] + tolerance):
            raise SafetyError(
                "当前姿态超出软件安全限位；请用官方低速恢复工具向安全区内移动，"
                "禁止直接 clamp 当前反馈造成跳变。\n"
                f"当前={np.rad2deg(q).round(2).tolist()}\n"
                f"软限位={np.rad2deg(limits).round(2).tolist()}")


class JointRateLimiter:
    """对目标轨迹限速、限加速度；复位时从实际反馈而不是旧目标继续。"""

    def __init__(self, max_velocity: np.ndarray, max_acceleration: np.ndarray,
                 joint_limiter: JointLimiter):
        self.max_velocity = np.asarray(max_velocity, dtype=float)
        self.max_acceleration = np.asarray(max_acceleration, dtype=float)
        self.joint_limiter = joint_limiter
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(7)

    def reset(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(7)

    def step(self, goal: np.ndarray, dt: float) -> np.ndarray:
        if self.position is None:
            raise RuntimeError("速度限制器尚未从实际反馈复位")
        dt = float(np.clip(dt, 1e-4, 0.1))
        goal = self.joint_limiter.clamp(goal)
        error = goal - self.position
        # 位置轨迹必须提前制动；只限制加速度会在小目标附近穿过目标。
        braking_velocity = np.sqrt(2.0 * self.max_acceleration * np.abs(error))
        v_cap = np.minimum(self.max_velocity, braking_velocity)

        # 六轴机械臂采用等比例协调速度/加速度缩放，确保直线平移（如纯 +X）不因单轴饱和而走偏；
        # 夹爪（第 7 路）保持独立，不拖慢机械臂本体。
        desired_velocity = np.zeros(7, dtype=float)
        arm_err = error[:6]
        arm_mask = np.abs(arm_err) > 1e-9
        if np.any(arm_mask):
            alpha = float(np.min(v_cap[:6][arm_mask] / np.abs(arm_err[arm_mask])))
            desired_velocity[:6] = arm_err * alpha
        desired_velocity[6] = math.copysign(v_cap[6], error[6]) if abs(error[6]) > 1e-9 else 0.0

        dv = desired_velocity - self.velocity
        dv_limit = self.max_acceleration * dt
        arm_ratio = float(np.max(np.abs(dv[:6]) / np.maximum(dv_limit[:6], 1e-12)))
        if arm_ratio > 1.0:
            dv[:6] /= arm_ratio
        dv[6] = float(np.clip(dv[6], -dv_limit[6], dv_limit[6]))

        self.velocity = np.clip(self.velocity + dv, -self.max_velocity, self.max_velocity)
        candidate = self.position + self.velocity * dt
        crossed = ((error > 0.0) & (candidate > goal)) | ((error < 0.0) & (candidate < goal))
        candidate = np.where(crossed, goal, candidate)
        self.velocity = np.where(crossed, 0.0, self.velocity)
        clipped = self.joint_limiter.clamp(candidate)
        self.velocity = np.where(candidate == clipped, self.velocity, 0.0)
        self.position = clipped
        return self.position.copy()


def _parse_workspace_origin(origin) -> np.ndarray:
    arr = np.asarray(origin, dtype=float).reshape(-1)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise SafetyError("工作空间原点参数无效")
    return arr


def workspace_contains(point: np.ndarray, *, radius: float = 0.58,
                       min_z: float = 0.04, max_z: float = 0.65,
                       min_reach: float = 0.10,
                       origin=(0.0, 0.0, 0.145),
                       tol: float = 1e-6) -> bool:
    pt = np.asarray(point, dtype=float).reshape(-1)
    if pt.shape != (3,) or not np.isfinite(pt).all():
        return False
    orig = _parse_workspace_origin(origin)
    reach = float(np.linalg.norm(pt - orig))
    return (float(min_reach) - tol <= reach <= float(radius) + tol
            and float(min_z) - tol <= float(pt[2]) <= min(float(max_z), float(orig[2]) + float(radius)) + tol)


def project_workspace(pose: np.ndarray, *, radius: float = 0.58,
                      min_z: float = 0.04, max_z: float = 0.65,
                      min_reach: float = 0.10,
                      origin=(0.0, 0.0, 0.145)) -> np.ndarray:
    """把笛卡尔目标投影到以肩部中心 origin 为球心的安全球壳，姿态保持不变。

    先限制 z，再只缩放相对 origin 的水平分量来满足半径和最小臂展，避免径向缩放把
    z 拉回 min_z 以下（或把点推出球面）。
    """
    result = np.asarray(pose, dtype=float).copy()
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise SafetyError("笛卡尔目标无效")
    orig = _parse_workspace_origin(origin)
    radius = float(radius)
    min_z = float(min_z)
    max_z = float(max_z)
    min_reach = float(min_reach)
    max_sphere_z = float(orig[2]) + radius
    if (not all(math.isfinite(value) for value in (radius, min_z, max_z, min_reach))
            or radius <= 0.0 or min_reach < 0.0 or min_reach > radius
            or min_z < 0.0 or min_z > max_sphere_z or min_z > max_z):
        raise SafetyError("工作空间参数无效")
    point = result[:3, 3]
    # 球面顶点也是 z 的物理上限；底端不得低于 min_z 或球底。
    z_low = max(min_z, float(orig[2]) - radius)
    z_high = min(max_z, max_sphere_z)
    if z_low > z_high:
        raise SafetyError("工作空间 z 范围与球体不相交")
    point[2] = np.clip(point[2], z_low, z_high)
    rel = point - orig

    def horizontal_norm() -> float:
        return float(math.hypot(rel[0], rel[1]))

    max_horizontal = math.sqrt(max(0.0, radius * radius - rel[2] * rel[2]))
    horizontal = horizontal_norm()
    if horizontal > max_horizontal:
        if max_horizontal <= 1e-12:
            rel[:2] = 0.0
        else:
            rel[:2] *= max_horizontal / horizontal

    reach = float(np.linalg.norm(rel))
    if reach < min_reach:
        required = math.sqrt(max(0.0, min_reach * min_reach - rel[2] * rel[2]))
        horizontal = horizontal_norm()
        if horizontal > 1e-12:
            rel[:2] *= required / horizontal
        else:
            rel[:2] = [required, 0.0]
    result[:3, 3] = orig + rel
    return result


def pose_reached(actual: np.ndarray, target: np.ndarray, *,
                 arm_tolerance_rad: float = 0.04,
                 gripper_tolerance_rad: float = 0.15) -> bool:
    """分轴判定六轴与夹爪是否到位，适应 MIT PD 重力静差与 345° 夹爪行程。"""
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    if actual.shape != (7,) or target.shape != (7,):
        return False
    arm_err = float(np.max(np.abs(actual[:6] - target[:6])))
    grip_err = float(abs(actual[6] - target[6]))
    return arm_err < float(arm_tolerance_rad) and grip_err < float(gripper_tolerance_rad)


class GripperTorqueProtection:
    """过力矩后锁存当前位置；只允许明确的反向卸载，不是硬件力矩上限。"""

    def __init__(self, threshold_nm: float = 1.0):
        self.threshold_nm = float(threshold_nm)
        self.hold: float | None = None
        self.blocked_direction = 0.0

    def reset(self) -> None:
        self.hold = None
        self.blocked_direction = 0.0

    def apply(self, requested: float, position: float, measured: float | None) -> float:
        if measured is None:
            return requested
        if not all(math.isfinite(v) for v in (requested, position, measured)):
            raise SafetyError("夹爪目标、位置或力矩反馈非有限值")
        overloaded = abs(measured) > self.threshold_nm
        if self.hold is not None:
            reversing = (requested - self.hold) * self.blocked_direction < -1e-6
            if reversing:
                if not overloaded:
                    self.reset()
                return requested
        if overloaded:
            if self.hold is None:
                error = requested - position
                self.blocked_direction = float(math.copysign(
                    1.0, error if abs(error) > 1e-6 else measured))
                self.hold = position
            elif (position - self.hold) * self.blocked_direction < 0:
                self.hold = position
        return requested if self.hold is None else float(self.hold)


def check_temperatures(temperatures: list[float | None], warn_c: float, trip_c: float) -> None:
    valid = [float(t) for t in temperatures if t is not None and math.isfinite(float(t))]
    if not valid:
        return
    hottest = max(valid)
    if hottest >= trip_c:
        raise SafetyError(f"MOS 温度 {hottest:.1f}°C ≥ {trip_c:.1f}°C，执行安全回零")
    if hottest >= warn_c:
        print(f"[安全警告] MOS 最高温度 {hottest:.1f}°C，请停止并冷却")


def validate_feedback(state, safety: dict, *, now: float, started: float) -> None:
    """拒绝陈旧反馈，并明确处理安全反馈不可用，而不是静默关闭保护。"""
    timeout = float(safety.get("feedback_timeout_s", 0.5))
    timestamp = getattr(state, "timestamp", None)
    if timestamp is None:
        if bool(safety.get("require_feedback_timestamp", False)):
            raise SafetyError("后端没有硬件反馈时间戳；严格反馈模式禁止继续运动")
    elif not math.isfinite(float(timestamp)) or not -1e-3 <= now - float(timestamp) <= timeout:
        raise SafetyError(f"硬件反馈已陈旧或时间无效：age={now - float(timestamp):.3f}s")
    if bool(safety.get("require_gripper_torque_feedback", True)):
        if not np.isfinite(state.torque[6]):
            raise SafetyError("夹爪力矩反馈不可用；严格模式禁止运行。请使用 MIT 或修复反馈")
    if bool(safety.get("require_temperature_feedback", True)):
        if now - started > float(safety.get("feedback_grace_s", 5.0)):
            if not any(value is not None and math.isfinite(float(value))
                       for value in state.temperatures):
                raise SafetyError("温度反馈在宽限期后仍不可用；严格模式禁止继续运行")
