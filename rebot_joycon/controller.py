from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from .backends import ArmBackend, ArmState
from .config import Settings
from .errors import SafeStopRequested
from .inputs import InputSample
from .kinematics import RSKinematics, acceptable_ik_result, make_pose
from .safety import (GripperTorqueProtection, JointLimiter, JointRateLimiter,
                     SafetyGate, check_temperatures, pose_reached,
                     project_workspace, validate_feedback, workspace_contains)


@dataclass
class ControlFrame:
    state: ArmState
    target: np.ndarray
    allowed: bool
    reason: str
    episode_events: list[int]


def limit_attitude_step(current_rot: np.ndarray, target_rot: np.ndarray,
                        max_rate: float, dt: float) -> np.ndarray:
    """在 SO(3) 上限制单步姿态角速度，防止 IMU 异常跳变导致腕部猛转。"""
    if max_rate <= 0.0:
        return target_rot.copy()
    rotation = target_rot @ current_rot.T
    delta = Rotation.from_matrix(rotation).as_rotvec()
    angle = float(np.linalg.norm(delta))
    limit = float(max_rate) * float(dt)
    if angle <= limit or angle < 1e-9:
        return target_rot.copy()
    return Rotation.from_rotvec(delta * (limit / angle)).as_matrix() @ current_rot


def tether_cartesian_target(current_pose: np.ndarray, desired: np.ndarray,
                            *, max_lead_m: float = 0.04,
                            max_lead_rad: float = 0.15) -> np.ndarray:
    """限制笛卡尔目标相对当前实际轨迹的最大超前量，防止低速下甩开导致反向延迟或 IK 死锁。"""
    tethered = desired.copy()
    dp = tethered[:3, 3] - current_pose[:3, 3]
    dist = float(np.linalg.norm(dp))
    if dist > max_lead_m and dist > 1e-9:
        tethered[:3, 3] = current_pose[:3, 3] + dp * (max_lead_m / dist)
    r_diff = tethered[:3, :3] @ current_pose[:3, :3].T
    rotvec = Rotation.from_matrix(r_diff).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if angle > max_lead_rad and angle > 1e-9:
        tethered[:3, :3] = (Rotation.from_rotvec(rotvec * (max_lead_rad / angle)).as_matrix()
                            @ current_pose[:3, :3])
    return tethered


def step_cartesian_goal(cartesian_target: np.ndarray, sample: InputSample,
                        dt: float, *, max_imu_rate_rad_s: float) -> np.ndarray:
    """根据输入采样推进一步未裁剪的笛卡尔期望位姿（支持末端体坐标增量姿态）。"""
    if (sample.cartesian_velocity_m_s is not None
            or sample.rotation_delta_body is not None
            or sample.target_rotation is not None):
        desired = cartesian_target.copy()
        if sample.cartesian_velocity_m_s is not None:
            vel = np.asarray(sample.cartesian_velocity_m_s, dtype=float)
            if vel.shape == (3,) and np.isfinite(vel).all():
                desired[:3, 3] = desired[:3, 3] + vel * dt
        if sample.rotation_delta_body is not None:
            delta_body = np.asarray(sample.rotation_delta_body, dtype=float)
            if delta_body.shape == (3, 3) and np.isfinite(delta_body).all():
                candidate = desired[:3, :3] @ delta_body
                desired[:3, :3] = limit_attitude_step(
                    cartesian_target[:3, :3], candidate, max_imu_rate_rad_s, dt,
                )
        elif sample.target_rotation is not None:
            rot = np.asarray(sample.target_rotation, dtype=float)
            if rot.shape == (3, 3) and np.isfinite(rot).all():
                desired[:3, :3] = limit_attitude_step(
                    cartesian_target[:3, :3], rot, max_imu_rate_rad_s, dt,
                )
        return desired
    if sample.absolute_pose is not None:
        desired = make_pose(sample.absolute_pose)
        desired[:3, :3] = limit_attitude_step(
            cartesian_target[:3, :3], desired[:3, :3], max_imu_rate_rad_s, dt,
        )
        return desired
    direction = sample.cartesian_direction
    desired = cartesian_target.copy()
    desired[:3, 3] += direction[:3] * 0.12 * dt
    if np.any(direction[3:]):
        rotation = Rotation.from_rotvec(direction[3:] * 0.8 * dt).as_matrix()
        desired[:3, :3] = rotation @ desired[:3, :3]
    return desired


class TeleopController:
    def __init__(self, settings: Settings, backend: ArmBackend, input_source,
                 kinematics: RSKinematics, mode: str,
                 on_episode_event: Callable[[int], None] | None = None,
                 event_log=None):
        if mode not in ("qpos", "gpos"):
            raise ValueError("mode 必须为 qpos 或 gpos")
        self.settings = settings
        self.backend = backend
        self.input = input_source
        self.kin = kinematics
        self.mode = mode
        self.on_episode_event = on_episode_event
        self.event_log = event_log
        safety = settings.section("safety")
        self.joints = JointLimiter(kinematics.limits, settings.gripper_limits,
                                   float(safety.get("joint_limit_scale", 0.7)))
        self.joints.validate(settings.home)
        self.rate = JointRateLimiter(settings.max_velocity, settings.max_acceleration,
                                     self.joints)
        self.gate = SafetyGate(float(safety.get("input_watchdog_s", 0.25)),
                               float(safety.get("startup_guard_s", 1.0)),
                               bool(safety.get("require_deadman", True)))
        self.gripper_protection = GripperTorqueProtection(
            float(safety.get("gripper_torque_trip_nm", 1.0)))
        self.target = settings.home.copy()
        self.cartesian_target = np.eye(4)
        self._last_time: float | None = None
        self.home_active = False
        # 启动回 HOME 请求在首次真正按住使能前一直保留，不能被启动保护吞掉。
        self.home_pending = False
        self._wait_reason = ""
        self._wait_print = 0.0
        self._home_print = 0.0
        self._last_telemetry = 0.0
        self._telemetry_dt_max = 0.0
        self._ik_fail_count = 0
        self._ik_warn_time = 0.0
        self._was_allowed = False
        self.started = False
        self.feedback_started = time.monotonic()

    def _log(self, event: str, **fields) -> None:
        if self.event_log is None:
            return
        try:
            self.event_log.write(event, **fields)
        except Exception:
            pass

    def start(self) -> None:
        state = self.backend.connect(enable=True)
        try:
            self.joints.validate(state.position, soft=True)
            self.feedback_started = time.monotonic()
            validate_feedback(state, self.settings.section("safety"), now=self.feedback_started,
                              started=self.feedback_started)
            clamped = self.joints.clamp(state.position)
            self.target = clamped.copy()
            self.cartesian_target = self.kin.forward(clamped[:6])
            self.rate.reset(clamped)
            self._rebase_input(clamped)
        except Exception:
            self.backend.safe_stop()
            raise
        self._last_time = time.monotonic()
        self.gate.started = time.monotonic()
        self._last_telemetry = self.gate.started
        self.started = True
        if self.mode == "gpos" and float(np.max(np.abs(state.position[:6]))) < 0.05:
            print("[提示] 当前位于折叠零位奇异区；建议先按 +（或键盘 0 / 启动加 --home-on-start）"
                  "平滑展开到 HOME，再进行笛卡尔遥操作。", flush=True)

    def _begin_home(self) -> None:
        self.home_active = True
        self.cartesian_target = self.kin.forward(self.settings.home[:6])
        self._rebase_input(self.settings.home)
        self._home_print = time.monotonic()
        self._log("home_start", current_deg=np.rad2deg(self.target).round(2).tolist())
        print("[HOME] 开始平滑回 HOME；请保持按住使能，到位后会提示", flush=True)

    def _rebase_input(self, q: np.ndarray) -> None:
        sync_anchor = getattr(self.input, "sync_anchor", None)
        if sync_anchor is not None:
            sync_anchor(self.kin.forward(q[:6]))
            return
        rebase = getattr(self.input, "rebase", None)
        if rebase is None:
            return
        pose = self.kin.pose(q[:6])
        rebase(pose[:3], pose[3:])

    def _workspace_kwargs(self) -> dict:
        return self.settings.workspace_kwargs()

    def _workspace_ok(self, q: np.ndarray) -> bool:
        point = self.kin.forward(q)[:3, 3]
        # 允许 0.5 mm 容差以匹配 IK 收敛精度（1e-4 m），避免在边界贴边时误拒可行解。
        return workspace_contains(point, **self._workspace_kwargs(), tol=5e-4)

    def _limit_attitude_step(self, target_rot: np.ndarray, dt: float) -> np.ndarray:
        """Limit orientation changes in SO(3), not independent RPY axes."""
        max_rate = float(self.settings.section("joycon").get("imu_max_rate_rad_s", 2.5))
        return limit_attitude_step(self.cartesian_target[:3, :3], target_rot, max_rate, dt)

    def _tether_cartesian_target(self, current_pose: np.ndarray, desired: np.ndarray) -> np.ndarray:
        """限制笛卡尔目标相对当前实际轨迹的最大超前量，防止低速下甩开导致反向延迟或 IK 死锁。"""
        safety = self.settings.section("safety")
        return tether_cartesian_target(
            current_pose, desired,
            max_lead_m=float(safety.get("max_cartesian_lead_m", 0.04)),
            max_lead_rad=float(safety.get("max_cartesian_lead_rad", 0.15)),
        )

    def _goal_from_input(self, sample: InputSample, dt: float) -> np.ndarray:
        goal = self.target.copy()
        max_velocity = self.settings.max_velocity
        if self.mode == "qpos":
            goal[:6] += sample.joint_direction * max_velocity[:6] * dt
        else:
            current_pose = self.kin.forward(self.target[:6])
            max_rate = float(self.settings.section("joycon").get("imu_max_rate_rad_s", 2.5))
            desired = step_cartesian_goal(
                self.cartesian_target, sample, dt, max_imu_rate_rad_s=max_rate,
            )

            projected = project_workspace(desired, **self._workspace_kwargs())
            self.cartesian_target = self._tether_cartesian_target(current_pose, projected)

            soft_arm_limits = self.joints.soft[:, :6]
            result = self.kin.ik(self.target[:6], self.cartesian_target,
                                 joint_limits=soft_arm_limits)
            if acceptable_ik_result(result) and self._workspace_ok(result.q):
                goal[:6] = result.q
            else:
                # 若因手腕到达软限位导致 6D 全姿态无法满足，回退为位置优先求解，避免整臂卡死。
                fallback = self.kin.ik(
                    self.target[:6], self.cartesian_target,
                    joint_limits=soft_arm_limits,
                    position_weight=6.0, orientation_weight=0.25,
                )
                if (fallback.improved and fallback.position_error <= 0.015
                        and self._workspace_ok(fallback.q)):
                    goal[:6] = fallback.q
                    self.cartesian_target = self.kin.forward(fallback.q)
                else:
                    self._ik_fail_count += 1
                    now = time.monotonic()
                    if now - self._ik_warn_time >= 2.0:
                        print(f"[IK] 当前目标已触及软限位或奇异区"
                              f"（pos_err={result.position_error * 1000:.1f}mm, "
                              f"rot_err={np.rad2deg(result.orientation_error):.1f}°），已贴边保持",
                              flush=True)
                        self._ik_warn_time = now
                    # 不积累不可达目标。
                    self.cartesian_target = current_pose

        if sample.gripper_absolute is not None:
            travel_s = float(self.settings.section("joycon").get("gripper_travel_time_s", 1.0))
            span = abs(self.joints.soft[1, 6] - self.joints.soft[0, 6])
            step = span / max(travel_s, 0.1) * dt
            goal[6] += np.clip(sample.gripper_absolute - self.target[6], -step, step)
        else:
            goal[6] += sample.gripper_direction * max_velocity[6] * dt
        return self.joints.clamp(goal)

    def _emit_telemetry(self, now: float, state: ArmState) -> None:
        if now - self._last_telemetry < 1.0:
            return
        self._last_telemetry = now
        valid_temps = [float(t) for t in state.temperatures if t is not None and np.isfinite(float(t))]
        max_temp = max(valid_temps) if valid_temps else None
        tcp = self.kin.forward(state.position[:6])[:3, 3]
        self._log(
            "telemetry",
            dt_max_ms=round(self._telemetry_dt_max * 1000.0, 2),
            ik_fails=self._ik_fail_count,
            home_active=self.home_active,
            q_deg=np.rad2deg(state.position).round(2).tolist(),
            tcp_xyz=tcp.round(4).tolist(),
            max_temp_c=round(max_temp, 1) if max_temp is not None else None,
            gripper_torque=round(float(state.torque[6]), 3) if np.isfinite(state.torque[6]) else None,
        )
        self._telemetry_dt_max = 0.0
        self._ik_fail_count = 0

    def step(self) -> ControlFrame:
        if not self.started:
            raise RuntimeError("控制器尚未 start")
        sample = self.input.poll()
        if sample.exit or sample.estop:
            raise SafeStopRequested("用户请求安全停止")
        now = time.monotonic()
        dt = self.settings.control_period if self._last_time is None else now - self._last_time
        self._last_time = now
        self._telemetry_dt_max = max(self._telemetry_dt_max, float(dt))
        state = self.backend.read_state()
        self.joints.validate(state.position, soft=True)
        safety = self.settings.section("safety")
        check_temperatures(state.temperatures, float(safety.get("temperature_warn_c", 75.0)),
                           float(safety.get("temperature_trip_c", 80.0)))
        validate_feedback(state, safety, now=now, started=self.feedback_started)
        status = self.gate.update(
            sample.deadman, sample.heartbeat, now=now,
            streaming=getattr(sample, "streaming_heartbeat", True),
        )
        if not status.allowed or dt > 0.25:
            # 绝不在正常待机（未按使能）时每帧用受重力下垂的 state.position 覆盖 self.target，
            # 否则无重力前馈的 MIT PD 会形成“下垂→重设目标→继续下垂”的滑落正反馈。
            # 仅在控制周期卡顿（dt > 0.25s）或实际位置与保持目标偏差过大（如受阻/外力推偏 > 14°）时，
            # 才从实际位置安全重同步。
            self.home_active = False
            tracking_lag = float(np.max(np.abs(state.position[:6] - self.target[:6])))
            if dt > 0.25 or tracking_lag > 0.25:
                hold = self.joints.clamp(state.position)
                self.target = hold
            else:
                hold = self.joints.clamp(self.target)
                self.target = hold
            self.cartesian_target = self.kin.forward(hold[:6])
            self.rate.reset(hold)
            self._rebase_input(hold)
            reason = status.reason or f"控制周期过长(dt={dt:.2f}s)，已复位轨迹"
            if reason != self._wait_reason:
                self._log("gate_wait", reason=reason, dt_ms=round(dt * 1000.0, 1))
            if reason != self._wait_reason or now - self._wait_print >= 3.0:
                suffix = "；按住使能（Joy-Con R / 键盘 Space）即开始回 HOME" if self.home_pending else ""
                print(f"[等待] {reason}，机械臂保持当前姿态{suffix}", flush=True)
                self._wait_reason = reason
                self._wait_print = now
            self._was_allowed = False
            self.backend.command(self.target)
            return ControlFrame(state, self.target.copy(), False, reason, [])

        if not self._was_allowed:
            # 使能上升沿：沿用当前保持目标（若偏差过大则重同步），锚定手柄零跳变离合。
            tracking_lag = float(np.max(np.abs(state.position[:6] - self.target[:6])))
            hold = self.joints.clamp(state.position if tracking_lag > 0.25 else self.target)
            self.target = hold
            self.cartesian_target = self.kin.forward(hold[:6])
            self.rate.reset(hold)
            self._rebase_input(hold)
            self._log("gate_active", q_deg=np.rad2deg(state.position).round(2).tolist())
            self._was_allowed = True

        self._wait_reason = ""
        if self.home_pending or sample.home:
            self.home_pending = False
            self._begin_home()
        goal = (self.settings.home.copy() if self.home_active
                else self._goal_from_input(sample, dt))
        target = self.rate.step(goal, dt)
        arm_tol = float(safety.get("home_tolerance_rad", 0.04))
        grip_tol = float(safety.get("gripper_home_tolerance_rad", 0.15))
        if self.home_active:
            if (pose_reached(state.position, self.settings.home,
                             arm_tolerance_rad=arm_tol, gripper_tolerance_rad=grip_tol)
                    and pose_reached(target, self.settings.home,
                                     arm_tolerance_rad=arm_tol, gripper_tolerance_rad=grip_tol)):
                self.home_active = False
                self._rebase_input(target)
                # HOME 后必须松开再重新按住，避免手柄在回 HOME 期间的位移突然生效。
                self.gate.fault_latched = True
                self._log("home_done", q_deg=np.rad2deg(state.position).round(2).tolist())
                print("[HOME] 已到位；请完全松开使能，再重新按住后开始操作", flush=True)
            elif now - self._home_print >= 1.5:
                arm_err_deg = float(np.rad2deg(np.max(np.abs(state.position[:6] - self.settings.home[:6]))))
                grip_err_deg = float(np.rad2deg(abs(state.position[6] - self.settings.home[6])))
                print(f"[HOME] 进行中：六轴最大误差 {arm_err_deg:.1f}°，夹爪误差 {grip_err_deg:.1f}°",
                      flush=True)
                self._home_print = now

        measured_torque = float(state.torque[6]) if np.isfinite(state.torque[6]) else None
        target[6] = self.gripper_protection.apply(target[6], state.position[6], measured_torque)
        target = self.joints.clamp(target)
        self.backend.command(target)
        self.target = target.copy()
        self._emit_telemetry(now, state)
        for event in sample.episode_events:
            if self.on_episode_event:
                self.on_episode_event(event)
        return ControlFrame(state, target.copy(), True, "", sample.episode_events)

    def close(self) -> None:
        try:
            self.input.close()
        finally:
            if self.backend.enabled and not self.backend.stopped:
                self.backend.safe_stop()
            else:
                # connect() 在使能前失败时也必须释放 CAN/SDK 资源。
                self.backend.abort_before_enable()
