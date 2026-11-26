from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator

from rebot_joycon.config import PROJECT_ROOT, Settings
from rebot_joycon.errors import SafeStopRequested
from rebot_joycon.inputs import JoyConInput
from rebot_joycon.kinematics import RSKinematics, acceptable_ik_result, make_pose
from rebot_joycon.safety import (JointLimiter, SafetyGate, pose_reached,
                                 project_workspace, workspace_contains)
from .robot import JOINT_NAMES


@TeleoperatorConfig.register_subclass("joycon_rs")
@dataclass
class JoyconRSConfig(TeleoperatorConfig):
    project_config: Path = PROJECT_ROOT / "config" / "default.yaml"


class JoyconRSTeleop(Teleoperator):
    config_class = JoyconRSConfig
    name = "joycon_rs"

    def __init__(self, config: JoyconRSConfig):
        super().__init__(config)
        self.config = config
        self.settings = Settings.load(config.project_config)
        self.kin = RSKinematics(self.settings.urdf_path)
        safety = self.settings.section("safety")
        self.joints = JointLimiter(self.kin.limits, self.settings.gripper_limits,
                                   float(safety.get("joint_limit_scale", 0.7)))
        self.joints.validate(self.settings.home)
        self.gate = SafetyGate(float(safety.get("input_watchdog_s", 0.25)),
                               float(safety.get("startup_guard_s", 1.0)), True)
        self.input = None
        self.feedback = None
        self.last_valid_q = self.settings.home[:6].copy()
        self.gripper_target = self.settings.home[6]
        self.cartesian_target = self.kin.forward(self.settings.home[:6])
        self.last_time = None
        self.home_active = False
        self.events = []
        self._was_allowed = False
        self._initialized_hold = False

    @property
    def action_features(self):
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def feedback_features(self):
        return self.action_features

    @property
    def is_connected(self):
        return self.input is not None

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        return None

    def configure(self):
        return None

    def connect(self, calibrate: bool = True):
        home_pose = self.kin.pose(self.settings.home[:6])
        scale = float(self.settings.section("safety").get("joint_limit_scale", 0.7))
        self.input = JoyConInput(self.settings, home_pose[:3],
                                 self.settings.gripper_limits[0] * scale,
                                 self.settings.gripper_limits[1], home_pose[3:])
        self.last_time = None
        self._was_allowed = False
        self._initialized_hold = False
        self.gate.started = time.monotonic()

    def _sync_anchor(self, q: np.ndarray) -> None:
        if self.input is None:
            return
        pose = self.kin.forward(q[:6])
        self.cartesian_target = pose.copy()
        self.input.sync_anchor(pose)

    def _workspace_ok(self, q: np.ndarray) -> bool:
        point = self.kin.forward(q)[:3, 3]
        return workspace_contains(point, **self.settings.workspace_kwargs(), tol=5e-4)

    def _limit_attitude_step(self, target_rot: np.ndarray, dt: float) -> np.ndarray:
        max_rate = float(self.settings.section("joycon").get("imu_max_rate_rad_s", 1.5))
        if max_rate <= 0.0:
            return target_rot.copy()
        current = self.cartesian_target[:3, :3]
        delta = Rotation.from_matrix(target_rot @ current.T).as_rotvec()
        angle = float(np.linalg.norm(delta))
        limit = max_rate * float(dt)
        if angle <= limit or angle < 1e-9:
            return target_rot.copy()
        return Rotation.from_rotvec(delta * (limit / angle)).as_matrix() @ current

    def _tether(self, current_pose: np.ndarray, desired: np.ndarray) -> np.ndarray:
        safety = self.settings.section("safety")
        max_lead_m = float(safety.get("max_cartesian_lead_m", 0.04))
        max_lead_rad = float(safety.get("max_cartesian_lead_rad", 0.25))
        out = desired.copy()
        dp = out[:3, 3] - current_pose[:3, 3]
        dist = float(np.linalg.norm(dp))
        if dist > max_lead_m and dist > 1e-9:
            out[:3, 3] = current_pose[:3, 3] + dp * (max_lead_m / dist)
        rotvec = Rotation.from_matrix(out[:3, :3] @ current_pose[:3, :3].T).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle > max_lead_rad and angle > 1e-9:
            out[:3, :3] = (Rotation.from_rotvec(rotvec * (max_lead_rad / angle)).as_matrix()
                           @ current_pose[:3, :3])
        return out

    def send_feedback(self, feedback):
        values = np.array([feedback[f"{name}.pos"] for name in JOINT_NAMES], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("非有限 RS 反馈")
        self.feedback = np.deg2rad(values)
        if not self._initialized_hold:
            clamped = self.joints.clamp(self.feedback)
            self.last_valid_q = clamped[:6].copy()
            self.gripper_target = float(clamped[6])
            self._sync_anchor(self.last_valid_q)
            self._initialized_hold = True

    def get_action(self):
        if self.input is None or self.feedback is None:
            raise RuntimeError("Joy-Con 和 RS 反馈必须先连接")
        sample = self.input.poll()
        if sample.exit or sample.estop:
            raise SafeStopRequested("Joy-Con 请求安全停止")
        status = self.gate.update(sample.deadman, sample.heartbeat,
                                  streaming=sample.streaming_heartbeat)
        now = time.monotonic()
        dt = (self.settings.control_period if self.last_time is None
              else min(max(now - self.last_time, 1e-4), 0.1))
        self.last_time = now
        if not status.allowed:
            self.home_active = False
            self._was_allowed = False
            # 待机保持指令目标（若外部推偏 >0.25 rad 才重同步），避免 PD 下垂越积越多。
            if float(np.max(np.abs(self.feedback[:6] - self.last_valid_q))) > 0.25:
                clamped = self.joints.clamp(self.feedback)
                self.last_valid_q = clamped[:6].copy()
                self.gripper_target = float(clamped[6])
            self._sync_anchor(self.last_valid_q)
            held = np.r_[self.last_valid_q, self.gripper_target]
            return {f"{name}.pos": math.degrees(value)
                    for name, value in zip(JOINT_NAMES, held, strict=True)}
        if not self._was_allowed:
            if float(np.max(np.abs(self.feedback[:6] - self.last_valid_q))) > 0.25:
                clamped = self.joints.clamp(self.feedback)
                self.last_valid_q = clamped[:6].copy()
                self.gripper_target = float(clamped[6])
            self._sync_anchor(self.last_valid_q)
            self._was_allowed = True
        if sample.home:
            self.home_active = True
            self._sync_anchor(self.settings.home[:6])
        safety = self.settings.section("safety")
        if self.home_active:
            q = self.settings.home[:6].copy()
            gripper = self.settings.home[6]
            self.last_valid_q = q.copy()
            self.gripper_target = gripper
            if pose_reached(self.feedback, self.settings.home,
                            arm_tolerance_rad=float(safety.get("home_tolerance_rad", 0.04)),
                            gripper_tolerance_rad=float(safety.get("gripper_home_tolerance_rad", 0.15))):
                self.home_active = False
                self.gate.fault_latched = True
        else:
            current_pose = self.kin.forward(self.last_valid_q)
            desired = self.cartesian_target.copy()
            if sample.cartesian_velocity_m_s is not None:
                desired[:3, 3] = desired[:3, 3] + np.asarray(sample.cartesian_velocity_m_s) * dt
            if sample.target_rotation is not None:
                desired[:3, :3] = self._limit_attitude_step(sample.target_rotation, dt)
            elif sample.absolute_pose is not None:
                desired = make_pose(sample.absolute_pose)
                desired[:3, :3] = self._limit_attitude_step(desired[:3, :3], dt)
            projected = project_workspace(desired, **self.settings.workspace_kwargs())
            self.cartesian_target = self._tether(current_pose, projected)
            soft_limits = self.joints.soft[:, :6]
            result = self.kin.ik(self.last_valid_q, self.cartesian_target, joint_limits=soft_limits)
            if acceptable_ik_result(result) and self._workspace_ok(result.q):
                q = result.q
            else:
                fallback = self.kin.ik(self.last_valid_q, self.cartesian_target,
                                       joint_limits=soft_limits,
                                       position_weight=6.0, orientation_weight=0.25)
                if fallback.improved and fallback.position_error <= 0.015 and self._workspace_ok(fallback.q):
                    q = fallback.q
                    self.cartesian_target = self.kin.forward(fallback.q)
                else:
                    q = self.last_valid_q
                    self.cartesian_target = current_pose
            span = abs(self.joints.soft[1, 6] - self.joints.soft[0, 6])
            step = span / max(float(self.settings.section("joycon").get("gripper_travel_time_s", 1.0)), 0.1) * dt
            self.gripper_target += np.clip(sample.gripper_absolute - self.gripper_target,
                                           -step, step)
            gripper = self.gripper_target
        target = self.joints.clamp(np.r_[q, gripper])
        self.last_valid_q = target[:6].copy()
        self.events.extend(sample.episode_events)
        return {f"{name}.pos": math.degrees(value)
                for name, value in zip(JOINT_NAMES, target, strict=True)}

    def update_recording_events(self, events):
        pending, self.events = self.events, []
        if pending:
            events["exit_early"] = True
            if -1 in pending:
                events["rerecord_episode"] = True

    def disconnect(self):
        if self.input is not None:
            self.input.close()
            self.input = None
