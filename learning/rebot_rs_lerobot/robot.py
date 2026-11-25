from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from rebot_joycon.backends import create_backend
from rebot_joycon.config import PROJECT_ROOT, Settings
from rebot_joycon.kinematics import RSKinematics
from rebot_joycon.safety import (GripperTorqueProtection, JointLimiter, JointRateLimiter,
                                 check_temperatures, pose_reached, validate_feedback,
                                 workspace_contains)

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
               "wrist_yaw", "wrist_roll", "gripper")


@dataclass
class RebotRSConfig:
    port: str = "can0"
    project_config: Path = PROJECT_ROOT / "config" / "default.yaml"
    control_backend: str = "auto"
    home_on_connect: bool = False
    max_relative_target: float | dict[str, float] | None = 5.0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("rebot_b601_rs")
@dataclass
class RebotRSRobotConfig(RobotConfig, RebotRSConfig):
    def __post_init__(self) -> None:
        RobotConfig.__post_init__(self)
        if self.port != "can0":
            raise ValueError("非 can0 通道请直接修改项目 config/rebotarm_rs.yaml")


class RebotRSFollower(Robot):
    config_class = RebotRSRobotConfig
    name = "rebot_b601_rs"
    supports_position_hold = True

    def __init__(self, config: RebotRSRobotConfig):
        super().__init__(config)
        self.config = config
        self.settings = Settings.load(config.project_config)
        self.settings.section("arm")["backend"] = config.control_backend
        self.kin = RSKinematics(self.settings.urdf_path)
        safety = self.settings.section("safety")
        self.joints = JointLimiter(self.kin.limits, self.settings.gripper_limits,
                                   float(safety.get("joint_limit_scale", 0.7)))
        self.joints.validate(self.settings.home)
        self.rate = JointRateLimiter(self.settings.max_velocity,
                                     self.settings.max_acceleration, self.joints)
        self.gripper = GripperTorqueProtection(
            self.settings.section("safety")["gripper_torque_trip_nm"])
        self.backend = create_backend(self.settings, config.control_backend)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.calibration = {}
        self._last_action_time = None
        self._feedback_started = time.monotonic()

    @property
    def _motors_ft(self):
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @cached_property
    def observation_features(self):
        features = dict(self._motors_ft)
        for name, camera in self.cameras.items():
            cfg = self.config.cameras[name]
            features[name] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{name}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def action_features(self):
        return dict(self._motors_ft)

    @property
    def is_connected(self):
        return self.backend.connected and all(c.is_connected for c in self.cameras.values())

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        return None

    def configure(self):
        return None

    def connect(self, calibrate: bool = True):
        state = self.backend.connect(enable=True)
        try:
            self.joints.validate(state.position, soft=True)
            self._feedback_started = time.monotonic()
            validate_feedback(state, self.settings.section("safety"),
                              now=self._feedback_started, started=self._feedback_started)
            self.rate.reset(state.position)
            for camera in self.cameras.values():
                camera.connect()
        except Exception:
            try:
                self.backend.safe_stop()
            finally:
                for camera in self.cameras.values():
                    if camera.is_connected:
                        camera.disconnect()
            raise
        self._last_action_time = time.monotonic()
        if self.config.home_on_connect:
            try:
                self.go_home()
            except Exception:
                try:
                    self.backend.safe_stop()
                except Exception as stop_error:
                    print(f"[arm] HOME 失败后的安全停止也失败：{stop_error}")
                for camera in self.cameras.values():
                    if camera.is_connected:
                        camera.disconnect()
                raise

    def _positions(self, state=None):
        state = self.backend.read_state() if state is None else state
        return dict(zip(JOINT_NAMES, np.rad2deg(state.position), strict=True))

    def get_observation(self):
        state = self.backend.read_state()
        observation = {f"{name}.pos": value for name, value in self._positions(state).items()}
        for name, camera in self.cameras.items():
            observation[name] = camera.read_latest()
            if getattr(camera, "use_depth", False):
                observation[f"{name}_depth"] = camera.read_latest_depth()
        return observation

    def send_action(self, action):
        state = self.backend.read_state()
        self.joints.validate(state.position, soft=True)
        safety = self.settings.section("safety")
        check_temperatures(state.temperatures, safety["temperature_warn_c"],
                           safety["temperature_trip_c"])
        now = time.monotonic()
        validate_feedback(state, safety, now=now, started=self._feedback_started)
        dt = (self.settings.control_period if self._last_action_time is None
              else max(0.0, now - self._last_action_time))
        self._last_action_time = now
        if dt > float(safety.get("action_watchdog_s", 0.5)):
            # 策略/采集线程卡顿后不得继续执行迟到的旧动作。
            self.rate.reset(state.position)
            self.backend.command(state.position.copy())
            return self._positions(state)
        goal = state.position.copy()
        for index, name in enumerate(JOINT_NAMES):
            key = f"{name}.pos"
            if key in action:
                value = float(action[key])
                if not math.isfinite(value):
                    raise ValueError(f"非有限动作：{key}")
                goal[index] = math.radians(value)
                if self.config.max_relative_target is not None:
                    limit = (self.config.max_relative_target.get(name, 5.0)
                             if isinstance(self.config.max_relative_target, dict)
                             else self.config.max_relative_target)
                    goal[index] = np.clip(goal[index], state.position[index] - math.radians(limit),
                                          state.position[index] + math.radians(limit))
        goal = self.joints.clamp(goal)
        candidate = self.rate.step(goal, min(dt, 0.1))
        point = self.kin.forward(candidate[:6])[:3, 3]
        if not workspace_contains(point, **self.settings.workspace_kwargs()):
            self.rate.reset(state.position)
            candidate = state.position.copy()
        torque = float(state.torque[6]) if np.isfinite(state.torque[6]) else None
        candidate[6] = self.gripper.apply(candidate[6], state.position[6], torque)
        self.backend.command(candidate)
        return {f"{name}.pos": math.degrees(value)
                for name, value in zip(JOINT_NAMES, candidate, strict=True)}

    def go_home(self):
        safety = self.settings.section("safety")
        current = self.backend.read_state().position
        est_time = float(np.max(np.abs(self.settings.home - current) / self.settings.max_velocity))
        budget = max(float(safety.get("home_timeout_s", 15.0)), est_time * 1.5 + 4.0)
        deadline = time.monotonic() + budget
        arm_tol = float(safety.get("home_tolerance_rad", 0.04))
        grip_tol = float(safety.get("gripper_home_tolerance_rad", 0.15))
        action = {f"{name}.pos": math.degrees(value)
                  for name, value in zip(JOINT_NAMES, self.settings.home, strict=True)}
        while time.monotonic() < deadline:
            self.send_action(action)
            if pose_reached(self.backend.read_state().position, self.settings.home,
                            arm_tolerance_rad=arm_tol, gripper_tolerance_rad=grip_tol):
                return
            time.sleep(self.settings.control_period)
        raise TimeoutError("HOME 未在超时内到位；保持使能，未直接失能")

    def disable_torque(self):
        # 为兼容 LeRobot API 保留名称，但实现不是危险的原地失能。
        self.backend.safe_stop()

    def disconnect(self):
        try:
            self.backend.safe_stop()
        finally:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
