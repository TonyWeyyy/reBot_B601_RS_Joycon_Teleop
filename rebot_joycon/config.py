from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"

# 兼容旧版 evdev 键名与新版原生 hidraw 键名。
LEGACY_BUTTON_ALIASES = {
    "BTN_TR": "r",
    "BTN_TR2": "zr",
    "BTN_THUMBR": "r_stick",
    "BTN_START": "plus",
    "BTN_SELECT": "minus",
    "BTN_MODE": "home",
    "BTN_EAST": "a",
    "BTN_SOUTH": "b",
    "BTN_WEST": "x",
    "BTN_NORTH": "y",
    "BTN_TL": "l",
    "BTN_TL2": "zl",
    "BTN_THUMBL": "l_stick",
    "BTN_Z": "capture",
}


def normalize_joycon_button(name: str) -> str:
    text = str(name).strip()
    if text in LEGACY_BUTTON_ALIASES:
        return LEGACY_BUTTON_ALIASES[text]
    return text.lower().replace("-", "_")


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是 {length} 个有限数值")
    return array


def _finite_scalar(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    return number


@dataclass(frozen=True)
class Settings:
    root: Path
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        config_path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("配置根节点必须是映射")
        settings = cls(config_path.parent.parent, data, config_path)
        settings.validate()
        return settings

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"配置 {name} 必须是映射")
        return value

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @property
    def urdf_path(self) -> Path:
        return self.resolve(self.section("arm").get("urdf", ""))

    @property
    def hardware_yaml(self) -> Path:
        return self.resolve(self.section("arm").get("hardware_yaml", ""))

    @property
    def control_period(self) -> float:
        return 1.0 / float(self.section("arm").get("control_rate_hz", 30.0))

    @property
    def home(self) -> np.ndarray:
        section = self.section("home")
        return np.r_[_finite_vector(section.get("q"), 6, "home.q"),
                     float(section.get("gripper", 0.0))]

    @property
    def gripper_limits(self) -> np.ndarray:
        value = _finite_vector(self.section("safety").get("gripper_limits_rad"), 2,
                               "safety.gripper_limits_rad")
        if value[0] >= value[1]:
            raise ValueError("夹爪限位顺序无效")
        return value

    @property
    def gripper_soft_limits(self) -> np.ndarray:
        scale = float(self.section("safety").get("joint_limit_scale", 0.7))
        return self.gripper_limits * scale

    @property
    def gripper_close_open(self) -> tuple[float, float]:
        """返回 (闭合弧度, 安全张开弧度)：0 为闭合点，非零端乘以软件缩放为安全张开点。"""
        soft = self.gripper_soft_limits
        if abs(soft[0]) <= abs(soft[1]):
            return float(soft[0]), float(soft[1])
        return float(soft[1]), float(soft[0])

    @property
    def max_velocity(self) -> np.ndarray:
        return _finite_vector(self.section("safety").get("max_joint_velocity_rad_s"),
                              7, "safety.max_joint_velocity_rad_s")

    @property
    def max_acceleration(self) -> np.ndarray:
        return _finite_vector(
            self.section("safety").get("max_joint_acceleration_rad_s2"),
            7, "safety.max_joint_acceleration_rad_s2")

    @property
    def workspace_origin(self) -> np.ndarray:
        raw = self.section("safety").get("workspace_origin_xyz", [0.0, 0.0, 0.145])
        return _finite_vector(raw, 3, "safety.workspace_origin_xyz")

    def workspace_kwargs(self) -> dict[str, Any]:
        safety = self.section("safety")
        return {
            "radius": float(safety.get("workspace_radius_m", 0.58)),
            "min_z": float(safety.get("workspace_min_z_m", 0.04)),
            "max_z": float(safety.get("workspace_max_z_m", 0.65)),
            "min_reach": float(safety.get("workspace_min_reach_m", 0.10)),
            "origin": self.workspace_origin,
        }

    def joycon_map(self) -> dict[str, Any]:
        path = self.resolve(self.section("joycon").get("map", "config/joycon_map.json"))
        data = json.loads(path.read_text(encoding="utf-8"))
        required = ("deadman", "estop", "home", "exit", "gripper")
        missing = [name for name in required if not data.get(name)]
        if missing:
            raise ValueError(f"Joy-Con 映射缺少：{', '.join(missing)}")
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            if key == "startup_guard_s":
                normalized[key] = float(value)
            elif isinstance(value, str):
                normalized[key] = normalize_joycon_button(value)
            else:
                normalized[key] = value
        normalized.setdefault("z_up", "x")
        normalized.setdefault("z_down", "b")
        normalized.setdefault("episode_end", "a")
        normalized.setdefault("episode_retry", "y")
        return normalized

    def validate(self) -> None:
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"RS URDF 不存在：{self.urdf_path}")
        if not self.hardware_yaml.is_file():
            raise FileNotFoundError(f"RS 硬件配置不存在：{self.hardware_yaml}")
        arm = self.section("arm")
        rate = _finite_scalar(arm.get("control_rate_hz", 30.0), "arm.control_rate_hz")
        if not 10.0 <= rate <= 100.0:
            raise ValueError("控制频率必须在 10–100 Hz")
        if np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0):
            raise ValueError("速度和加速度限制必须为正数")
        if np.any(self.max_velocity[:6] > 2.5) or self.max_velocity[6] > 3.5:
            raise ValueError("关节最大速度超出安全上限（六轴 ≤ 2.5 rad/s，夹爪 ≤ 3.5 rad/s）")

        safety = self.section("safety")
        scale = _finite_scalar(safety.get("joint_limit_scale", 0.7),
                               "safety.joint_limit_scale")
        if not 0.1 <= scale <= 0.7:
            raise ValueError("joint_limit_scale 必须在 0.1–0.7；禁止放大官方限位")
        origin = self.workspace_origin
        if not -0.2 <= origin[2] <= 0.3:
            raise ValueError("workspace_origin_xyz 的 z 坐标超出合理范围")
        radius = _finite_scalar(safety.get("workspace_radius_m", 0.58),
                                "safety.workspace_radius_m")
        if not 0.05 <= radius <= 0.62:
            raise ValueError("工作空间半径不得超过 0.62 m（约 70% RS 肩部有效臂展）")
        min_z = _finite_scalar(safety.get("workspace_min_z_m", 0.04),
                               "safety.workspace_min_z_m")
        max_z = _finite_scalar(safety.get("workspace_max_z_m", 0.65),
                               "safety.workspace_max_z_m")
        min_reach = _finite_scalar(safety.get("workspace_min_reach_m", 0.10),
                                   "safety.workspace_min_reach_m")
        max_sphere_z = float(origin[2]) + radius
        if min_z < 0.0 or max_z <= min_z or max_z > 0.75 or min_z > max_sphere_z:
            raise ValueError("工作空间 z 范围无效")
        if min_reach <= 0.0 or min_reach > radius:
            raise ValueError("workspace_min_reach_m 必须为正且不超过工作空间半径")
        lead_m = _finite_scalar(safety.get("max_cartesian_lead_m", 0.04),
                                "safety.max_cartesian_lead_m")
        lead_rad = _finite_scalar(safety.get("max_cartesian_lead_rad", 0.25),
                                  "safety.max_cartesian_lead_rad")
        if not 0.005 <= lead_m <= 0.15 or not 0.02 <= lead_rad <= 0.8:
            raise ValueError("笛卡尔跟踪超前限制超出安全范围")
        watchdog = _finite_scalar(safety.get("input_watchdog_s", 0.25),
                                  "safety.input_watchdog_s")
        if not 0.0 < watchdog <= 0.5:
            raise ValueError("输入看门狗必须在 0–0.5 s")
        startup_guard = _finite_scalar(safety.get("startup_guard_s", 1.0),
                                       "safety.startup_guard_s")
        if not 0.2 <= startup_guard <= 5.0:
            raise ValueError("safety.startup_guard_s 必须在 0.2–5 s")
        if not bool(safety.get("require_deadman", True)):
            raise ValueError("真机安全链禁止关闭 require_deadman")
        if not bool(safety.get("require_imu_heartbeat", True)):
            raise ValueError("Joy-Con 安全链禁止关闭 require_imu_heartbeat")
        feedback_timeout = _finite_scalar(safety.get("feedback_timeout_s", 0.5),
                                          "safety.feedback_timeout_s")
        feedback_grace = _finite_scalar(safety.get("feedback_grace_s", 5.0),
                                        "safety.feedback_grace_s")
        if not 0.1 <= feedback_timeout <= 2.0 or not 0.0 <= feedback_grace <= 15.0:
            raise ValueError("反馈超时必须在 0.1–2 s，启动宽限必须在 0–15 s")
        warn_c = _finite_scalar(safety.get("temperature_warn_c", 75.0),
                                "safety.temperature_warn_c")
        trip_c = _finite_scalar(safety.get("temperature_trip_c", 80.0),
                                "safety.temperature_trip_c")
        if not 0.0 < warn_c < trip_c or trip_c > 80.0:
            raise ValueError("温度阈值必须为正、warn < trip，且跳闸不得超过 80°C")
        torque_trip = _finite_scalar(safety.get("gripper_torque_trip_nm", 1.0),
                                     "safety.gripper_torque_trip_nm")
        if not 0.0 < torque_trip <= 1.0:
            raise ValueError("gripper_torque_trip_nm 必须在 0–1 N·m")
        home_timeout = _finite_scalar(safety.get("home_timeout_s", 15.0),
                                      "safety.home_timeout_s")
        home_tolerance = _finite_scalar(safety.get("home_tolerance_rad", 0.04),
                                        "safety.home_tolerance_rad")
        gripper_tol = _finite_scalar(safety.get("gripper_home_tolerance_rad", 0.15),
                                     "safety.gripper_home_tolerance_rad")
        if (home_timeout <= 0.0 or not 0.0 < home_tolerance <= 0.2
                or not 0.0 < gripper_tol <= 0.5):
            raise ValueError("HOME 超时或容差无效")

        home = self.home
        if not np.isfinite(home).all():
            raise ValueError("HOME 必须是七个有限数值")
        gripper_limits = self.gripper_limits
        if not gripper_limits[0] - 1e-9 <= home[6] <= gripper_limits[1] + 1e-9:
            raise ValueError("home.gripper 必须落在 safety.gripper_limits_rad 内")

        joycon = self.section("joycon")
        for key, minimum, maximum in (
            ("linear_speed_m_s", 0.01, 0.35),
            ("vertical_speed_m_s", 0.01, 0.30),
            ("stick_deadzone", 0.02, 0.45),
            ("position_scale", 1e-6, 2.0),
            ("rotation_scale", 1e-6, 2.0),
            ("gripper_travel_time_s", 0.1, 10.0),
            ("imu_yaw_scale", 1e-6, 2.0),
            ("imu_gyro_deadband_rad_s", 0.0, 0.2),
            ("imu_lowpass_alpha", 1e-6, 1.0),
            ("imu_max_rate_rad_s", 0.05, 3.0),
            ("exit_hold_s", 0.0, 5.0),
        ):
            if key not in joycon:
                continue
            value = _finite_scalar(joycon[key], f"joycon.{key}")
            if value < minimum or (maximum is not None and value > maximum):
                raise ValueError(f"joycon.{key} 超出允许范围")
        if "stick_axis_signs" in joycon:
            signs = _finite_vector(joycon["stick_axis_signs"], 2, "joycon.stick_axis_signs")
            if not np.all(np.isin(np.abs(signs), [1.0])):
                raise ValueError("joycon.stick_axis_signs 每个元素必须为 +1 或 -1")
        if "imu_axis_signs" in joycon:
            signs = _finite_vector(joycon["imu_axis_signs"], 3, "joycon.imu_axis_signs")
            if not np.all(np.isin(np.abs(signs), [1.0])):
                raise ValueError("joycon.imu_axis_signs 每个元素必须为 +1 或 -1")

        backend = arm.get("backend", "auto")
        if backend not in ("auto", "mit", "pos_vel", "fake"):
            raise ValueError(f"未知后端：{backend}")

    def with_max_arm_velocity(self, value: float | None) -> None:
        """CLI 使用的进程内覆盖；可降低或按安全上限提高六轴最大速度（0.005–2.0 rad/s）。"""
        if value is None:
            return
        if not math.isfinite(value) or not 0.005 <= value <= 2.0:
            raise ValueError("--max-vel 必须在 0.005–2.0 rad/s")
        velocities = self.max_velocity.copy()
        velocities[:6] = float(value)
        accelerations = self.max_acceleration.copy()
        # 提高速度上限时同步匹配制动加速度，避免高速下刹车距离过长。
        accelerations[:6] = np.clip(
            np.maximum(accelerations[:6], float(value) * 3.5), 0.1, 8.0,
        )
        self.section("safety")["max_joint_velocity_rad_s"] = velocities.tolist()
        self.section("safety")["max_joint_acceleration_rad_s2"] = accelerations.tolist()
