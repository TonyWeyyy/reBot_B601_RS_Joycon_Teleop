from __future__ import annotations

import glob
import importlib.util
import math
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

JOYCON_VENDOR_ID = 0x057E
JOYCON_L_PRODUCT_ID = 0x2006
JOYCON_R_PRODUCT_ID = 0x2007

_INPUT_REPORT_SIZE = 49
_RUMBLE_NEUTRAL = b"\x00\x01\x40\x40\x00\x01\x40\x40"
# STM LSM6DS3 (±8g / ±2000 dps): accel = 4g / 16384, gyro ≈ 0.070 deg/s/LSB.
_ACCEL_G_PER_LSB = 4.0 / 16384.0
_GYRO_RAD_S_PER_LSB = math.radians(0.070)
_STICK_DEFAULT_CENTER = 2048.0
_STICK_DEFAULT_SPAN = 1400.0


def load_hid_module():
    """优先加载 Linux hidraw 后端，使蓝牙 Joy-Con 稳定可见；回退标准 hid。"""
    try:
        import hidraw
        if hasattr(hidraw, "device") or hasattr(hidraw, "Device"):
            return hidraw
    except Exception:
        pass
    try:
        import hid
        if hasattr(hid, "device") or hasattr(hid, "Device"):
            return hid
    except Exception:
        pass
    purelib = Path(sysconfig.get_paths()["purelib"])
    for path in sorted(glob.glob(str(purelib / "hid.cpython-*.so"))):
        spec = importlib.util.spec_from_file_location("hid", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["hid"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop("hid", None)
            continue
        if hasattr(module, "device") or hasattr(module, "Device"):
            return module
        sys.modules.pop("hid", None)
    raise ImportError(
        "未找到可用的 hidraw/hidapi 后端；Linux 请安装 libhidapi-hidraw0 或 "
        "conda install -c conda-forge hidapi，并确认 /dev/hidraw* 权限"
    )


def enumerate_joycons(side: str = "right", hid_module=None) -> list[dict[str, Any]]:
    """严格按指定左右侧枚举 Joy-Con，绝不用对侧手柄替代。"""
    hid_mod = load_hid_module() if hid_module is None else hid_module
    wanted = "left" if "left" in str(side).lower() else "right"
    target_pid = JOYCON_L_PRODUCT_ID if wanted == "left" else JOYCON_R_PRODUCT_ID
    matches: list[dict[str, Any]] = []
    for info in hid_mod.enumerate(JOYCON_VENDOR_ID, target_pid):
        if info.get("vendor_id") != JOYCON_VENDOR_ID or info.get("product_id") != target_pid:
            continue
        serial = info.get("serial_number") or info.get("serial") or ""
        path = info.get("path")
        if not serial and not path:
            continue
        matches.append({
            "vendor_id": int(info["vendor_id"]),
            "product_id": int(info["product_id"]),
            "serial": str(serial),
            "path": path,
            "product_string": str(info.get("product_string") or ""),
        })
    return matches


class _HidHandle:
    """统一封装 cython-hidapi / hidraw (hid.device) 与 pyhidapi (hid.Device)。"""

    def __init__(self, hid_module, info: dict[str, Any]):
        self._dev = None
        self._is_caps_device = False
        if hasattr(hid_module, "device"):
            dev = hid_module.device()
            opened = False
            if info.get("path") is not None and hasattr(dev, "open_path"):
                try:
                    dev.open_path(info["path"])
                    opened = True
                except OSError:
                    opened = False
            if not opened:
                dev.open(info["vendor_id"], info["product_id"], info.get("serial") or None)
            self._dev = dev
            self._is_caps_device = False
        elif hasattr(hid_module, "Device"):
            if info.get("path") is not None:
                self._dev = hid_module.Device(path=info["path"])
            else:
                self._dev = hid_module.Device(
                    vid=info["vendor_id"], pid=info["product_id"],
                    serial=info.get("serial") or None)
            self._is_caps_device = True
        else:
            raise RuntimeError("当前 hid 模块既没有 device() 也没有 Device()")

    def write(self, data: bytes) -> None:
        if self._dev is None:
            return
        self._dev.write(data)

    def read(self, size: int = _INPUT_REPORT_SIZE, timeout_ms: int = 30) -> bytes:
        if self._dev is None:
            return b""
        if self._is_caps_device:
            raw = self._dev.read(size, timeout=timeout_ms)
        else:
            try:
                raw = self._dev.read(size, timeout_ms=timeout_ms)
            except TypeError:
                raw = self._dev.read(size, timeout_ms)
        return bytes(raw) if raw else b""

    def close(self) -> None:
        dev, self._dev = self._dev, None
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass


def _int16_le(lo: int, hi: int) -> int:
    value = (hi << 8) | lo
    return value - 65536 if value >= 32768 else value


def parse_report_0x30(
    report: bytes,
    *,
    is_right: bool = True,
) -> tuple[dict[str, bool], tuple[int, int], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """解析标准 49 字节 Joy-Con 0x30 输入报告（纯函数，便于测试）。"""
    if len(report) < _INPUT_REPORT_SIZE or report[0] != 0x30:
        raise ValueError("不是有效的 49 字节 Joy-Con 0x30 报告")
    b3, b4, b5 = report[3], report[4], report[5]
    if is_right:
        buttons = {
            "y": bool(b3 & 0x01),
            "x": bool(b3 & 0x02),
            "b": bool(b3 & 0x04),
            "a": bool(b3 & 0x08),
            "sr": bool(b3 & 0x10),
            "sl": bool(b3 & 0x20),
            "r": bool(b3 & 0x40),
            "zr": bool(b3 & 0x80),
            "minus": bool(b4 & 0x01),
            "plus": bool(b4 & 0x02),
            "r_stick": bool(b4 & 0x04),
            "home": bool(b4 & 0x10),
        }
        stick_h = report[9] | ((report[10] & 0x0F) << 8)
        stick_v = (report[10] >> 4) | (report[11] << 4)
    else:
        buttons = {
            "down": bool(b5 & 0x01),
            "up": bool(b5 & 0x02),
            "right": bool(b5 & 0x04),
            "left": bool(b5 & 0x08),
            "sr": bool(b5 & 0x10),
            "sl": bool(b5 & 0x20),
            "l": bool(b5 & 0x40),
            "zl": bool(b5 & 0x80),
            "minus": bool(b4 & 0x01),
            "plus": bool(b4 & 0x02),
            "l_stick": bool(b4 & 0x08),
            "capture": bool(b4 & 0x20),
        }
        stick_h = report[6] | ((report[7] & 0x0F) << 8)
        stick_v = (report[7] >> 4) | (report[8] << 4)

    accel: list[tuple[int, int, int]] = []
    gyro: list[tuple[int, int, int]] = []
    for idx in range(3):
        base = 13 + idx * 12
        ax = _int16_le(report[base], report[base + 1])
        ay = _int16_le(report[base + 2], report[base + 3])
        az = _int16_le(report[base + 4], report[base + 5])
        gx = _int16_le(report[base + 6], report[base + 7])
        gy = _int16_le(report[base + 8], report[base + 9])
        gz = _int16_le(report[base + 10], report[base + 11])
        accel.append((ax, ay, az))
        gyro.append((gx, gy, gz))
    return buttons, (stick_h, stick_v), accel, gyro


class QuaternionAttitudeFilter:
    """无奇异点 SO(3) 互补姿态解算器（世界坐标：+X 前、+Y 左、+Z 上）。

    使用四元数/旋转矩阵直接积分陀螺角速度，并用重力加速度仅修正横滚/俯仰倾角，
    因此：
    - 偏航（yaw）连续无跳变，不会在 ±90° 或 ±180° 反向；
    - 静止陀螺死区可抑制 yaw 漂移；
    - 无欧拉角万向节锁。
    """

    def __init__(self, *, is_right: bool = True,
                 gyro_deadband_rad_s: float = 0.03,
                 accel_gain: float = 2.0,
                 lowpass_alpha: float = 0.20):
        self.is_right = is_right
        self.gyro_deadband_rad_s = max(0.0, float(gyro_deadband_rad_s))
        self.accel_gain = max(0.0, float(accel_gain))
        self.lowpass_alpha = float(np.clip(lowpass_alpha, 0.01, 1.0))
        self._rot = np.eye(3, dtype=float)
        self._filtered = np.eye(3, dtype=float)
        self._initialized = False

    def reset(self) -> None:
        self._rot = np.eye(3, dtype=float)
        self._filtered = np.eye(3, dtype=float)
        self._initialized = False

    def _to_robot_body(self, vec: np.ndarray) -> np.ndarray:
        """把 Joy-Con 硬件 IMU 坐标变换为右手机器人坐标（+X 前、+Y 左、+Z 上）。

        右柄按键朝上、R/ZR 朝前平放时，硬件 IMU 满足 +X 朝前、+Y 朝右、+Z 朝下，
        绕 X 旋转 180° 即得 (x, -y, -z)。左柄 IMU 本身与右柄 y/z 相反，因此为 (x, y, z)。
        """
        if self.is_right:
            return np.array([vec[0], -vec[1], -vec[2]], dtype=float)
        return np.array([vec[0], vec[1], vec[2]], dtype=float)

    def update(self, accel_g: tuple[float, float, float] | np.ndarray,
               gyro_rad_s: tuple[float, float, float] | np.ndarray,
               dt: float) -> np.ndarray:
        dt = float(np.clip(dt, 1e-4, 0.05))
        a = self._to_robot_body(np.asarray(accel_g, dtype=float))
        g = self._to_robot_body(np.asarray(gyro_rad_s, dtype=float))
        if self.gyro_deadband_rad_s > 0.0:
            g = np.where(np.abs(g) < self.gyro_deadband_rad_s, 0.0, g)

        a_norm = float(np.linalg.norm(a))
        if not self._initialized and 0.5 <= a_norm <= 1.5:
            # 首帧用重力方向初始化水平倾角（yaw=0）。
            g_meas = a / a_norm
            z_world = np.array([0.0, 0.0, 1.0])
            axis = np.cross(g_meas, z_world)
            sin_a = float(np.linalg.norm(axis))
            cos_a = float(np.clip(np.dot(g_meas, z_world), -1.0, 1.0))
            if sin_a > 1e-6:
                angle = math.atan2(sin_a, cos_a)
                self._rot = Rotation.from_rotvec((axis / sin_a) * angle).as_matrix()
            else:
                self._rot = np.eye(3) if cos_a >= 0.0 else np.diag([1.0, -1.0, -1.0])
            self._filtered = self._rot.copy()
            self._initialized = True
            return self._filtered.copy()

        omega = g.copy()
        if self.accel_gain > 0.0 and 0.75 <= a_norm <= 1.25:
            g_meas = a / a_norm
            g_pred = self._rot.T @ np.array([0.0, 0.0, 1.0])
            # 叉乘误差严格垂直于重力，只修正横滚/俯仰，不污染 yaw。
            tilt_err = np.cross(g_meas, g_pred)
            omega = omega + self.accel_gain * tilt_err

        step_vec = omega * dt
        if float(np.linalg.norm(step_vec)) > 1e-12:
            self._rot = self._rot @ Rotation.from_rotvec(step_vec).as_matrix()
            # 正交化防数值漂移。
            u, _, vt = np.linalg.svd(self._rot)
            self._rot = u @ vt
            if np.linalg.det(self._rot) < 0.0:
                u[:, -1] *= -1.0
                self._rot = u @ vt

        # SO(3) 指数平滑，消除高频手抖且无欧拉角跳变。
        delta = Rotation.from_matrix(self._rot @ self._filtered.T).as_rotvec()
        self._filtered = Rotation.from_rotvec(delta * self.lowpass_alpha).as_matrix() @ self._filtered
        return self._filtered.copy()

    @property
    def rotation(self) -> np.ndarray:
        return self._filtered.copy()


@dataclass
class JoyConSnapshot:
    buttons: dict[str, bool] = field(default_factory=dict)
    stick_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    last_packet_monotonic: float | None = None
    packet_count: int = 0
    battery_level: int = 0


def apply_stick_deadzone(raw_h: float, raw_v: float, *,
                         center_h: float = _STICK_DEFAULT_CENTER,
                         center_v: float = _STICK_DEFAULT_CENTER,
                         span: float = _STICK_DEFAULT_SPAN,
                         deadzone: float = 0.15,
                         axis_signs: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """将原始 12-bit 摇杆读数转为机器人平面 [+X 前, +Y 左] 归一化速度 [-1, 1]。"""
    nx = (float(raw_v) - float(center_v)) / max(float(span), 100.0)
    ny = -(float(raw_h) - float(center_h)) / max(float(span), 100.0)
    nx = float(np.clip(nx, -1.0, 1.0)) * float(axis_signs[0])
    ny = float(np.clip(ny, -1.0, 1.0)) * float(axis_signs[1])

    def _dz(val: float) -> float:
        mag = abs(val)
        if mag <= deadzone:
            return 0.0
        return math.copysign((mag - deadzone) / max(1.0 - deadzone, 1e-6), val)

    return np.array([_dz(nx), _dz(ny)], dtype=float)


class NativeJoyConDriver:
    """单 hidraw 句柄、带锁快照、非阻塞可干净退出的 Joy-Con 驱动。"""

    def __init__(self, *, device: str = "right",
                 gyro_deadband_rad_s: float = 0.03,
                 lowpass_alpha: float = 0.20,
                 stick_deadzone: float = 0.15,
                 stick_axis_signs: tuple[float, float] = (1.0, 1.0),
                 calibration_s: float = 1.2,
                 hid_module=None):
        self.side = "left" if "left" in str(device).lower() else "right"
        self.is_right = self.side == "right"
        self.stick_deadzone = float(stick_deadzone)
        self.stick_axis_signs = (float(stick_axis_signs[0]), float(stick_axis_signs[1]))
        self._hid_mod = load_hid_module() if hid_module is None else hid_module
        devices = enumerate_joycons(self.side, hid_module=self._hid_mod)
        if not devices:
            raise RuntimeError(
                f"未找到已连接的 {self.side} Joy-Con（VID=0x057E）；"
                "请确认手柄已蓝牙配对唤醒、未分配给对侧、且当前用户拥有 /dev/hidraw* 权限"
            )
        self.device_info = devices[0]
        try:
            self._handle = _HidHandle(self._hid_mod, self.device_info)
        except OSError as exc:
            raise RuntimeError(
                f"打开 {self.side} Joy-Con hidraw 失败：{exc}；"
                "请检查 /dev/hidraw* 权限及是否被其他程序占用"
            ) from exc

        self._packet_number = 0
        self._accel_offset = np.zeros(3, dtype=float)
        self._accel_scale = np.ones(3, dtype=float)
        self._gyro_offset = np.zeros(3, dtype=float)
        self._gyro_scale = np.ones(3, dtype=float)
        self._stick_center = np.array([_STICK_DEFAULT_CENTER, _STICK_DEFAULT_CENTER], dtype=float)
        self._filter = QuaternionAttitudeFilter(
            is_right=self.is_right,
            gyro_deadband_rad_s=gyro_deadband_rad_s,
            lowpass_alpha=lowpass_alpha,
        )
        self._lock = threading.Lock()
        self._snapshot = JoyConSnapshot()
        self._running = False
        self._thread: threading.Thread | None = None

        try:
            self._initialize_device()
            self._calibrate_stationary(float(calibration_s))
            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop,
                name=f"rebot-joycon-{self.side}",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self.close()
            raise

    def _send_subcmd(self, subcmd: int, arg: bytes, *, expect_ack: bool = False,
                     timeout_s: float = 0.35) -> bytes | None:
        payload = bytes([0x01, self._packet_number & 0x0F]) + _RUMBLE_NEUTRAL + bytes([subcmd]) + arg
        self._packet_number = (self._packet_number + 1) & 0x0F
        self._handle.write(payload)
        if not expect_ack:
            return None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pkt = self._handle.read(_INPUT_REPORT_SIZE, timeout_ms=25)
            if len(pkt) >= 15 and pkt[0] == 0x21 and pkt[14] == subcmd:
                return pkt
        return None

    def _spi_read(self, address: int, size: int) -> bytes | None:
        arg = address.to_bytes(4, "little") + bytes([size])
        reply = self._send_subcmd(0x10, arg, expect_ack=True, timeout_s=0.35)
        if reply is None or not (reply[13] & 0x80) or len(reply) < 20 + size:
            return None
        return reply[20:20 + size]

    def _initialize_device(self) -> None:
        # 1) 尝试读取出厂/用户 IMU 标定参数（失败不阻塞，后续静态标定兜底）。
        try:
            user_magic = self._spi_read(0x8026, 2)
            imu_cal = self._spi_read(0x8028, 24) if user_magic == b"\xB2\xA1" else self._spi_read(0x6020, 24)
            if imu_cal is not None and len(imu_cal) == 24:
                self._accel_offset = np.array([
                    _int16_le(imu_cal[0], imu_cal[1]),
                    _int16_le(imu_cal[2], imu_cal[3]),
                    _int16_le(imu_cal[4], imu_cal[5]),
                ], dtype=float)
                accel_sens = np.array([
                    _int16_le(imu_cal[6], imu_cal[7]),
                    _int16_le(imu_cal[8], imu_cal[9]),
                    _int16_le(imu_cal[10], imu_cal[11]),
                ], dtype=float)
                self._accel_scale = np.where(
                    (accel_sens > 1000.0) & (accel_sens != 0x4000),
                    float(0x4000) / accel_sens,
                    1.0,
                )
                self._gyro_offset = np.array([
                    _int16_le(imu_cal[12], imu_cal[13]),
                    _int16_le(imu_cal[14], imu_cal[15]),
                    _int16_le(imu_cal[16], imu_cal[17]),
                ], dtype=float)
                gyro_sens = np.array([
                    _int16_le(imu_cal[18], imu_cal[19]),
                    _int16_le(imu_cal[20], imu_cal[21]),
                    _int16_le(imu_cal[22], imu_cal[23]),
                ], dtype=float)
                self._gyro_scale = np.where(
                    (gyro_sens > 1000.0) & (gyro_sens != 0x343B),
                    float(0x343B) / gyro_sens,
                    1.0,
                )
        except Exception:
            pass
        # 2) 开启 6 轴 IMU + 切换到标准 0x30 60Hz 全报告模式 + 点亮 1 号 LED。
        self._send_subcmd(0x40, b"\x01")
        time.sleep(0.03)
        self._send_subcmd(0x03, b"\x30")
        time.sleep(0.03)
        self._send_subcmd(0x30, b"\x01")

    def _calibrate_stationary(self, duration_s: float) -> None:
        """启动静止标定：测量实际陀螺零偏与摇杆中心（免受第三方手柄偏移影响）。"""
        if duration_s <= 0.0:
            return
        print(f"[joycon] 已连接 {self.side} Joy-Con；静止标定陀螺与摇杆中心（{duration_s:.1f}s），请勿晃动...",
              flush=True)
        gyro_samples: list[tuple[int, int, int]] = []
        stick_samples: list[tuple[int, int]] = []
        deadline = time.monotonic() + duration_s
        rot = self._filter.rotation
        while time.monotonic() < deadline:
            pkt = self._handle.read(_INPUT_REPORT_SIZE, timeout_ms=30)
            if len(pkt) < _INPUT_REPORT_SIZE or pkt[0] != 0x30:
                continue
            buttons, stick_raw, accel_raw, gyro_raw = parse_report_0x30(pkt, is_right=self.is_right)
            # 开启 IMU 后的首包常为全零上电噪声，必须跳过避免拉偏陀螺零偏与倾角初值。
            if all(v == (0, 0, 0) for v in accel_raw) and all(v == (0, 0, 0) for v in gyro_raw):
                continue
            gyro_samples.extend(gyro_raw)
            stick_samples.append(stick_raw)
            now = time.monotonic()
            # 用静止期加速度先把滤波器倾角初始化并收敛到位。
            for ax, ay, az in accel_raw:
                accel_g = ((np.array([ax, ay, az], dtype=float) - self._accel_offset)
                           * self._accel_scale * _ACCEL_G_PER_LSB)
                rot = self._filter.update(accel_g, np.zeros(3), 0.005)
            with self._lock:
                self._snapshot.buttons = buttons
                self._snapshot.rotation = rot
                self._snapshot.last_packet_monotonic = now
                self._snapshot.packet_count += 1
                self._snapshot.battery_level = (pkt[2] >> 5) & 0x07

        if len(gyro_samples) < 9:
            raise RuntimeError(
                f"{self.side} Joy-Con 未收到 0x30 IMU 报告；"
                "请检查手柄电量、蓝牙连接、hidraw 权限，并确保没有其他程序占用该手柄"
            )
        self._gyro_offset = np.mean(np.asarray(gyro_samples, dtype=float), axis=0)
        self._stick_center = np.mean(np.asarray(stick_samples, dtype=float), axis=0)
        with self._lock:
            self._snapshot.rotation = self._filter.rotation
            self._snapshot.stick_xy = np.zeros(2, dtype=float)
        print(f"[joycon] 标定完成（样本 {len(gyro_samples)}，摇杆中心 "
              f"{self._stick_center.round(1).tolist()}）", flush=True)

    def _process_packet(self, pkt: bytes, now: float, prev_time: float | None) -> None:
        buttons, stick_raw, accel_raw, gyro_raw = parse_report_0x30(pkt, is_right=self.is_right)
        dt_pkt = 0.015 if prev_time is None else float(np.clip(now - prev_time, 0.003, 0.06))
        dt_sub = dt_pkt / 3.0
        rot = self._filter.rotation
        for idx in range(3):
            accel_g = ((np.asarray(accel_raw[idx], dtype=float) - self._accel_offset)
                       * self._accel_scale * _ACCEL_G_PER_LSB)
            gyro_rad = ((np.asarray(gyro_raw[idx], dtype=float) - self._gyro_offset)
                        * self._gyro_scale * _GYRO_RAD_S_PER_LSB)
            rot = self._filter.update(accel_g, gyro_rad, dt_sub)
        stick_xy = apply_stick_deadzone(
            stick_raw[0], stick_raw[1],
            center_h=float(self._stick_center[0]),
            center_v=float(self._stick_center[1]),
            deadzone=self.stick_deadzone,
            axis_signs=self.stick_axis_signs,
        )
        battery = (pkt[2] >> 5) & 0x07
        with self._lock:
            self._snapshot = JoyConSnapshot(
                buttons=buttons,
                stick_xy=stick_xy,
                rotation=rot,
                last_packet_monotonic=now,
                packet_count=self._snapshot.packet_count + 1,
                battery_level=battery,
            )

    def _read_loop(self) -> None:
        prev_time: float | None = None
        while self._running:
            try:
                pkt = self._handle.read(_INPUT_REPORT_SIZE, timeout_ms=30)
            except OSError:
                # 设备断连：退出线程，不再刷新 last_packet_monotonic，看门狗将立即接管。
                self._running = False
                return
            if len(pkt) < _INPUT_REPORT_SIZE or pkt[0] != 0x30:
                continue
            now = time.monotonic()
            try:
                self._process_packet(pkt, now, prev_time)
                prev_time = now
            except Exception:
                continue

    def snapshot(self) -> JoyConSnapshot:
        with self._lock:
            return JoyConSnapshot(
                buttons=dict(self._snapshot.buttons),
                stick_xy=self._snapshot.stick_xy.copy(),
                rotation=self._snapshot.rotation.copy(),
                last_packet_monotonic=self._snapshot.last_packet_monotonic,
                packet_count=self._snapshot.packet_count,
                battery_level=self._snapshot.battery_level,
            )

    def close(self) -> None:
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._handle.close()
