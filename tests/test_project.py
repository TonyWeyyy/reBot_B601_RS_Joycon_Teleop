from __future__ import annotations

import copy
import json
import math
import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from rebot_joycon.backends import (ArmState, FakeBackend, PosVelBackend, RebotMITBackend,
                                   _load_mit_controller)
from rebot_joycon.calibration import finite_or_none
from rebot_joycon.config import Settings, normalize_joycon_button
from rebot_joycon.controller import TeleopController
from rebot_joycon.errors import SafetyError
from rebot_joycon.inputs import InputSample, JoyConInput, KeyboardInput
from rebot_joycon.joycon_driver import (
    JOYCON_L_PRODUCT_ID,
    JOYCON_R_PRODUCT_ID,
    JOYCON_VENDOR_ID,
    JoyConSnapshot,
    QuaternionAttitudeFilter,
    apply_stick_deadzone,
    enumerate_joycons,
    parse_report_0x30,
)
from rebot_joycon.kinematics import (IKResult, RSKinematics,
                                     acceptable_ik_result, make_pose)
from rebot_joycon.safety import (GripperTorqueProtection, JointLimiter, JointRateLimiter,
                                 SafetyGate, check_temperatures, pose_reached,
                                 project_workspace, validate_feedback, workspace_contains)
from rebot_joycon.simulator import MuJoCoView


class ScriptedInput:
    def __init__(self, samples=None):
        self.samples = list(samples or [])
        self.closed = False

    def poll(self):
        if self.samples:
            sample = self.samples.pop(0)
        else:
            sample = InputSample(True, None)
        sample.heartbeat = time.monotonic()
        return sample

    def close(self):
        self.closed = True


class FakeJoyConDriver:
    def __init__(self):
        self.snap = JoyConSnapshot(
            buttons={},
            stick_xy=np.zeros(2, dtype=float),
            rotation=np.eye(3, dtype=float),
            last_packet_monotonic=time.monotonic(),
            packet_count=1,
            battery_level=4,
        )
        self.closed = False

    def snapshot(self) -> JoyConSnapshot:
        return JoyConSnapshot(
            buttons=dict(self.snap.buttons),
            stick_xy=self.snap.stick_xy.copy(),
            rotation=self.snap.rotation.copy(),
            last_packet_monotonic=self.snap.last_packet_monotonic,
            packet_count=self.snap.packet_count,
            battery_level=self.snap.battery_level,
        )

    def close(self) -> None:
        self.closed = True


def _build_report_0x30(
    *,
    b3: int = 0,
    b4: int = 0,
    b5: int = 0,
    stick_r: tuple[int, int] = (2048, 2048),
    accel: tuple[int, int, int] = (0, 0, -4096),
    gyro: tuple[int, int, int] = (0, 0, 0),
) -> bytes:
    buf = bytearray(49)
    buf[0] = 0x30
    buf[2] = 0x80
    buf[3] = b3
    buf[4] = b4
    buf[5] = b5
    rh, rv = stick_r
    buf[9] = rh & 0xFF
    buf[10] = ((rh >> 8) & 0x0F) | ((rv & 0x0F) << 4)
    buf[11] = (rv >> 4) & 0xFF
    for idx in range(3):
        base = 13 + idx * 12
        vals = list(accel) + list(gyro)
        for j, val in enumerate(vals):
            u16 = val & 0xFFFF
            buf[base + j * 2] = u16 & 0xFF
            buf[base + j * 2 + 1] = (u16 >> 8) & 0xFF
    return bytes(buf)


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings.load()
        cls.kin = RSKinematics(cls.settings.urdf_path)

    def test_01_config_and_model(self):
        self.assertEqual(self.kin.limits.shape, (2, 6))
        self.assertEqual(self.kin.forward(self.settings.home[:6]).shape, (4, 4))
        self.assertTrue(self.settings.hardware_yaml.is_file())
        hw_cfg = yaml.safe_load(self.settings.hardware_yaml.read_text(encoding="utf-8"))
        self.assertTrue(self.settings.resolve(hw_cfg["urdf_path"]).is_file())
        jmap = self.settings.joycon_map()
        self.assertEqual(jmap["deadman"], "r")
        self.assertEqual(jmap["z_up"], "x")
        self.assertEqual(jmap["z_down"], "b")
        self.assertEqual(jmap["home"], "plus")
        self.assertEqual(normalize_joycon_button("BTN_TR"), "r")

    def test_02_home_workspace_and_level_orientation(self):
        pose = self.kin.pose(self.settings.home[:6])
        ws = self.settings.workspace_kwargs()
        self.assertTrue(workspace_contains(pose[:3], **ws))
        # HOME 应为水平向前（rpy ≈ 0），且距离边界留有充足余量（>= 0.10 m）。
        np.testing.assert_allclose(pose[3:], np.zeros(3), atol=1e-2)
        sh_dist = float(np.linalg.norm(pose[:3] - ws["origin"]))
        self.assertLessEqual(sh_dist, ws["radius"] - 0.10)
        limiter = JointLimiter(self.kin.limits, self.settings.gripper_limits)
        margins = np.rad2deg(np.minimum(
            self.settings.home[:6] - limiter.soft[0, :6],
            limiter.soft[1, :6] - self.settings.home[:6],
        ))
        self.assertGreaterEqual(float(np.min(margins)), 30.0)

    def test_03_soft_limits(self):
        limiter = JointLimiter(self.kin.limits, self.settings.gripper_limits)
        self.assertLessEqual(np.max(np.abs(limiter.soft[:, 0])), np.deg2rad(150) * 0.7 + 1e-9)
        self.assertAlmostEqual(limiter.soft[1, 1], self.kin.limits[1, 1] * 0.7)
        limiter.validate(self.settings.home)

    def test_04_rate_and_acceleration(self):
        limiter = JointLimiter(self.kin.limits, self.settings.gripper_limits)
        rate = JointRateLimiter(np.full(7, 0.1), np.full(7, 0.2), limiter)
        rate.reset(np.zeros(7))
        first = rate.step(np.full(7, 0.5), 0.1)
        second = rate.step(np.full(7, 0.5), 0.1)
        self.assertLessEqual(np.max(np.abs(first)), 0.0020001)
        self.assertLessEqual(np.max(np.abs(second - first)) / 0.1, 0.100001)

    def test_05_workspace_projection(self):
        origin = np.array([0.0, 0.0, 0.145])
        pose = make_pose(np.array([2.0, 0.0, 0.25, 0, 0, 0]))
        projected = project_workspace(pose, radius=0.58, min_z=0.04, max_z=0.65,
                                      min_reach=0.10, origin=origin)
        self.assertAlmostEqual(float(np.linalg.norm(projected[:3, 3] - origin)), 0.58, places=6)
        self.assertTrue(workspace_contains(projected[:3, 3], radius=0.58, min_z=0.04,
                                           max_z=0.65, min_reach=0.10, origin=origin))

    def test_06_gate_watchdog_latch(self):
        gate = SafetyGate(0.25, 0.0, True)
        now = time.monotonic()
        self.assertTrue(gate.update(True, now, now=now).allowed)
        self.assertFalse(gate.update(True, now - 1, now=now).allowed)
        self.assertFalse(gate.update(True, now, now=now).allowed)
        self.assertFalse(gate.update(False, now, now=now).allowed)
        self.assertTrue(gate.update(True, now, now=now).allowed)

    def test_07_gripper_reverse_release(self):
        protection = GripperTorqueProtection(1.0)
        held = protection.apply(-1.0, -0.5, 2.0)
        self.assertEqual(held, -0.5)
        self.assertEqual(protection.apply(0.0, -0.5, 0.0), 0.0)
        self.assertIsNone(protection.hold)

    def test_08_temperature_none_and_trip(self):
        check_temperatures([None, None], 75, 80)
        with self.assertRaises(SafetyError):
            check_temperatures([None, 81], 75, 80)

    def test_09_ik_near_zero_moves(self):
        seed = np.zeros(6)
        target_q = np.array([0.0, 0.08, 0.08, 0.0, 0.0, 0.0])
        result = self.kin.ik(seed, self.kin.forward(target_q), max_iterations=6)
        self.assertTrue(np.isfinite(result.q).all())
        self.assertTrue(result.improved)
        self.assertGreater(np.linalg.norm(result.q), 0.0)

    def test_10_controller_requires_deadman(self):
        source = ScriptedInput([InputSample(False, None)])
        backend = FakeBackend(self.settings.home)
        controller = TeleopController(self.settings, backend, source, self.kin, "qpos")
        controller.start()
        controller.gate.started -= 2
        before = backend.commands[-1].copy() if backend.commands else backend.position.copy()
        frame = controller.step()
        self.assertFalse(frame.allowed)
        self.assertTrue(np.allclose(frame.target, before))
        controller.close()
        self.assertEqual(backend.safe_stop_calls, 1)

    def test_11_home_is_persistent(self):
        samples = [InputSample(True, None, home=True)] + [InputSample(True, None) for _ in range(20)]
        source = ScriptedInput(samples)
        backend = FakeBackend(np.zeros(7))
        controller = TeleopController(self.settings, backend, source, self.kin, "qpos")
        controller.start()
        controller.gate.started -= 2
        controller.step()
        after_first = controller.target.copy()
        self.assertFalse(np.allclose(after_first, self.settings.home))
        for _ in range(20):
            controller.step()
        self.assertLess(np.linalg.norm(controller.target - self.settings.home),
                        np.linalg.norm(after_first - self.settings.home))
        controller.close()

    def test_12_mit_safe_stop_contract(self):
        backend = RebotMITBackend.__new__(RebotMITBackend)
        backend.connected = True
        backend.enabled = True
        backend.stopped = False
        backend.controller = Mock()
        backend.settings = self.settings
        backend.read_state = Mock(return_value=ArmState(
            np.zeros(7), np.zeros(7), np.zeros(7), [30.0] * 7, time.monotonic()))
        with patch("rebot_joycon.backends.time.sleep", return_value=None):
            backend.safe_stop()
        backend.controller.stop.assert_called_once_with(return_to_zero=False, wait=True)
        self.assertTrue(backend.stopped)

    def test_13_nonfinite_command_rejected(self):
        backend = FakeBackend(self.settings.home)
        backend.connect()
        with self.assertRaises(Exception):
            backend.command(np.full(7, np.nan))
        backend.safe_stop()

    def test_14_right_joycon_strict_pid_never_substitutes_left(self):
        fake_hid = types.SimpleNamespace(
            enumerate=lambda vid, pid: [
                {"vendor_id": JOYCON_VENDOR_ID, "product_id": JOYCON_L_PRODUCT_ID,
                 "serial_number": "AA:BB:CC", "path": b"/dev/hidraw0", "product_string": "Joy-Con (L)"},
                {"vendor_id": JOYCON_VENDOR_ID, "product_id": JOYCON_R_PRODUCT_ID,
                 "serial_number": "DD:EE:FF", "path": b"/dev/hidraw1", "product_string": "Joy-Con (R)"},
            ]
        )
        right = enumerate_joycons("right", hid_module=fake_hid)
        self.assertEqual(len(right), 1)
        self.assertEqual(right[0]["product_id"], JOYCON_R_PRODUCT_ID)

        left_only_hid = types.SimpleNamespace(
            enumerate=lambda vid, pid: [
                {"vendor_id": JOYCON_VENDOR_ID, "product_id": JOYCON_L_PRODUCT_ID,
                 "serial_number": "AA:BB:CC", "path": b"/dev/hidraw0", "product_string": "Joy-Con (L)"},
            ]
        )
        self.assertEqual(enumerate_joycons("right", hid_module=left_only_hid), [])

    def test_15_parse_report_0x30_scheme_a_buttons_and_stick(self):
        # 右柄：R(0x40) + X(0x02) + ZR(0x80) 在 b3；Plus(0x02) + Home(0x10) + R-Stick(0x04) 在 b4。
        pkt = _build_report_0x30(
            b3=0x40 | 0x02 | 0x80,
            b4=0x02 | 0x10 | 0x04,
            stick_r=(1000, 3200),
            accel=(100, -200, -4096),
            gyro=(10, -20, 30),
        )
        buttons, stick_raw, accel, gyro = parse_report_0x30(pkt, is_right=True)
        self.assertTrue(buttons["r"])
        self.assertTrue(buttons["x"])
        self.assertTrue(buttons["zr"])
        self.assertTrue(buttons["plus"])
        self.assertTrue(buttons["home"])
        self.assertTrue(buttons["r_stick"])
        self.assertFalse(buttons["b"])
        self.assertEqual(stick_raw, (1000, 3200))
        self.assertEqual(accel[0], (100, -200, -4096))
        self.assertEqual(gyro[0], (10, -20, 30))
        xy = apply_stick_deadzone(1000, 3200, center_h=2048, center_v=2048, deadzone=0.15)
        self.assertGreater(xy[0], 0.5)  # 推前 -> +X
        self.assertGreater(xy[1], 0.5)  # 推左 -> +Y

    def test_16_calibration_nonfinite_values_become_null(self):
        values = finite_or_none([np.nan, np.inf, -np.inf, 1.25, None])
        self.assertEqual(values, [None, None, None, 1.25, None])
        self.assertEqual(json.loads(json.dumps(values, allow_nan=False)), values)

    def test_17_start_failure_releases_unenabled_backend(self):
        class FailingBackend(FakeBackend):
            def connect(self, *, enable=True):
                self.connected = True
                raise SafetyError("模拟使能前失败")

        source = ScriptedInput()
        backend = FailingBackend(self.settings.home)
        controller = TeleopController(self.settings, backend, source, self.kin, "qpos")
        with self.assertRaises(SafetyError):
            controller.start()
        controller.close()
        self.assertFalse(backend.connected)
        self.assertEqual(backend.safe_stop_calls, 0)
        self.assertTrue(source.closed)

    def test_18_quaternion_filter_continuous_yaw_and_zero_stationary_drift(self):
        filt = QuaternionAttitudeFilter(is_right=True, gyro_deadband_rad_s=0.03,
                                        accel_gain=2.0, lowpass_alpha=1.0)
        # 静止平放（右柄 IMU +Z 朝下，重力反作用 accel = (0, 0, -1g)）。
        r0 = filt.update((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), 0.005)
        np.testing.assert_allclose(r0, np.eye(3), atol=1e-6)
        # 静止期间小于死区的微噪声不应引起任何漂移。
        for _ in range(200):
            r_still = filt.update((0.0, 0.0, -1.0), (0.01, -0.01, 0.02), 0.005)
        np.testing.assert_allclose(r_still, np.eye(3), atol=1e-6)
        # 绕机器人 +Z（右柄 IMU -Z）连续旋转 150°，绝不在 90° 反向。
        total_rad = math.radians(150.0)
        steps = 300
        dt = 0.005
        omega = total_rad / (steps * dt)
        prev_yaw = 0.0
        for _ in range(steps):
            # IMU 绕 -Z 旋转 => gyro_imu_z = -omega => 机器人 +Z 旋转 +omega
            rot = filt.update((0.0, 0.0, -1.0), (0.0, 0.0, -omega), dt)
            yaw = float(Rotation.from_matrix(rot).as_euler("xyz")[2])
            self.assertGreaterEqual(yaw, prev_yaw - 1e-6)
            prev_yaw = yaw
        self.assertAlmostEqual(prev_yaw, total_rad, places=2)

    def test_19_keyboard_joint_and_gripper_keys_are_independent(self):
        source = KeyboardInput.__new__(KeyboardInput)
        source._lock = threading.Lock()
        source._home_edge = False
        source._exit_edge = False
        source._listener = types.SimpleNamespace(running=True)
        source._pressed = {"1"}
        sample = source.poll()
        self.assertEqual(sample.joint_direction[0], 1.0)
        self.assertEqual(sample.gripper_direction, 0.0)
        source._pressed = {"7"}
        sample = source.poll()
        self.assertTrue(np.all(sample.joint_direction == 0.0))
        self.assertEqual(sample.gripper_direction, 1.0)

    def test_20_current_rebot_package_layout_is_supported(self):
        class FakeMITController:
            pass

        package = types.ModuleType("rebot")
        package.__path__ = []
        package.ReBotRSMITController = FakeMITController
        robot = types.ModuleType("rebot.robot")
        robot.ReBotRSMITController = FakeMITController
        package.robot = robot
        with patch.dict(sys.modules, {"rebot": package, "rebot.robot": robot}):
            self.assertIs(_load_mit_controller(), FakeMITController)

    def test_21_current_mit_feedback_api_is_supported(self):
        class CurrentMITController:
            last_temperatures = [None, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]

            def read_joint_angles(self):
                return [0.0, 10.0, -10.0, 5.0, -5.0, 90.0, 45.0]

        backend = RebotMITBackend.__new__(RebotMITBackend)
        backend.controller = CurrentMITController()
        state = backend.read_state()
        np.testing.assert_allclose(state.position,
                                   np.deg2rad([0, 10, -10, 5, -5, 90, 45]))
        self.assertEqual(state.temperatures[0], None)
        self.assertEqual(state.temperatures[6], 35.0)
        self.assertTrue(np.isnan(state.torque[0]))

    def test_22_joycon_scheme_a_controls_and_deadman_clutch(self):
        driver = FakeJoyConDriver()
        home_pose = self.kin.pose(self.settings.home[:6])
        joy = JoyConInput(self.settings, home_pose[:3], -4.0, 0.0, home_pose[3:], driver=driver)
        joy._started -= 2.0  # 跳过启动保护
        joy.sync_anchor(self.kin.forward(self.settings.home[:6]))

        # 未按 R 时按 ZR 不应提前切换夹爪。
        driver.snap.buttons = {"zr": True}
        s0 = joy.poll()
        self.assertFalse(s0.deadman)
        self.assertEqual(s0.gripper_absolute, 0.0)

        # 松开 ZR 后按住 R + X（上升）+ 推摇杆前左 + 按 ZR（张开）+ 按 A（episode 结束）。
        driver.snap.buttons = {"r": True, "x": True, "a": True}
        driver.snap.stick_xy = np.array([1.0, -0.5])
        s1 = joy.poll()
        self.assertTrue(s1.deadman)
        self.assertGreater(s1.cartesian_velocity_m_s[0], 0.05)
        self.assertLess(s1.cartesian_velocity_m_s[1], -0.02)
        self.assertGreater(s1.cartesian_velocity_m_s[2], 0.05)
        self.assertEqual(s1.episode_events, [1])

        driver.snap.buttons = {"r": True, "b": True, "zr": True}
        s2 = joy.poll()
        self.assertLess(s2.cartesian_velocity_m_s[2], -0.05)
        self.assertEqual(s2.gripper_absolute, -4.0)

        # 按住 R + R-Stick 立即触发 estop。
        driver.snap.buttons = {"r": True, "r_stick": True}
        self.assertTrue(joy.poll().estop)

        # 点按 + 触发 home；短按 Home 不触发退出，长按 >= exit_hold_s 才触发退出。
        driver.snap.buttons = {"plus": True, "home": True}
        s_home = joy.poll()
        self.assertTrue(s_home.home)
        self.assertFalse(s_home.exit)
        joy._exit_press_started -= 1.0
        self.assertTrue(joy.poll().exit)

        joy.close()
        self.assertTrue(driver.closed)

    def test_23_joycon_so3_body_clutch_tracks_without_jump_and_ignores_calib_yaw(self):
        driver = FakeJoyConDriver()
        # 即使开机桌面校准朝向有 90° 偏航，体坐标相对离合姿态也只按手柄自身轴右乘到机械臂，不串轴、无跳变。
        base_rot = Rotation.from_euler("xyz", [0.1, -0.2, math.radians(90.0)]).as_matrix()
        driver.snap.rotation = base_rot.copy()
        home_tf = self.kin.forward(self.settings.home[:6])
        joy = JoyConInput(self.settings, home_tf[:3, 3], 4.0, 0.0, driver=driver)
        joy.sync_anchor(home_tf)

        # 未转动手柄时，目标姿态严格等于机械臂当前姿态（无跳变）。
        s0 = joy.poll()
        self.assertIsNotNone(s0.target_rotation)
        np.testing.assert_allclose(s0.target_rotation, home_tf[:3, :3], atol=1e-9)

        # 手柄绕自身体坐标 +X 转 20°（物理滚转），机械臂末端应绕末端局部 +X 转 20°（右乘）。
        delta = Rotation.from_euler("xyz", [math.radians(20.0), 0.0, 0.0]).as_matrix()
        driver.snap.rotation = base_rot @ delta
        s1 = joy.poll()
        np.testing.assert_allclose(s1.target_rotation, home_tf[:3, :3] @ delta, atol=1e-6)

        # R 离合上升沿重新锚定当前手柄朝向，保持零跳变。
        driver.snap.buttons = {"r": True}
        driver.snap.rotation = driver.snap.rotation @ delta
        clutch_in = joy.poll()
        np.testing.assert_allclose(clutch_in.target_rotation, home_tf[:3, :3], atol=1e-9)

    def test_24_pos_vel_reads_mos_temperature(self):
        names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]

        class FakeMotor:
            def __init__(self, temperature):
                self.temperature = temperature

            def get_state(self):
                return types.SimpleNamespace(pos=0.1, vel=0.0, torq=0.0,
                                             t_mos=self.temperature)

        class FakeArm:
            joint_names = names
            control_loop_active = False
            _motor_map = {name: FakeMotor(40.0 + index) for index, name in enumerate(names)}

            def get_state(self):
                return np.zeros(7), np.zeros(7), np.zeros(7)

        backend = PosVelBackend.__new__(PosVelBackend)
        backend.connected = True
        backend.enabled = False
        backend.stopped = False
        backend.arm = FakeArm()
        state = backend.read_state()
        self.assertEqual(state.temperatures, [40.0 + index for index in range(7)])

    def test_25_mit_control_error_is_a_safety_error(self):
        class FailedController:
            last_error = RuntimeError("control thread died")
            started = True
            last_temperatures = [None] * 7

            def read_joint_angles(self):
                return [0.0] * 7

        backend = RebotMITBackend.__new__(RebotMITBackend)
        backend.controller = FailedController()
        backend.connected = True
        backend.enabled = True
        backend.stopped = False
        with self.assertRaisesRegex(SafetyError, "MIT 控制线程异常"):
            backend.read_state()

    def test_26_ik_respects_soft_limits_directly(self):
        limiter = JointLimiter(self.kin.limits, self.settings.gripper_limits)
        qh = self.settings.home[:6]
        th = self.kin.forward(qh)
        # 施加较大俯仰，验证传入 soft limits 后 IK 解绝不越过软限位。
        target = th.copy()
        target[:3, :3] = Rotation.from_euler("xyz", [0.0, math.radians(60.0), 0.0]).as_matrix() @ th[:3, :3]
        result = self.kin.ik(qh, target, joint_limits=limiter.soft[:, :6])
        self.assertTrue(np.all(result.q >= limiter.soft[0, :6] - 1e-9))
        self.assertTrue(np.all(result.q <= limiter.soft[1, :6] + 1e-9))

    def test_27_low_speed_gpos_joycon_never_deadlocks(self):
        # 复现真机 --max-vel 0.05 场景：持续推摇杆 + 旋转手柄 60 帧，机械臂必须平滑前进且不死锁。
        low_settings = Settings(self.settings.root, copy.deepcopy(self.settings.raw), self.settings.path)
        low_settings.with_max_arm_velocity(0.05)
        driver = FakeJoyConDriver()
        home_pose = self.kin.pose(low_settings.home[:6])
        joy = JoyConInput(low_settings, home_pose[:3], -4.0, 0.0, home_pose[3:], driver=driver)
        joy._started -= 2.0
        backend = FakeBackend(low_settings.home)
        controller = TeleopController(low_settings, backend, joy, self.kin, "gpos")
        controller.start()
        controller.gate.started -= 2.0

        start_tcp = self.kin.forward(low_settings.home[:6])[:3, 3]
        driver.snap.buttons = {"r": True}
        controller._last_time = time.monotonic() - 0.033
        controller.step()  # 先离合切入
        driver.snap.buttons = {"r": True, "x": True}
        driver.snap.stick_xy = np.array([1.0, 0.5])
        driver.snap.rotation = Rotation.from_euler("xyz", [0.0, 0.0, math.radians(35.0)]).as_matrix()
        for _ in range(60):
            driver.snap.last_packet_monotonic = time.monotonic()
            controller._last_time = time.monotonic() - 0.033
            frame = controller.step()
            self.assertTrue(frame.allowed)
        end_tf = self.kin.forward(controller.target[:6])
        end_tcp = end_tf[:3, 3]
        end_yaw_deg = float(np.rad2deg(Rotation.from_matrix(end_tf[:3, :3]).as_euler("xyz")[2]))
        self.assertGreater(float(np.linalg.norm(end_tcp - start_tcp)), 0.010)
        self.assertGreater(end_tcp[2], start_tcp[2])
        self.assertGreater(end_yaw_deg, 2.0)
        controller.close()

    def test_28_pos_vel_falls_back_to_mech_pos(self):
        names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]

        class FakeMotor:
            def __init__(self, position):
                self.position = position

            def get_state(self):
                return None

            def robstride_get_param_f32(self, param, timeout_ms=1000):
                self.last_param = (param, timeout_ms)
                return self.position

        class FakeArm:
            joint_names = names
            control_loop_active = False
            _motor_map = {name: FakeMotor(0.1 * index) for index, name in enumerate(names)}

            def get_state(self):
                return np.zeros(7), np.zeros(7), np.zeros(7)

        backend = PosVelBackend.__new__(PosVelBackend)
        backend.connected = True
        backend.enabled = False
        backend.stopped = False
        backend._fallback_warned = True
        backend.arm = FakeArm()
        state = backend.read_state()
        np.testing.assert_allclose(state.position, [0.1 * index for index in range(7)])
        self.assertTrue(np.isfinite(state.velocity).all())
        self.assertTrue(np.isnan(state.torque).all())
        self.assertEqual(state.temperatures, [None] * 7)
        self.assertEqual(backend.arm._motor_map["joint1"].last_param, (0x7019, 200))

    def test_29_gripper_zero_feedback_tolerance(self):
        limiter = JointLimiter(self.kin.limits, self.settings.gripper_limits)
        q = np.zeros(7)
        q[6] = np.deg2rad(-0.25)
        limiter.validate(q)
        q[6] = np.deg2rad(-2.0)
        with self.assertRaises(SafetyError):
            limiter.validate(q)

    def test_30_split_arm_gripper_pose_reached(self):
        target = np.zeros(7)
        actual = np.zeros(7)
        actual[1] = 0.03  # 1.7° (< 0.04 arm_tol)
        actual[6] = -0.10  # 5.7° (< 0.15 gripper_tol)
        self.assertTrue(pose_reached(actual, target, arm_tolerance_rad=0.04, gripper_tolerance_rad=0.15))
        actual[1] = 0.06  # 3.4° (> 0.04 arm_tol)
        self.assertFalse(pose_reached(actual, target, arm_tolerance_rad=0.04, gripper_tolerance_rad=0.15))

    def test_31_workspace_projection_keeps_z_after_radius_clip(self):
        origin = np.array([0.0, 0.0, 0.145])
        low = project_workspace(make_pose(np.array([0.9, 0.0, 0.01, 0, 0, 0])),
                                radius=0.58, min_z=0.04, max_z=0.65, min_reach=0.10, origin=origin)
        self.assertGreaterEqual(low[2, 3], 0.04 - 1e-9)
        self.assertLessEqual(float(np.linalg.norm(low[:3, 3] - origin)), 0.58 + 1e-9)
        high = project_workspace(make_pose(np.array([0.0, 0.0, 1.2, 0, 0, 0])),
                                 radius=0.58, min_z=0.04, max_z=0.65, min_reach=0.10, origin=origin)
        self.assertLessEqual(high[2, 3], 0.65 + 1e-9)
        near = project_workspace(make_pose(np.array([0.0, 0.0, 0.145, 0, 0, 0])),
                                 radius=0.58, min_z=0.04, max_z=0.65, min_reach=0.10, origin=origin)
        self.assertGreaterEqual(float(np.linalg.norm(near[:3, 3] - origin)), 0.10 - 1e-9)

    def test_32_config_rejects_zero_rate_and_bad_workspace(self):
        raw = copy.deepcopy(self.settings.raw)
        raw["arm"]["control_rate_hz"] = 0
        bad = Settings(self.settings.root, raw, self.settings.path)
        with self.assertRaisesRegex(ValueError, "控制频率"):
            bad.validate()
        raw = copy.deepcopy(self.settings.raw)
        raw["safety"]["workspace_min_reach_m"] = 0.7
        bad = Settings(self.settings.root, raw, self.settings.path)
        with self.assertRaisesRegex(ValueError, "workspace_min_reach_m"):
            bad.validate()
        raw = copy.deepcopy(self.settings.raw)
        raw["safety"]["workspace_min_z_m"] = 0.8
        bad = Settings(self.settings.root, raw, self.settings.path)
        with self.assertRaisesRegex(ValueError, "工作空间 z"):
            bad.validate()

    def test_33_partial_ik_is_not_accepted_for_teleop(self):
        q = np.zeros(6)
        self.assertFalse(acceptable_ik_result(
            IKResult(q, False, True, 0.01, 0.05, 0.01)))
        self.assertTrue(acceptable_ik_result(
            IKResult(q, False, True, 0.01, 0.005, 0.01)))
        self.assertTrue(acceptable_ik_result(
            IKResult(q, True, True, 0.01, 0.0, 0.0)))

    def test_34_keyboard_holding_space_does_not_false_timeout_and_dead_listener_trips(self):
        source = KeyboardInput.__new__(KeyboardInput)
        source._lock = threading.Lock()
        source._home_edge = source._exit_edge = False
        source._listener = types.SimpleNamespace(running=True)
        source._pressed = {"space"}
        source._last_event = time.monotonic() - 5.0  # 5 秒前按下 Space 一直未松开
        sample = source.poll()
        self.assertFalse(sample.streaming_heartbeat)
        gate = SafetyGate(0.25, 0.0, True)
        status = gate.update(sample.deadman, sample.heartbeat, streaming=sample.streaming_heartbeat)
        self.assertTrue(status.allowed)

        # 监听线程意外停止时必须立即触发看门狗并锁存。
        source._listener.running = False
        dead_sample = source.poll()
        self.assertIsNone(dead_sample.heartbeat)
        dead_status = gate.update(dead_sample.deadman, dead_sample.heartbeat,
                                  streaming=dead_sample.streaming_heartbeat)
        self.assertFalse(dead_status.allowed)
        self.assertTrue(gate.fault_latched)

    def test_35_mit_nonfinite_torque_is_unavailable_not_fatal(self):
        class Controller:
            last_temperatures = [None] * 7

            def read_joint_angles(self):
                return [0.0] * 7

            def get_torques(self):
                return [np.nan] * 6 + [1.0]

        backend = RebotMITBackend.__new__(RebotMITBackend)
        backend.controller = Controller()
        backend.connected = True
        backend.enabled = False
        backend.stopped = False
        state = backend.read_state()
        self.assertTrue(np.isnan(state.torque[:6]).all())
        self.assertEqual(state.torque[6], 1.0)

    def test_36_rate_limiter_does_not_overshoot_static_goal(self):
        limiter = JointLimiter(self.kin.limits, self.settings.gripper_limits)
        rate = JointRateLimiter(np.ones(7), np.full(7, 0.2), limiter)
        rate.reset(np.zeros(7))
        goal = np.full(7, 0.05)
        goal[6] = 0.0
        for _ in range(100):
            candidate = rate.step(goal, 0.1)
            self.assertTrue(np.all(candidate <= goal + 1e-12))
        np.testing.assert_allclose(candidate, goal)

    def test_37_stale_feedback_is_rejected(self):
        state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), [30.0] * 7, 10.0)
        with self.assertRaisesRegex(SafetyError, "陈旧"):
            validate_feedback(state, self.settings.section("safety"), now=11.0, started=0.0)

    def test_38_missing_gripper_torque_is_rejected_in_strict_mode(self):
        state = ArmState(np.zeros(7), np.zeros(7), np.full(7, np.nan), [30.0] * 7, 10.0)
        with self.assertRaisesRegex(SafetyError, "力矩反馈"):
            validate_feedback(state, self.settings.section("safety"), now=10.0, started=0.0)

    def test_39_missing_temperature_is_rejected_after_grace(self):
        state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), [None] * 7, 10.0)
        validate_feedback(state, self.settings.section("safety"), now=10.0, started=9.0)
        state.timestamp = 15.0
        with self.assertRaisesRegex(SafetyError, "温度反馈"):
            validate_feedback(state, self.settings.section("safety"), now=15.0, started=0.0)

    def test_40_mit_read_only_enable_is_refused(self):
        backend = RebotMITBackend.__new__(RebotMITBackend)
        backend.controller = Mock()
        with self.assertRaisesRegex(SafetyError, "拒绝连接"):
            backend.connect(enable=False)
        backend.controller.connect.assert_not_called()
        backend.controller.disable_motors.assert_not_called()

    def test_41_pos_vel_partial_enable_never_disconnects(self):
        backend = PosVelBackend.__new__(PosVelBackend)
        backend.arm = Mock()
        backend.arm.arm.mode_pos_vel.return_value = True
        backend.arm.has_gripper = False
        backend.arm.enable_all.side_effect = RuntimeError("partial enable")
        backend.read_state = Mock(return_value=ArmState(
            np.zeros(7), np.zeros(7), np.zeros(7), [30.0] * 7, time.monotonic()))
        backend.connected = False
        backend.enabled = False
        backend.stopped = False
        with self.assertRaisesRegex(SafetyError, "禁止直接 disconnect"):
            backend.connect(enable=True)
        backend.arm.disconnect.assert_not_called()
        self.assertTrue(backend.connected)
        self.assertTrue(backend.enabled)

    def test_42_mujoco_view_locks_during_qpos_update(self):
        events = []

        class FakeLock:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *args):
                events.append("exit")

        class FakeViewer:
            def is_running(self):
                return True

            def lock(self):
                return FakeLock()

            def sync(self):
                events.append("sync")

        view = MuJoCoView(self.settings, headless=False)
        view.model = object()
        view.data = types.SimpleNamespace(qpos=np.zeros(8))
        view._arm_addresses = list(range(6))
        view._finger_targets = [(6, 0.05), (7, 0.0715)]
        view.mujoco = types.SimpleNamespace(
            mj_forward=lambda m, d: events.append("mj_forward"))
        view.viewer = FakeViewer()
        view.update(self.settings.home)
        self.assertEqual(events, ["enter", "mj_forward", "exit", "sync"])

    def test_43_home_waits_for_actual_feedback(self):
        class LagBackend(FakeBackend):
            def command(self, q):
                self.target = np.asarray(q).copy()  # 模拟反馈滞后，不立即更新 position

        backend = LagBackend(np.zeros(7))
        source = ScriptedInput()
        controller = TeleopController(self.settings, backend, source, self.kin, "qpos")
        controller.start()
        controller.gate.started -= 2
        controller.home_active = True
        controller.target = self.settings.home.copy()
        controller.rate.reset(self.settings.home)
        controller.step()  # Fake 反馈仍是零位，不能因内部目标到位就结束 HOME。
        self.assertTrue(controller.home_active)
        self.assertFalse(controller.gate.fault_latched)

    def test_44_config_rejects_disabled_safety_and_excessive_limits(self):
        for key, value in (("require_deadman", False), ("require_imu_heartbeat", False),
                           ("temperature_trip_c", 100.0), ("gripper_torque_trip_nm", 2.0),
                           ("workspace_radius_m", 0.85), ("joint_limit_scale", 0.9)):
            raw = copy.deepcopy(self.settings.raw)
            raw["safety"][key] = value
            with self.assertRaises(ValueError):
                Settings(self.settings.root, raw, self.settings.path).validate()

    def test_45_missing_hardware_timestamp_requires_explicit_policy(self):
        state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), [30.0] * 7, None)
        validate_feedback(state, self.settings.section("safety"), now=10.0, started=0.0)
        raw = copy.deepcopy(self.settings.raw)
        raw["safety"]["require_feedback_timestamp"] = True
        with self.assertRaisesRegex(SafetyError, "时间戳"):
            validate_feedback(state, raw["safety"], now=10.0, started=0.0)

    def test_46_arm_zero_feedback_tolerance(self):
        limiter = JointLimiter(self.kin.limits, self.settings.gripper_limits)
        q = np.zeros(7)
        q[1] = np.deg2rad(0.03)
        q[2] = np.deg2rad(-0.11)
        limiter.validate(q)
        np.testing.assert_allclose(limiter.clamp(q)[2], 0.0)
        q[2] = np.deg2rad(-2.0)
        with self.assertRaises(SafetyError):
            limiter.validate(q)

    def test_47_mit_motorbridge_cache_provides_gripper_torque(self):
        class Motor:
            def __init__(self, index):
                self.index = index

            def get_state(self):
                if self.index == 5:
                    raise RuntimeError("模拟该路反馈尚未填充")
                return types.SimpleNamespace(vel=0.01 * self.index,
                                             torq=0.1 * self.index)

        class Controller:
            last_temperatures = [30.0] * 7
            motors = [Motor(index) for index in range(7)]

            def read_joint_angles(self):
                return [0.0] * 7

        backend = RebotMITBackend.__new__(RebotMITBackend)
        backend.controller = Controller()
        backend.connected = True
        backend.enabled = True
        backend.stopped = False
        backend._prefer_cached_position = False
        state = backend.read_state()
        self.assertAlmostEqual(state.torque[6], 0.6)
        self.assertAlmostEqual(state.velocity[6], 0.06)
        self.assertTrue(np.isnan(state.torque[5]))
        self.assertEqual(state.velocity[5], 0.0)

    def test_48_home_on_start_waits_for_first_deadman(self):
        source = ScriptedInput([InputSample(False, None) for _ in range(3)])
        backend = FakeBackend(np.zeros(7))
        controller = TeleopController(self.settings, backend, source, self.kin, "qpos")
        controller.start()
        controller.home_pending = True
        controller.gate.started -= 2
        for _ in range(3):
            controller.step()
        self.assertTrue(controller.home_pending)
        self.assertFalse(controller.home_active)
        np.testing.assert_allclose(controller.target, np.zeros(7), atol=1e-9)
        controller.step()
        self.assertFalse(controller.home_pending)
        self.assertTrue(controller.home_active)
        self.assertGreater(controller.target[1], 0.0)
        controller.close()

    def test_49_second_ctrl_c_emergency_disables_mit(self):
        backend = RebotMITBackend.__new__(RebotMITBackend)
        backend.connected = True
        backend.enabled = True
        backend.stopped = False
        backend.controller = Mock()
        backend.settings = self.settings
        backend.read_state = Mock(return_value=ArmState(
            np.zeros(7), np.zeros(7), np.zeros(7), [30.0] * 7, time.monotonic()))
        with patch("rebot_joycon.backends.time.sleep", side_effect=KeyboardInterrupt):
            backend.safe_stop()
        backend.controller.stop.assert_called_once_with(return_to_zero=False, wait=True)
        self.assertTrue(backend.stopped)
        self.assertFalse(backend.enabled)
        self.assertFalse(backend.connected)

    def test_50_mit_cached_position_is_checked_against_sdk(self):
        angles = [0.0, 10.0, -10.0, 5.0, -5.0, 90.0, 45.0]

        class Motor:
            def __init__(self, pos):
                self.pos = pos

            def get_state(self):
                return types.SimpleNamespace(pos=self.pos, vel=0.0, torq=0.0)

        class Controller:
            last_temperatures = [30.0] * 7

            def __init__(self, offset=0.0):
                self.motors = [Motor(np.deg2rad(angle) + offset) for angle in angles]

            def read_joint_angles(self):
                return list(angles)

        backend = RebotMITBackend.__new__(RebotMITBackend)
        backend.controller = Controller()
        backend._calibrate_position_source()
        self.assertTrue(backend._prefer_cached_position)
        np.testing.assert_allclose(backend.read_state().position,
                                   np.deg2rad(angles), atol=1e-9)
        backend.controller = Controller(offset=np.deg2rad(5.0))
        backend._calibrate_position_source()
        self.assertFalse(backend._prefer_cached_position)
        np.testing.assert_allclose(backend.read_state().position,
                                   np.deg2rad(angles), atol=1e-9)

    def test_51_joycon_stationary_packet_heartbeat_and_timeout(self):
        driver = FakeJoyConDriver()
        home_pose = self.kin.pose(self.settings.home[:6])
        joy = JoyConInput(self.settings, home_pose[:3], -4.0, 0.0, home_pose[3:], driver=driver)
        gate = SafetyGate(0.25, 0.0, True)
        driver.snap.buttons = {"r": True}
        driver.snap.last_packet_monotonic = time.monotonic()
        sample = joy.poll()
        self.assertTrue(gate.update(sample.deadman, sample.heartbeat,
                                    streaming=sample.streaming_heartbeat).allowed)
        # 模拟蓝牙断连导致数据包停止超过 0.25s。
        driver.snap.last_packet_monotonic = time.monotonic() - 1.0
        stale = joy.poll()
        status = gate.update(stale.deadman, stale.heartbeat, streaming=stale.streaming_heartbeat)
        self.assertFalse(status.allowed)
        self.assertTrue(gate.fault_latched)

    def test_52_gravity_compensation_cli_parses_without_typeerror(self):
        from gravity_compensation import run_gravity_compensation as gc
        with patch("builtins.input", return_value="NO"):
            rc = gc.main([])
        self.assertEqual(rc, 0)

    def test_53_gravity_compensation_verified_state_rejects_missing_cache(self):
        from gravity_compensation import run_gravity_compensation as gc

        class EmptyMotor:
            def get_state(self):
                return None

        class DummyArm:
            joint_names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
            _motor_map = {name: EmptyMotor() for name in joint_names}

            def get_state(self):
                return np.zeros(7), np.zeros(7), np.zeros(7)

        with self.assertRaisesRegex(RuntimeError, "反馈缺失"):
            gc._read_verified_arm_state(DummyArm())

    def test_54_ik_yaw_and_coordinated_straight_line_motion(self):
        limiter = JointLimiter(self.kin.limits, self.settings.gripper_limits)
        qh = self.settings.home[:6]
        th = self.kin.forward(qh)
        for yaw_deg in (24.0, -24.0):
            target = th.copy()
            target[:3, :3] = Rotation.from_euler("xyz", [0.0, 0.0, math.radians(yaw_deg)]).as_matrix() @ th[:3, :3]
            res = self.kin.ik(qh, target, joint_limits=limiter.soft[:, :6])
            self.assertTrue(res.success)
            self.assertLess(res.position_error, 1e-3)
            self.assertLess(res.orientation_error, 1e-2)

        # 验证协调 6 轴限速下纯 +X 推进不发生严重 Z 轴漂移。
        driver = FakeJoyConDriver()
        backend = FakeBackend(self.settings.home)
        home_pose = self.kin.pose(self.settings.home[:6])
        joy = JoyConInput(self.settings, home_pose[:3], -4.0, 0.0, home_pose[3:], driver=driver)
        joy._started -= 2.0
        ctrl = TeleopController(self.settings, backend, joy, self.kin, "gpos")
        ctrl.start()
        ctrl.gate.started -= 2.0
        p0 = self.kin.forward(backend.position[:6])[:3, 3]
        driver.snap.buttons = {"r": True}
        driver.snap.stick_xy = np.array([1.0, 0.0])
        for _ in range(30):
            driver.snap.last_packet_monotonic = time.monotonic()
            ctrl._last_time = time.monotonic() - 0.033
            ctrl.step()
        p1 = self.kin.forward(backend.position[:6])[:3, 3]
        delta = p1 - p0
        self.assertGreater(delta[0], 0.04)
        self.assertLess(abs(delta[2]), 0.01)
        ctrl.close()

    def test_55_headless_mujoco_view_skips_missing_mujoco(self):
        with patch.dict(sys.modules, {"mujoco": None}):
            with MuJoCoView(self.settings, headless=True) as view:
                self.assertIsNone(view.model)
                self.assertTrue(view.running)
                view.update(self.settings.home)
            with self.assertRaisesRegex(RuntimeError, "未安装 mujoco"):
                with MuJoCoView(self.settings, headless=False):
                    pass

    def test_56_calibrate_stationary_skips_zero_warmup_and_publishes_rotation(self):
        from rebot_joycon.joycon_driver import NativeJoyConDriver

        zero_pkt = _build_report_0x30(accel=(0, 0, 0), gyro=(0, 0, 0))
        # 模拟平放微倾 Joy-Con（ax=+400, az=-4096）且存在固定陀螺零偏 (18, -31, -12)。
        tilted_pkt = _build_report_0x30(
            stick_r=(2100, 1810),
            accel=(400, 0, -4096),
            gyro=(18, -31, -12),
        )
        packets = [zero_pkt] + [tilted_pkt] * 12
        idx = {"i": 0}

        class FakeHandle:
            def read(self, size=49, timeout_ms=30):
                i = idx["i"]
                idx["i"] += 1
                if i < len(packets):
                    return packets[i]
                time.sleep(0.01)
                return tilted_pkt

        drv = NativeJoyConDriver.__new__(NativeJoyConDriver)
        drv.side = "right"
        drv.is_right = True
        drv.stick_deadzone = 0.15
        drv.stick_axis_signs = (1.0, 1.0)
        drv._accel_offset = np.zeros(3)
        drv._accel_scale = np.ones(3)
        drv._gyro_offset = np.zeros(3)
        drv._gyro_scale = np.ones(3)
        drv._stick_center = np.array([2048.0, 2048.0])
        drv._filter = QuaternionAttitudeFilter(is_right=True)
        drv._lock = threading.Lock()
        drv._snapshot = JoyConSnapshot()
        drv._handle = FakeHandle()

        drv._calibrate_stationary(0.12)
        np.testing.assert_allclose(drv._gyro_offset, [18.0, -31.0, -12.0], atol=1e-6)
        snap = drv.snapshot()
        # 标定结束瞬间的 snapshot.rotation 必须已经是倾斜后的实际姿态，而不是单位矩阵。
        self.assertGreater(float(np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(snap.rotation).as_rotvec()))), 3.0)

    def test_57_fake_backend_survives_past_feedback_grace(self):
        source = ScriptedInput([InputSample(True, None)])
        backend = FakeBackend(self.settings.home)
        controller = TeleopController(self.settings, backend, source, self.kin, "gpos")
        controller.start()
        controller.gate.started -= 10.0
        controller.feedback_started -= 10.0  # 越过 feedback_grace_s=5.0s
        frame = controller.step()
        self.assertTrue(frame.allowed)
        self.assertEqual(frame.state.temperatures, [25.0] * 7)
        controller.close()

    def test_58_idle_hold_does_not_ratchet_gravity_sag(self):
        # 复现无重力补偿 MIT PD 在 HOME 悬空约 3° 静差下，松开使能不应把 target 往下拉导致越掉越低。
        class DroopyBackend(FakeBackend):
            def read_state(self):
                # 模拟重力使 J2 下沉 +2.5°, J3 下垂 -3.1°, J4 下垂 -2.8°
                sagged = self.target.copy()
                sagged[1] += math.radians(2.5)
                sagged[2] -= math.radians(3.1)
                sagged[3] -= math.radians(2.8)
                return ArmState(sagged, np.zeros(7), np.zeros(7), [30.0] * 7, time.monotonic())

        samples = [InputSample(False, None) for _ in range(15)] + [InputSample(True, None)]
        source = ScriptedInput(samples)
        backend = DroopyBackend(self.settings.home)
        controller = TeleopController(self.settings, backend, source, self.kin, "gpos")
        controller.start()
        # start() 从实际姿态起步，随后手动模拟已规划到 HOME 的保持目标。
        controller.target = self.settings.home.copy()
        controller.rate.reset(self.settings.home)
        controller.gate.started -= 2.0

        # 松开使能连续 15 帧，command 必须死死守住 settings.home，绝不往下滑。
        for _ in range(15):
            controller._last_time = time.monotonic() - 0.033
            frame = controller.step()
            self.assertFalse(frame.allowed)
            np.testing.assert_allclose(frame.target, self.settings.home, atol=1e-9)

        # 重新按下使能时同样保持 settings.home，不瞬间丢掉 PD 保持力矩。
        controller._last_time = time.monotonic() - 0.033
        active = controller.step()
        self.assertTrue(active.allowed)
        np.testing.assert_allclose(active.target, self.settings.home, atol=1e-6)
        controller.close()

    def test_59_home_completes_under_mit_static_droop_and_boundary_ik_accepted(self):
        class DroopyHomeBackend(FakeBackend):
            def read_state(self):
                sagged = self.target.copy()
                sagged[2] -= math.radians(3.2)  # 实测 MIT 伸展态 J3 约 3.2° 重力静差
                return ArmState(sagged, np.zeros(7), np.zeros(7), [30.0] * 7, time.monotonic())

        samples = [InputSample(True, None, home=True)] + [InputSample(True, None) for _ in range(10)]
        source = ScriptedInput(samples)
        backend = DroopyHomeBackend(self.settings.home)
        controller = TeleopController(self.settings, backend, source, self.kin, "gpos")
        controller.start()
        controller.target = self.settings.home.copy()
        controller.rate.reset(self.settings.home)
        controller.gate.started -= 2.0
        controller.step()
        self.assertFalse(controller.home_active)
        self.assertTrue(controller.gate.fault_latched)

        # 验证 z=0.04m 边界附近因 IK 1e-4 收敛精度导致的 0.05mm 误差不会被 _workspace_ok 误拒。
        p = self.kin.forward(self.settings.home[:6]).copy()
        p[2, 3] = 0.03995
        q_boundary = self.kin.ik(self.settings.home[:6], p, joint_limits=controller.joints.soft[:, :6]).q
        self.assertTrue(controller._workspace_ok(q_boundary))
        controller.close()

    def test_60_body_frame_pitch_is_invariant_to_desk_yaw_and_arm_yaw(self):
        # 机械臂基座已左转 J1=40°，且手柄校准桌面朝向偏转 90° 时，俯仰手柄只能引起末端纯局部俯仰，不能串成横滚。
        driver = FakeJoyConDriver()
        calib_rot = Rotation.from_euler("xyz", [0.0, 0.25, math.radians(90.0)]).as_matrix()
        driver.snap.rotation = calib_rot.copy()

        start_q = self.settings.home.copy()
        start_q[0] = math.radians(40.0)
        joy = JoyConInput(self.settings, self.kin.pose(start_q[:6])[:3], -4.0, 0.0, driver=driver)
        joy._started -= 2.0
        backend = FakeBackend(start_q)
        controller = TeleopController(self.settings, backend, joy, self.kin, "gpos")
        controller.start()
        controller.gate.started -= 2.0

        start_rot = self.kin.forward(start_q[:6])[:3, :3]
        driver.snap.buttons = {"r": True}
        controller._last_time = time.monotonic() - 0.033
        controller.step()  # 离合切入

        # 绕手柄自身体坐标 +Y 轴缓慢俯仰 +15°（分 15 帧），再保持 20 帧让加速度限幅后的关节平滑到位。
        for step_idx in range(1, 36):
            pitch = math.radians(min(step_idx, 15) * 1.0)
            driver.snap.rotation = calib_rot @ Rotation.from_euler("xyz", [0.0, pitch, 0.0]).as_matrix()
            driver.snap.last_packet_monotonic = time.monotonic()
            controller._last_time = time.monotonic() - 0.033
            frame = controller.step()
            self.assertTrue(frame.allowed)

        end_rot = self.kin.forward(controller.target[:6])[:3, :3]
        local_rpy_deg = np.rad2deg(Rotation.from_matrix(start_rot.T @ end_rot).as_euler("xyz"))
        self.assertGreater(local_rpy_deg[1], 12.0)  # 局部俯仰平滑接近目标 15°
        self.assertLess(abs(local_rpy_deg[0]), 1.5)  # 几乎无局部横滚串扰
        self.assertLess(abs(local_rpy_deg[2]), 1.5)  # 几乎无局部偏航串扰
        controller.close()

    def test_61_cyclic_3d_rotation_has_zero_holonomy_and_clipping_drift(self):
        # 复现 --max-vel 0.12 且开启 imu_yaw_scale=0.8 时，三维反复晃动手腕再回到原位，末端绝不能持续朝单侧漂移。
        low_settings = Settings(self.settings.root, copy.deepcopy(self.settings.raw), self.settings.path)
        low_settings.with_max_arm_velocity(0.12)
        driver = FakeJoyConDriver()
        calib_rot = Rotation.from_euler("xyz", [0.0, 0.25, math.radians(90.0)]).as_matrix()
        driver.snap.rotation = calib_rot.copy()
        home_pose = self.kin.pose(low_settings.home[:6])
        close_rad, open_rad = low_settings.gripper_close_open
        joy = JoyConInput(low_settings, home_pose[:3], open_rad, close_rad, home_pose[3:], driver=driver)
        joy._started -= 2.0
        backend = FakeBackend(low_settings.home)
        controller = TeleopController(low_settings, backend, joy, self.kin, "gpos")
        controller.start()
        controller.gate.started -= 2.0

        driver.snap.buttons = {"r": True}
        controller._last_time = time.monotonic() - 0.033
        controller.step()  # 离合切入

        # 在含各轴耦合的三维 ±25° 周期摆动下刻意触发 tether 限幅，然后手柄回正。
        for i in range(300):
            ang = math.radians(25.0) * math.sin(2.0 * math.pi * 0.5 * (i * 0.033))
            driver.snap.rotation = calib_rot @ Rotation.from_euler("xyz", [ang, 0.7 * ang, 0.5 * ang]).as_matrix()
            driver.snap.last_packet_monotonic = time.monotonic()
            controller._last_time = time.monotonic() - 0.033
            controller.step()

        driver.snap.rotation = calib_rot.copy()
        for _ in range(120):
            driver.snap.last_packet_monotonic = time.monotonic()
            controller._last_time = time.monotonic() - 0.033
            controller.step()

        end_tf = self.kin.forward(controller.target[:6])
        end_rpy_deg = np.rad2deg(Rotation.from_matrix(end_tf[:3, :3]).as_euler("xyz"))
        np.testing.assert_allclose(end_tf[:3, 3], home_pose[:3], atol=1e-4)
        self.assertLess(float(np.max(np.abs(end_rpy_deg))), 0.1)
        controller.close()

    def test_62_rs_gripper_positive_open_limits_and_simulator_mapping(self):
        close_rad, open_rad = self.settings.gripper_close_open
        self.assertAlmostEqual(close_rad, 0.0, places=9)
        self.assertGreater(open_rad, 4.0)

        # 验证 ZR 在 RS 正向夹爪行程下能从闭合零位平滑张开到 > 2.0 rad，再次点按回到 0。
        driver = FakeJoyConDriver()
        home_pose = self.kin.pose(self.settings.home[:6])
        joy = JoyConInput(self.settings, home_pose[:3], open_rad, close_rad, home_pose[3:], driver=driver)
        joy._started -= 2.0
        init_q = self.settings.home.copy()
        init_q[6] = math.radians(-0.22)  # 复现真机闭合零点编码器微负偏
        backend = FakeBackend(init_q)
        controller = TeleopController(self.settings, backend, joy, self.kin, "gpos")
        controller.start()
        controller.gate.started -= 2.0

        driver.snap.buttons = {"r": True}
        controller._last_time = time.monotonic() - 0.033
        controller.step()
        driver.snap.buttons = {"r": True, "zr": True}
        controller._last_time = time.monotonic() - 0.033
        controller.step()
        driver.snap.buttons = {"r": True}
        for _ in range(70):
            driver.snap.last_packet_monotonic = time.monotonic()
            controller._last_time = time.monotonic() - 0.033
            frame = controller.step()
        self.assertGreater(frame.target[6], 2.0)

        driver.snap.buttons = {"r": True, "zr": True}
        controller._last_time = time.monotonic() - 0.033
        controller.step()
        driver.snap.buttons = {"r": True}
        for _ in range(120):
            driver.snap.last_packet_monotonic = time.monotonic()
            controller._last_time = time.monotonic() - 0.033
            frame = controller.step()
        self.assertAlmostEqual(frame.target[6], 0.0, places=3)

        # 验证 MuJoCo 视图同步：0 闭合 -> 0，+open_rad 张开 -> full_open。
        qpos = np.zeros(8)
        view = MuJoCoView.__new__(MuJoCoView)
        view.settings = self.settings
        view.model = object()
        view.data = types.SimpleNamespace(qpos=qpos)
        view.mujoco = types.SimpleNamespace(mj_forward=lambda _m, _d: None)
        view._arm_addresses = list(range(6))
        view._finger_targets = [(6, 0.05), (7, 0.0715)]
        view._apply_qpos(np.zeros(7))
        self.assertAlmostEqual(float(qpos[6]), 0.0, places=9)
        self.assertAlmostEqual(float(qpos[7]), 0.0, places=9)
        q_open = np.zeros(7)
        q_open[6] = open_rad
        view._apply_qpos(q_open)
        self.assertAlmostEqual(float(qpos[6]), 0.05, places=9)
        self.assertAlmostEqual(float(qpos[7]), 0.0715, places=9)
        controller.close()

    def test_63_gripper_initial_open_sync_and_torque_hold_immediate_release(self):
        close_rad, open_rad = self.settings.gripper_close_open
        driver = FakeJoyConDriver()
        home_pose = self.kin.pose(self.settings.home[:6])
        joy = JoyConInput(self.settings, home_pose[:3], open_rad, close_rad, home_pose[3:], driver=driver)
        joy._started -= 2.0

        # 启动时若物理夹爪原本处于张开 +3.5 rad，首次按住 R 不能误把夹爪甩回 0；点按 ZR 后应改为闭合。
        start_open = self.settings.home.copy()
        start_open[6] = 3.5
        backend = FakeBackend(start_open)
        controller = TeleopController(self.settings, backend, joy, self.kin, "gpos")
        controller.start()
        controller.gate.started -= 2.0

        driver.snap.buttons = {"r": True}
        for _ in range(5):
            driver.snap.last_packet_monotonic = time.monotonic()
            controller._last_time = time.monotonic() - 0.033
            frame = controller.step()
        self.assertAlmostEqual(frame.target[6], 3.5, places=6)

        # 点按 ZR 闭合时若在中途 +1.8 rad 夹住物体（力矩 -1.5 Nm），再按 ZR 张开须立即向上松开，不滞后。
        driver.snap.buttons = {"r": True, "zr": True}
        controller._last_time = time.monotonic() - 0.033
        controller.step()
        driver.snap.buttons = {"r": True}

        def read_pinched():
            pos = backend.position.copy()
            torq = np.zeros(7)
            if pos[6] <= 1.85:
                pos[6] = 1.8
                torq[6] = -1.5
            return ArmState(pos, np.zeros(7), torq, [30.0] * 7, time.monotonic())

        backend.read_state = read_pinched
        for _ in range(60):
            driver.snap.last_packet_monotonic = time.monotonic()
            controller._last_time = time.monotonic() - 0.033
            frame = controller.step()
        self.assertIsNotNone(controller.gripper_protection.hold)
        held = float(frame.target[6])
        self.assertAlmostEqual(held, 1.8, places=2)

        # 恢复正常读数并按 ZR 要求张开：必须立刻摆脱 held 向上运动，不被 rate.position 滞后拖住。
        backend.read_state = lambda: ArmState(backend.position.copy(), np.zeros(7), np.zeros(7),
                                              [30.0] * 7, time.monotonic())
        driver.snap.buttons = {"r": True, "zr": True}
        controller._last_time = time.monotonic() - 0.033
        release_frame = controller.step()
        self.assertIsNone(controller.gripper_protection.hold)
        self.assertGreater(release_frame.target[6], held)
        controller.close()

    def test_64_calibrate_stationary_rejects_shaky_pickup_bias(self):
        from rebot_joycon.joycon_driver import NativeJoyConDriver

        quiet_pkt = _build_report_0x30(stick_r=(2040, 1815), accel=(0, 0, -4096), gyro=(18, -31, -12))
        shake_pos = _build_report_0x30(stick_r=(2100, 1820), accel=(0, 0, -4096), gyro=(400, -500, 700))
        shake_neg = _build_report_0x30(stick_r=(2000, 1810), accel=(0, 0, -4096), gyro=(-200, 300, -100))

        def run_cal(packets, factory_gyro):
            idx = {"i": 0}

            class FakeHandle:
                def read(self, size=49, timeout_ms=30):
                    i = idx["i"]
                    idx["i"] += 1
                    if i < len(packets):
                        return packets[i]
                    time.sleep(0.01)
                    return packets[-1]

            drv = NativeJoyConDriver.__new__(NativeJoyConDriver)
            drv.side = "right"
            drv.is_right = True
            drv.stick_deadzone = 0.18
            drv.stick_axis_signs = (1.0, 1.0)
            drv._accel_offset = np.zeros(3)
            drv._accel_scale = np.ones(3)
            drv._gyro_offset = np.array(factory_gyro, dtype=float)
            drv._gyro_scale = np.ones(3)
            drv._stick_center = np.array([2048.0, 2048.0])
            drv._filter = QuaternionAttitudeFilter(is_right=True)
            drv._lock = threading.Lock()
            drv._snapshot = JoyConSnapshot()
            drv._handle = FakeHandle()
            drv._calibrate_stationary(0.12)
            return drv._gyro_offset

        # 场景 1：前段静置、后段提前拿起晃动，必须选中前段最静窗口零偏，不被晃动拉偏。
        partial_shake = [quiet_pkt] * 16 + [shake_pos, shake_neg] * 10
        offset1 = run_cal(partial_shake, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(offset1, [18.0, -31.0, -12.0], atol=1e-6)

        # 场景 2：标定全程持续晃动时保留 SPI 出厂零偏，避免整局产生恒定漂移速度。
        full_shake = [shake_pos, shake_neg] * 15
        offset2 = run_cal(full_shake, [16.0, -31.0, -8.0])
        np.testing.assert_allclose(offset2, [16.0, -31.0, -8.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
