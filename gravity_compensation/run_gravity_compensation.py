#!/usr/bin/env python3
"""RS 官方 Pinocchio 重力前馈；Space 按住使能，释放后验证实际反馈回零再断开。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebot_joycon.backends import ArmState, _prefer_environment_packages
from rebot_joycon.config import Settings
from rebot_joycon.inputs import KeyboardInput
from rebot_joycon.kinematics import RSKinematics
from rebot_joycon.safety import (JointLimiter, SafetyGate, check_temperatures,
                                 validate_feedback)


def _read_verified_arm_state(arm) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float | None]]:
    """逐路读取电机真实反馈；不把缺失缓存的 0 误认为已回零。"""
    q, velocity, torque = arm.get_state()
    q = np.asarray(q, dtype=float).copy()
    velocity = np.asarray(velocity, dtype=float).copy()
    torque = np.asarray(torque, dtype=float).copy()
    temperatures: list[float | None] = [None] * 7
    motors = getattr(arm, "_motor_map", {})
    joint_names = list(getattr(arm, "joint_names", []))
    for index, name in enumerate(joint_names[:7]):
        motor = motors.get(name)
        motor_state = None if motor is None else motor.get_state()
        if motor_state is not None:
            q[index] = float(motor_state.pos)
            velocity[index] = float(motor_state.vel)
            torque[index] = float(motor_state.torq)
            temperatures[index] = next(
                (float(getattr(motor_state, attr)) for attr in
                 ("t_mos", "mos_temperature", "temperature", "temp")
                 if getattr(motor_state, attr, None) is not None), None)
        else:
            get_param = getattr(motor, "robstride_get_param_f32", None)
            if get_param is None:
                raise RuntimeError(f"重力补偿反馈缺失：{name}，禁止将 0 当作有效反馈")
            q[index] = float(get_param(0x7019, timeout_ms=200))
            velocity[index] = 0.0
            torque[index] = np.nan
    if q.shape != (7,) or velocity.shape != (7,) or torque.shape != (7,) or not np.isfinite(q).all():
        raise RuntimeError("重力补偿反馈维度或数值无效")
    return q, velocity, torque, temperatures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)
    if not args.yes and input("清场并扶稳机械臂，输入 GC 继续：").strip() != "GC":
        return 0
    sdk_path = settings.section("arm").get("sdk_path")
    if not sdk_path:
        raise RuntimeError("重力补偿需要在 arm.sdk_path 配置官方 reBotArm_control_py 路径")
    _prefer_environment_packages()
    sys.path.insert(0, str(settings.resolve(sdk_path)))
    from reBotArm_control_py.actuator import RebotArm, load_gravity_compensation_config
    from reBotArm_control_py.controllers import GravityCompensation

    keyboard = KeyboardInput(settings)
    arm = RebotArm(str(settings.hardware_yaml))
    config = load_gravity_compensation_config("basic", str(settings.hardware_yaml))
    ctrl = GravityCompensation(arm, kp=float(config.get("kp", 15.0)),
                               kd=float(config.get("kd", 2.5)),
                               tau_scale=config.get("tau_scale", 1.0),
                               transition_duration=float(config.get("transition_duration", 0.5)),
                               log_every=int(config.get("log_every", 20)))
    kin = RSKinematics(settings.urdf_path)
    safety = settings.section("safety")
    limiter = JointLimiter(kin.limits, settings.gripper_limits,
                           float(safety.get("joint_limit_scale", 0.7)))
    started = False
    gate = SafetyGate(float(safety.get("input_watchdog_s", 0.25)),
                      float(safety.get("startup_guard_s", 1.0)), True,
                      streaming_heartbeat=False)
    feedback_started = time.monotonic()
    zero_tol = np.deg2rad(1.0)
    try:
        print("按住 Space 才会启动补偿；松开 Space 或按 Esc 即平滑回零。")
        while True:
            sample = keyboard.poll()
            if sample.exit:
                break
            if gate.update(sample.deadman, sample.heartbeat,
                           streaming=sample.streaming_heartbeat).allowed:
                break
            time.sleep(0.02)
        else:
            return 0
        if sample.exit:
            return 0
        ctrl.start()
        started = True
        feedback_started = time.monotonic()
        while True:
            sample = keyboard.poll()
            now = time.monotonic()
            if (sample.exit or not gate.update(
                    sample.deadman, sample.heartbeat, now=now,
                    streaming=sample.streaming_heartbeat).allowed):
                break
            q, velocity, torque, temperatures = _read_verified_arm_state(arm)
            state = ArmState(q, velocity, torque, temperatures, now)
            validate_feedback(state, safety, now=now, started=feedback_started)
            check_temperatures(temperatures, float(safety["temperature_warn_c"]),
                               float(safety["temperature_trip_c"]))
            # 只检查六轴，且保留 1° 零位容差，避免折叠位 -0.1° 误触发退出。
            if (np.any(q[:6] < limiter.soft[0, :6] - zero_tol)
                    or np.any(q[:6] > limiter.soft[1, :6] + zero_tol)):
                print("[安全] 超出六轴软限位，停止重力补偿")
                break
            if np.isfinite(torque[6]) and abs(float(torque[6])) > float(safety["gripper_torque_trip_nm"]):
                raise RuntimeError("夹爪过力矩，停止重力补偿")
            time.sleep(0.03)
    finally:
        keyboard.close()
        if started:
            end_error = None
            try:
                ctrl.end()
            except Exception as exc:
                # end() 失败也必须继续尝试安全回零；不要在弯曲姿态直接失能。
                end_error = exc
                print(f"[安全] 停止重力补偿失败，继续安全回零：{exc}")
            try:
                ctrl.safe_home(max_vel=0.25, timeout=float(safety["home_timeout_s"]))
                q, _, _, _ = _read_verified_arm_state(arm)
                if np.max(np.abs(q[:6])) >= float(safety.get("home_tolerance_rad", 0.04)):
                    raise RuntimeError(
                        "六轴回零后反馈仍未到位：保持使能，禁止直接 disconnect/disable_all")
                arm.disconnect()
            except Exception:
                if end_error is not None:
                    print(f"[安全] 原始停止错误：{end_error}")
                raise
            if end_error is not None:
                raise end_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
