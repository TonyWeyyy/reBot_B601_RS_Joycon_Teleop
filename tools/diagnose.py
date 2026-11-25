#!/usr/bin/env python3
"""B601-RS 一键软硬件体检工具（只读、绝不使能电机）。"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebot_joycon.config import Settings
from rebot_joycon.joycon_driver import NativeJoyConDriver, enumerate_joycons
from rebot_joycon.kinematics import RSKinematics
from rebot_joycon.safety import JointLimiter, workspace_contains


def _check_can(channel: str = "can0") -> tuple[bool, str]:
    oper = Path(f"/sys/class/net/{channel}/operstate")
    if not oper.exists():
        return False, f"{channel} 接口不存在（请检查 PCAN 连接并运行 bash tools/setup_can.sh {channel}）"
    state = oper.read_text(encoding="utf-8").strip()
    return state == "up", f"{channel} operstate={state}"


def _check_sdk(settings: Settings) -> list[tuple[str, bool, str]]:
    results = []
    for label, key, modules in (
        ("MIT SDK", "rebot_sdk_path", ("rebot_control", "rebot")),
        ("POS_VEL SDK", "sdk_path", ("reBotArm_control_py",)),
    ):
        raw = settings.section("arm").get(key)
        if not raw:
            results.append((label, False, f"未配置 arm.{key}"))
            continue
        resolved = settings.resolve(raw)
        if not resolved.is_dir():
            results.append((label, False, f"目录不存在：{resolved}"))
            continue
        sys.path.insert(0, str(resolved))
        found = [m for m in modules if importlib.util.find_spec(m) is not None]
        if found:
            results.append((label, True, f"可导入 {', '.join(found)} ({resolved})"))
        else:
            results.append((label, False, f"目录存在但找不到模块 {modules}：{resolved}"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--joycon", action="store_true",
                        help="连接右 Joy-Con 做 3 秒实时包率与姿态漂移体检")
    args = parser.parse_args(argv)

    print("=== 1. 配置与运动学检查 ===")
    settings = Settings.load(args.config)
    kin = RSKinematics(settings.urdf_path, settings.section("arm").get("end_effector_frame", "gripper_end"))
    limiter = JointLimiter(kin.limits, settings.gripper_limits,
                           float(settings.section("safety").get("joint_limit_scale", 0.7)))
    limiter.validate(settings.home)
    pose = kin.pose(settings.home[:6])
    ws = settings.workspace_kwargs()
    in_ws = workspace_contains(pose[:3], **ws)
    margins = np.rad2deg(np.minimum(settings.home[:6] - limiter.soft[0, :6],
                                    limiter.soft[1, :6] - settings.home[:6]))
    print(f"  [OK] URDF: {settings.urdf_path.name}")
    print(f"  [OK] HOME TCP={pose[:3].round(3).tolist()} m, rpy={np.rad2deg(pose[3:]).round(1).tolist()} deg")
    print(f"  [{'OK' if in_ws else 'FAIL'}] 工作空间内={in_ws} "
          f"(肩距={np.linalg.norm(pose[:3] - ws['origin']):.3f}m / 上限={ws['radius']:.2f}m)")
    print(f"  [OK] HOME 距软限位最小余量={float(np.min(margins)):.1f}°")
    print(f"  [OK] Joy-Con 键位映射: {settings.joycon_map()}")

    print("\n=== 2. CAN 与 SDK 检查 ===")
    can_ok, can_msg = _check_can("can0")
    print(f"  [{'OK' if can_ok else 'WARN'}] CAN: {can_msg}")
    for label, ok, msg in _check_sdk(settings):
        print(f"  [{'OK' if ok else 'WARN'}] {label}: {msg}")

    print("\n=== 3. Joy-Con hidraw 检查 ===")
    side = str(settings.section("joycon").get("device", "right"))
    try:
        found = enumerate_joycons(side)
        if found:
            print(f"  [OK] 发现 {len(found)} 个 {side} Joy-Con: {found[0].get('product_string')} "
                  f"serial={found[0].get('serial')}")
        else:
            print(f"  [WARN] 未发现 {side} Joy-Con（请唤醒手柄并确认蓝牙已连、/dev/hidraw* 有权限）")
    except Exception as exc:
        print(f"  [WARN] hidraw 枚举失败：{exc}")

    if args.joycon:
        print("\n=== 4. Joy-Con 3 秒实时包率体检 ===")
        driver = NativeJoyConDriver(device=side)
        try:
            start_snap = driver.snapshot()
            t0 = time.monotonic()
            r0 = start_snap.rotation.copy()
            time.sleep(3.0)
            end_snap = driver.snapshot()
            dt = max(time.monotonic() - t0, 1e-6)
            rate = (end_snap.packet_count - start_snap.packet_count) / dt
            drift_deg = float(np.rad2deg(np.linalg.norm(
                Rotation.from_matrix(end_snap.rotation @ r0.T).as_rotvec())))
            print(f"  [OK] 数据包速率: {rate:.1f} Hz（正常 ~60 Hz）")
            print(f"  [OK] 3 秒静止姿态漂移: {drift_deg:.2f}°")
            print(f"  [OK] 当前摇杆 XY: {end_snap.stick_xy.round(3).tolist()}")
            print(f"  [OK] 当前按下键: {[k for k, v in end_snap.buttons.items() if v]}")
        finally:
            driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
