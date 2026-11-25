#!/usr/bin/env python3
"""右 Joy-Con 原生 hidraw 诊断与按键映射向导（方案 A）。"""
from __future__ import annotations

import argparse
import json
import select
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebot_joycon.joycon_driver import NativeJoyConDriver

DEFAULT_OUTPUT = ROOT / "config" / "joycon_map.json"
ITEMS = (
    ("deadman", "按住使能（推荐 r）", True),
    ("z_up", "纯 Z 轴上升（推荐 x）", True),
    ("z_down", "纯 Z 轴下降（推荐 b）", True),
    ("gripper", "夹爪张开/闭合切换（推荐 zr）", True),
    ("home", "平滑回 HOME（推荐 plus / + 键）", False),
    ("exit", "长按安全回零并退出（推荐 home）", False),
    ("estop", "紧急停止（按住使能同时按，推荐 r_stick / 摇杆按下）", True),
    ("episode_end", "LeRobot 提前结束 episode（推荐 a）", False),
    ("episode_retry", "LeRobot 重录 episode（推荐 y）", False),
)


def capture_button(driver: NativeJoyConDriver, prompt: str, mandatory: bool) -> str | None:
    print(prompt + ("（必需，30 秒超时）" if mandatory else "（Enter 跳过，30 秒超时）"),
          flush=True)
    # 先等待所有按键释放，防止前一步按键连带。
    while any(driver.snapshot().buttons.values()):
        time.sleep(0.02)
    deadline = time.monotonic() + 30.0
    prev = driver.snapshot().buttons
    while time.monotonic() < deadline:
        readable, _, _ = select.select([sys.stdin], [], [], 0.03)
        if sys.stdin in readable:
            sys.stdin.readline()
            if mandatory:
                print("必需键不能跳过，请按手柄按键。", flush=True)
                continue
            return None
        curr = driver.snapshot().buttons
        for name, state in curr.items():
            if state and not prev.get(name, False):
                print(f"  捕获按键：{name}", flush=True)
                return name
        prev = curr
    if mandatory:
        raise TimeoutError(f"必需键超时：{prompt}")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="右 Joy-Con 原生 hidraw 诊断与交互按键映射")
    parser.add_argument("--device", choices=("right", "left"), default="right")
    parser.add_argument("--print", action="store_true",
                        help="实时打印按键、摇杆 XY 与 IMU 欧拉角（Ctrl+C 退出）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    driver = NativeJoyConDriver(device=args.device)
    try:
        if args.print:
            print("实时监控 Joy-Con 输入（按 Ctrl+C 退出）：", flush=True)
            while True:
                snap = driver.snapshot()
                pressed = [k for k, v in snap.buttons.items() if v]
                rpy = np.rad2deg(Rotation.from_matrix(snap.rotation).as_euler("xyz"))
                print(
                    f"\rpkts={snap.packet_count:5d} "
                    f"stick_xy=[{snap.stick_xy[0]:+.2f},{snap.stick_xy[1]:+.2f}] "
                    f"rpy_deg=[{rpy[0]:+6.1f},{rpy[1]:+6.1f},{rpy[2]:+6.1f}] "
                    f"btns={pressed}          ",
                    end="",
                    flush=True,
                )
                time.sleep(0.05)

        old = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
        mapping = dict(old)
        mapping.setdefault("deadman", "r")
        mapping.setdefault("z_up", "x")
        mapping.setdefault("z_down", "b")
        mapping.setdefault("gripper", "zr")
        mapping.setdefault("home", "plus")
        mapping.setdefault("exit", "home")
        mapping.setdefault("estop", "r_stick")
        mapping.setdefault("episode_end", "a")
        mapping.setdefault("episode_retry", "y")

        for role, prompt, mandatory in ITEMS:
            while True:
                value = capture_button(driver, f"{role}: {prompt}", mandatory)
                if value is None:
                    break
                duplicates = [other for other, code in mapping.items()
                              if code == value and other != role]
                if duplicates:
                    print(f"  按键 {value} 已分配给 {duplicates}，请换键重试。", flush=True)
                    continue
                mapping[role] = value
                break
        mapping["startup_guard_s"] = 1.0
        print("\n新映射：", json.dumps(mapping, indent=2, ensure_ascii=False))
        if input("写入配置？输入 y 确认：").strip().lower() != "y":
            print("未写入")
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            shutil.copy2(args.output, args.output.with_suffix(args.output.suffix + ".bak"))
        args.output.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"已写入 {args.output}")
        return 0
    except KeyboardInterrupt:
        print("\n已退出")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
