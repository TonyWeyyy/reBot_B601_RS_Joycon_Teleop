from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .backends import create_backend
from .config import Settings


def finite_or_none(values) -> list[float | None]:
    """JSON 不承载 NaN/Infinity；缺失的 MIT 力矩或温度反馈写成 null。"""
    result = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        number = float(value)
        result.append(number if np.isfinite(number) else None)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="采集 B601-RS 七路反馈；不发送运动目标、不写电机硬件零点")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", choices=("pos_vel", "mit"), default="pos_vel")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)
    output = (args.output or settings.root / "config" / "calibration_snapshot.json").resolve()
    backend = create_backend(settings, args.backend)
    try:
        state = backend.connect(enable=False)
        data = {
            "model": "reBot-B601-RS",
            "created_unix": time.time(),
            "backend": backend.name,
            "urdf": str(settings.urdf_path),
            "read_only": True,
            "hardware_zero_written": False,
            "position_rad": state.position.tolist(),
            "position_deg": np.rad2deg(state.position).tolist(),
            "torque": finite_or_none(state.torque),
            "temperature_c": finite_or_none(state.temperatures),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")
        print(f"已保存只读快照：{output}")
        print("未发送运动命令、未写硬件零点；仅 POS_VEL 支持不使能快照，MIT 只读请求会被拒绝。")
        if all(value is None for value in data["temperature_c"]):
            print("[警告] 七路温度均为 null：本机固件没有提供温度反馈，"
                  "软件 75/80°C 保护不会生效，请勿依赖它。")
        if all(value is None for value in data["torque"]):
            print("[警告] 七路力矩均为 null：夹爪 1 N·m 软件保护不会生效。")
        return 0
    finally:
        backend.abort_before_enable()
