#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebot_joycon.backends import create_backend
from rebot_joycon.config import Settings
from rebot_joycon.kinematics import RSKinematics
from rebot_joycon.safety import JointLimiter, workspace_contains


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 HOME，或只读保存当前姿态为 HOME 候选")
    parser.add_argument("--config", default=None)
    parser.add_argument("--save-current", action="store_true",
                        help="读取当前反馈写 home_candidate.yaml；仅 POS_VEL 不使能读取，MIT 请求会被拒绝")
    parser.add_argument("--backend", choices=("pos_vel", "mit"), default="pos_vel")
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)
    kin = RSKinematics(settings.urdf_path)
    limiter = JointLimiter(kin.limits, settings.gripper_limits,
                           float(settings.section("safety").get("joint_limit_scale", 0.7)))
    gripper = settings.home[6]
    if args.save_current:
        backend = create_backend(settings, args.backend)
        try:
            state = backend.connect(enable=False)
            q = state.position[:6]
            gripper = state.position[6]
        finally:
            backend.abort_before_enable()
    else:
        q = settings.home[:6]
    pose = kin.pose(q)
    J = kin.jacobian(q)
    manip = float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
    ws = settings.workspace_kwargs()
    sh_dist = float(np.linalg.norm(pose[:3] - ws["origin"]))
    margins_deg = np.rad2deg(np.minimum(q - limiter.soft[0, :6], limiter.soft[1, :6] - q))
    in_ws = workspace_contains(pose[:3], **ws)
    print("HOME 六轴 deg:", np.rad2deg(q).round(2).tolist())
    print("软限位余量 deg:", margins_deg.round(1).tolist(),
          f"(最小 {float(np.min(margins_deg)):.1f}°)")
    print("TCP xyz(m):", pose[:3].round(4).tolist(),
          "rpy(deg):", np.rad2deg(pose[3:]).round(2).tolist())
    print("肩部距离 %.3f m（安全上限 %.2f m，在工作空间内=%s）；可操作度 %.5f" %
          (sh_dist, ws["radius"], in_ws, manip))
    if args.save_current:
        output = settings.root / "config" / "home_candidate.yaml"
        output.write_text(yaml.safe_dump({"home": {"q": q.tolist(),
                                                   "gripper": float(gripper)}},
                                         allow_unicode=True, sort_keys=False),
                          encoding="utf-8")
        print(f"已写候选：{output}；确认 FK 和现场姿态后手动合入 default.yaml")
    else:
        print("要运动到 HOME，请用示例加 --home-on-start，并全程按住使能。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
