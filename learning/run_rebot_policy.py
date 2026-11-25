#!/usr/bin/env python3
"""B601-RS LeRobot 策略启动器；修改顶部路径后运行。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LEROBOT = ROOT / ".deps" / "rebot-joycon-upstream" / "Learning_Model" / "lerobot"
if os.environ.get("CONDA_PREFIX"):
    PYTHON = Path(sys.executable)
else:
    PYTHON = (ROOT / ".venv" / "bin" / "python"
              if (ROOT / ".venv" / "bin" / "python").is_file() else Path(sys.executable))
POLICY_PATH = LEROBOT / "outputs/train/rebot_smoke/checkpoints/last/pretrained_model"
CAMERA_INDEX = 0
DEVICE = "cuda"
TASK = "抓取物体并放到目标区域"
DURATION = 10


def main() -> int:
    required = [PYTHON, LEROBOT / "pyproject.toml"] + [
        POLICY_PATH / name for name in ("config.json", "model.safetensors",
                                        "policy_preprocessor.json", "policy_postprocessor.json")]
    missing = [path for path in required if not path.is_file()]
    if missing:
        for path in missing:
            print(f"文件缺失：{path}", file=sys.stderr)
        return 1
    cameras = {"front": {"type": "opencv", "index_or_path": CAMERA_INDEX,
                         "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}
    command = [str(PYTHON), "-m", "lerobot.scripts.lerobot_rollout",
               "--strategy.type=base", "--inference.type=sync",
               f"--policy.path={POLICY_PATH.resolve()}", f"--device={DEVICE}",
               "--robot.type=rebot_b601_rs", "--robot.port=can0",
               "--robot.control_backend=mit", "--robot.id=rebot_rs",
               f"--robot.project_config={(ROOT / 'config/default.yaml').resolve()}",
               f"--robot.cameras={json.dumps(cameras)}", f"--task={TASK}", "--fps=30",
               f"--duration={DURATION}", "--interactive=true",
               "--return_to_initial_position=false", "--play_sounds=false"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "learning"), str(ROOT),
                                         env.get("PYTHONPATH", "")])
    print("启动保持当前姿态；/start 执行策略，/reset 平滑回 HOME，/stop 安全回零退出。")
    return subprocess.run(command, cwd=LEROBOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
