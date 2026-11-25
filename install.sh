#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON=python
else
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN=python3.12
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "缺少 Python 3.10+；请安装 python3.12 或激活 conda 环境" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m venv .venv
  PYTHON="$ROOT/.venv/bin/python"
fi
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "需要 Python 3.10+，当前：$("$PYTHON" --version 2>&1)" >&2
  exit 1
fi
"$PYTHON" -m pip install -r requirements.txt
JOYCON_COMMIT="3e37ebfdf1db88fe4ecaa7018ae9c8c49efcde0e"
if [[ ! -d joycon-robotics ]]; then
  git clone https://github.com/box2ai-robotics/joycon-robotics.git joycon-robotics
  git -C joycon-robotics checkout --detach "$JOYCON_COMMIT"
elif [[ -d joycon-robotics/.git ]]; then
  ACTUAL="$(git -C joycon-robotics rev-parse HEAD)"
  [[ "$ACTUAL" == "$JOYCON_COMMIT" ]] || {
    echo "Joy-Con 上游版本不匹配：$ACTUAL；需要 $JOYCON_COMMIT" >&2
    exit 1
  }
fi
"$PYTHON" -m pip install -e ./joycon-robotics
"$PYTHON" -c "import numpy, scipy, yaml, mujoco, evdev, pynput; print('基础依赖检查通过')"
"$PYTHON" tools/smoke_test.py
