#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="$ROOT/.deps"
UPSTREAM="$DEPS/rebot-joycon-upstream"
LEROBOT_COMMIT="fd1b61bcd0dd2343f647d7236900adb9102ac9b5"
mkdir -p "$DEPS"
if [[ ! -d "$UPSTREAM/.git" ]]; then
  git clone https://github.com/box2ai-robotics/rebot-joycon.git "$UPSTREAM"
fi
git -C "$UPSTREAM" fetch origin "$LEROBOT_COMMIT"
git -C "$UPSTREAM" checkout --detach "$LEROBOT_COMMIT"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON="$(command -v python)"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  echo "先运行项目根目录 install.sh，并激活其 conda 或 .venv 环境" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "缺少 uv；请在当前环境执行 python -m pip install uv" >&2
  exit 1
fi
cd "$UPSTREAM/Learning_Model/lerobot"
uv pip install --python "$PYTHON" -e '.[hardware,training,test]'
if [[ -d "$ROOT/joycon-robotics" ]]; then
  uv pip install --python "$PYTHON" -e "$ROOT/joycon-robotics"
fi
"$PYTHON" -m pip check
