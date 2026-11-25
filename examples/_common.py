from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(mode: str, input_kind: str, real: bool) -> None:
    from rebot_joycon.cli import main
    argv = ["--mode", mode, "--input", input_kind]
    if real:
        argv.append("--real")
    raise SystemExit(main(argv + sys.argv[1:]))
