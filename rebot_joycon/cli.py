from __future__ import annotations

import argparse
import math
import signal
import sys
import time
import traceback

from .backends import create_backend
from .config import Settings
from .controller import TeleopController
from .errors import SafeStopRequested
from .inputs import JoyConInput, KeyboardInput
from .kinematics import RSKinematics
from .logging_utils import EventLog
from .simulator import MuJoCoView


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="reBot B601-RS 键盘／Joy-Con 安全遥操作")
    parser.add_argument("--mode", choices=("qpos", "gpos"), required=True,
                        help="qpos=关节空间，gpos=末端位姿/IK")
    parser.add_argument("--input", choices=("keyboard", "joycon"), required=True)
    parser.add_argument("--real", action="store_true", help="连接真实 B601-RS；默认仅仿真")
    parser.add_argument("--dry-run", action="store_true", help="使用真实输入但 Fake 机械臂")
    parser.add_argument("--backend", choices=("auto", "mit", "pos_vel"), default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-vel", type=float, default=None,
                        help="覆盖六轴最大速度(rad/s，允许 0.005–2.0；首次真机建议 0.05)")
    parser.add_argument("--home-on-start", action="store_true",
                        help="显式要求启动后平滑回 HOME；默认保持当前姿态")
    parser.add_argument("--headless", action="store_true", help="不打开 MuJoCo 窗口")
    parser.add_argument("--duration", type=float, default=None, help="运行时长秒数，默认不限")
    parser.add_argument("--yes", action="store_true", help="跳过真机交互确认（仅限已清场测试）")
    return parser


def _install_termination_signals() -> dict[int, object]:
    """将 SIGTERM/SIGHUP 转为 SafeStopRequested，避免非 Ctrl+C 终止绕过 safe_stop。"""
    previous: dict[int, object] = {}

    def _handle(signum, _frame):
        raise SafeStopRequested(f"收到系统终止信号({signum})，执行安全回零")

    for sig_name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle)
        except (OSError, ValueError):
            pass
    return previous


def _restore_signals(previous: dict[int, object]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0.0):
        raise ValueError("--duration 必须是正的有限秒数")
    if args.real and args.dry_run:
        print("--real 与 --dry-run 不能同时使用", file=sys.stderr)
        return 2
    if args.mode == "qpos" and args.input == "joycon":
        print("Joy-Con 只提供笛卡尔位姿，请使用 --mode gpos", file=sys.stderr)
        return 2
    settings = Settings.load(args.config)
    settings.with_max_arm_velocity(args.max_vel)
    if args.real and settings.section("arm").get("real_arm_confirmation", True) and not args.yes:
        print("\n真机确认：B601-RS 已固定，工作区无人无遮挡，手在电源开关旁。")
        if args.home_on_start:
            print("启动后按住使能（Joy-Con R / 键盘 Space）即平滑回 HOME；到位后松开再按住开始操作。")
        else:
            print("程序默认保持当前姿态；运动需按住使能，退出会平滑回零后再失能。")
        if input("输入大写 RS 继续：").strip() != "RS":
            print("已取消，未连接机械臂")
            return 0
    kin = RSKinematics(settings.urdf_path, settings.section("arm").get("end_effector_frame",
                                                                       "gripper_end"))
    log = EventLog(settings.resolve(settings.section("logging").get("directory", "logs")))
    backend = None
    input_source = None
    controller = None
    failure = None
    saved_signals = _install_termination_signals()
    try:
        home_pose = kin.pose(settings.home[:6])
        gripper_close, gripper_open = settings.gripper_close_open
        input_source = (JoyConInput(settings, home_pose[:3],
                                    gripper_open, gripper_close, home_pose[3:])
                        if args.input == "joycon" else KeyboardInput(settings))
        backend = create_backend(settings, "fake" if not args.real or args.dry_run else args.backend,
                                 initial=settings.home)
        controller = TeleopController(settings, backend, input_source, kin, args.mode,
                                      on_episode_event=lambda event: log.write("episode_button",
                                                                                value=event),
                                      event_log=log)
        log.write("startup", real=args.real, dry_run=args.dry_run, mode=args.mode,
                  input=args.input, backend=backend.name, urdf=str(settings.urdf_path),
                  max_velocity=settings.max_velocity.tolist(), home=settings.home.tolist())
        headless = args.headless or bool(settings.section("simulation").get("headless", False))
        # 真机模式下渲染限频 15 Hz，避免 OpenGL 阻塞 30 Hz 实机控制环。
        view_interval = 1.0 / 15.0 if args.real else 0.0
        last_view_sync = 0.0
        with MuJoCoView(settings, headless=headless) as view:
            view.update(settings.home)
            controller.start()
            if args.home_on_start:
                controller.home_pending = True
            began = time.monotonic()
            duration = (args.duration if args.duration is not None
                        else settings.section("simulation").get("duration_s"))
            while view.running and (duration is None or time.monotonic() - began < float(duration)):
                tick = time.monotonic()
                frame = controller.step()
                if not headless and (tick - last_view_sync >= view_interval):
                    view.update(frame.target)
                    last_view_sync = tick
                time.sleep(max(0.0, settings.control_period - (time.monotonic() - tick)))
    except (KeyboardInterrupt, SafeStopRequested) as exc:
        reason = str(exc) or "用户中断"
        print(f"\n停止：{reason}")
        log.write("stop_requested", reason=reason)
    except Exception as exc:  # finally 仍走统一安全停止，不直接 disable。
        failure = exc
        log.write("error", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        _restore_signals(saved_signals)
        try:
            if controller is not None:
                controller.close()
            else:
                if input_source is not None:
                    input_source.close()
                if backend is not None:
                    backend.abort_before_enable()
            log.write("shutdown", safe=True)
        except Exception as exc:
            failure = exc
            print(f"\n[严重] 安全停止失败：{exc}", file=sys.stderr)
            print("不要在弯曲姿态直接失能或断电；扶稳机械臂并排除故障。", file=sys.stderr)
            log.write("shutdown", safe=False, error=str(exc))
        finally:
            log.close()
    return 2 if failure is not None else 0
