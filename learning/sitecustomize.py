"""PYTHONPATH 包含本目录时，自动注册项目自定义 LeRobot 设备。"""
try:
    import rebot_rs_lerobot  # noqa: F401
except ImportError:
    # 训练等不需要自定义硬件注册的场景不被 sitecustomize 阻断。
    pass
