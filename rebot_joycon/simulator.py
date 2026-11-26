from __future__ import annotations

import numpy as np

from .config import Settings


class MuJoCoView:
    """轻量 RS URDF 的可视化；显示经过安全层的实际发送目标。"""

    def __init__(self, settings: Settings, headless: bool = False):
        self.settings = settings
        self.headless = headless
        self.model = None
        self.data = None
        self.viewer = None
        self._arm_addresses = []
        self._finger_targets = []

    def _model_paths(self):
        configured = self.settings.section("simulation").get("urdf")
        if configured:
            path = self.settings.resolve(configured)
            if path.is_file():
                return [path]
            print(f"[sim] 配置的可视化模型不存在，回退自动选择：{path}")
        candidates = []
        sdk_path = self.settings.section("arm").get("sdk_path")
        if sdk_path:
            official = self.settings.resolve(sdk_path) / "urdf/RS/urdf/ReBot_Arm_RS.urdf"
            if official.is_file():
                candidates.append(official)
        candidates.append(self.settings.urdf_path)
        return candidates

    def __enter__(self):
        try:
            import mujoco
        except ImportError as exc:
            if self.headless:
                print("[sim] 无头模式跳过 MuJoCo（当前环境未安装 mujoco）", flush=True)
                return self
            raise RuntimeError(
                "当前 Python 环境未安装 mujoco；真机请加 --headless，"
                "或执行 pip install mujoco 后再启用仿真显示"
            ) from exc
        self.mujoco = mujoco
        errors = []
        for path in self._model_paths():
            try:
                self.model = mujoco.MjModel.from_xml_path(str(path))
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
            print(f"[sim] MuJoCo 显示模型：{path}")
            break
        else:
            raise RuntimeError("MuJoCo 模型加载失败：\n" + "\n".join(errors))
        self.data = mujoco.MjData(self.model)
        self._arm_addresses = [self.model.jnt_qposadr[self.model.joint(f"joint{i}").id]
                               for i in range(1, 7)]
        self._finger_targets = []
        for name, full_open in (("gripper_joint1", 0.05), ("gripper_joint2", 0.0715),
                                ("joint7", 0.05), ("joint_left", 0.05),
                                ("joint_right", 0.0715)):
            try:
                joint_id = self.model.joint(name).id
            except KeyError:
                continue
            self._finger_targets.append((self.model.jnt_qposadr[joint_id], full_open))
        if not self.headless:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            center = np.asarray(self.model.stat.center, dtype=float)
            self.viewer.cam.azimuth = 45
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = max(0.8, float(self.model.stat.extent) * 1.8)
            self.viewer.cam.lookat = ([float(center[0]), float(center[1]),
                                       max(float(center[2]), 0.25)]
                                      if np.isfinite(center).all() else [0.0, 0.0, 0.25])
        return self

    @property
    def running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def _apply_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[self._arm_addresses] = q[:6]
        scale = float(self.settings.section("safety").get("joint_limit_scale", 0.7))
        lower, upper = self.settings.gripper_limits * scale
        fraction = float(np.clip((upper - q[6]) / (upper - lower), 0.0, 1.0))
        for address, full_open in self._finger_targets:
            self.data.qpos[address] = full_open * fraction
        self.mujoco.mj_forward(self.model, self.data)

    def update(self, q: np.ndarray) -> None:
        if self.model is None:
            return
        q = np.asarray(q, dtype=float)
        if self.viewer is not None:
            if not self.viewer.is_running():
                return
            # 必须在 viewer.lock() 内修改 qpos 并执行 mj_forward，避免与渲染线程竞态。
            with self.viewer.lock():
                self._apply_qpos(q)
            self.viewer.sync()
        else:
            self._apply_qpos(q)

    def __exit__(self, *args) -> None:
        if self.viewer is not None:
            self.viewer.close()
