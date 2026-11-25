from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _origin(xyz: str | None, rpy: str | None) -> np.ndarray:
    transform = np.eye(4)
    if xyz:
        transform[:3, 3] = np.fromstring(xyz, sep=" ")
    if rpy:
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(rpy, sep=" ")).as_matrix()
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    return transform


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    improved: bool
    manipulability: float
    position_error: float
    orientation_error: float


def acceptable_ik_result(result: IKResult, *, max_position_error: float = 0.015,
                         max_orientation_error: float = 0.08) -> bool:
    """Reject partial IK solutions unless they are close to the requested pose."""
    if result.success:
        return True
    return (result.improved
            and math.isfinite(result.position_error)
            and math.isfinite(result.orientation_error)
            and result.position_error <= max_position_error
            and result.orientation_error <= max_orientation_error)


class RSKinematics:
    """从 RS 官方 URDF 读取的六轴 FK／自适应阻尼 IK。

    使用模型坐标，不使用 DM 校准角。雅可比和 IK 只取末端帧之前的六个
    旋转关节，夹爪两个平移关节不会混入机械臂配置。
    """

    def __init__(self, urdf_path: str | Path, end_effector_frame: str = "gripper_end"):
        self.urdf_path = Path(urdf_path).resolve()
        self.end_effector_frame = end_effector_frame
        root = ET.parse(self.urdf_path).getroot()
        joints: dict[str, JointSpec] = {}
        children: dict[str, list[JointSpec]] = {}
        for element in root.findall("joint"):
            limit = element.find("limit")
            origin = element.find("origin")
            axis = element.find("axis")
            joint = JointSpec(
                name=element.attrib["name"],
                parent=element.find("parent").attrib["link"],
                child=element.find("child").attrib["link"],
                origin=_origin(origin.attrib.get("xyz") if origin is not None else None,
                               origin.attrib.get("rpy") if origin is not None else None),
                axis=(np.fromstring(axis.attrib["xyz"], sep=" ") if axis is not None
                      else np.zeros(3)),
                lower=float(limit.attrib.get("lower", -np.inf)) if limit is not None else -np.inf,
                upper=float(limit.attrib.get("upper", np.inf)) if limit is not None else np.inf,
            )
            joints[joint.name] = joint
            children.setdefault(joint.parent, []).append(joint)
        link = end_effector_frame
        chain: list[JointSpec] = []
        while link != "base_link":
            matches = [j for j in joints.values() if j.child == link]
            if len(matches) != 1:
                raise ValueError(f"URDF 末端链不唯一或缺少 base_link：{link}")
            chain.append(matches[0])
            link = matches[0].parent
        chain.reverse()
        self.chain = chain
        self.joints = [j for j in chain if np.linalg.norm(j.axis) > 0 and j.name.startswith("joint")]
        if len(self.joints) != 6 or [j.name for j in self.joints] != [f"joint{i}" for i in range(1, 7)]:
            raise ValueError("RS 末端链必须依次包含 joint1…joint6")
        for joint in self.joints:
            norm = np.linalg.norm(joint.axis)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError(f"{joint.name} 轴无效")
            object.__setattr__(joint, "axis", joint.axis / norm)
        self.limits = np.array([[j.lower for j in self.joints],
                                [j.upper for j in self.joints]], dtype=float)
        if not np.isfinite(self.limits).all() or np.any(self.limits[0] >= self.limits[1]):
            raise ValueError("RS 六轴 URDF 限位无效")

    def _transforms(self, q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("机械臂配置必须是六个有限数值")
        transform = np.eye(4)
        frames: list[np.ndarray] = []
        index = 0
        for joint in self.chain:
            transform = transform @ joint.origin
            if any(joint is controlled for controlled in self.joints):
                frames.append(transform.copy())
                transform = transform @ _axis_rotation(joint.axis, float(q[index]))
                index += 1
        return transform, frames

    def forward(self, q: np.ndarray) -> np.ndarray:
        return self._transforms(q)[0]

    def pose(self, q: np.ndarray) -> np.ndarray:
        transform = self.forward(q)
        return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        end, frames = self._transforms(q)
        jacobian = np.zeros((6, 6))
        for index, (joint, frame) in enumerate(zip(self.joints, frames, strict=True)):
            axis = frame[:3, :3] @ joint.axis
            jacobian[:3, index] = np.cross(axis, end[:3, 3] - frame[:3, 3])
            jacobian[3:, index] = axis
        return jacobian

    @staticmethod
    def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = np.zeros(6)
        error[:3] = target[:3, 3] - current[:3, 3]
        rotation = target[:3, :3] @ current[:3, :3].T
        error[3:] = Rotation.from_matrix(rotation).as_rotvec()
        return error

    def ik(self, seed: np.ndarray, target: np.ndarray, *, max_iterations: int = 25,
           manipulability_min: float = 0.005,
           joint_limits: np.ndarray | None = None,
           position_weight: float = 4.0,
           orientation_weight: float = 1.0) -> IKResult:
        """在指定关节限位内求解自适应阻尼 DLS-IK（默认使用软限位以避免后截断漂移）。"""
        limits = self.limits if joint_limits is None else np.asarray(joint_limits, dtype=float)
        if limits.shape != (2, 6) or not np.isfinite(limits).all() or np.any(limits[0] >= limits[1]):
            raise ValueError("IK 关节限位必须是有效的 2×6 数组")
        # 始终与硬限位取交集，绝不越过 URDF 物理限位。
        low = np.maximum(self.limits[0], limits[0])
        high = np.minimum(self.limits[1], limits[1])
        q = np.clip(np.asarray(seed, dtype=float), low, high)
        target = np.asarray(target, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or target.shape != (4, 4):
            raise ValueError("IK 种子或目标无效")
        if not np.isfinite(target).all():
            raise ValueError("IK 目标必须有限")
        weights = np.array([position_weight, position_weight, position_weight,
                            orientation_weight, orientation_weight, orientation_weight],
                           dtype=float)
        initial = self.pose_error(self.forward(q), target)
        initial_score = float(np.linalg.norm(weights * initial))
        best_q = q.copy()
        best_score = initial_score
        best_manip = 0.0
        for _ in range(max_iterations):
            current = self.forward(q)
            error = self.pose_error(current, target)
            position_error = float(np.linalg.norm(error[:3]))
            orientation_error = float(np.linalg.norm(error[3:]))
            if position_error < 1e-4 and orientation_error < 1e-3:
                return IKResult(q, True, best_score < initial_score, best_manip,
                                position_error, orientation_error)
            jacobian = self.jacobian(q)
            product = jacobian @ jacobian.T
            manip = float(np.sqrt(max(0.0, np.linalg.det(product))))
            best_manip = manip
            deficit = max(0.0, manipulability_min - manip) / max(manipulability_min, 1e-9)
            damping = 0.02 + 0.28 * min(deficit, 1.0)
            weighted_j = weights[:, None] * jacobian
            weighted_error = weights * error
            jjt_damped = weighted_j @ weighted_j.T + (damping ** 2) * np.eye(6)
            dq = weighted_j.T @ np.linalg.solve(jjt_damped, weighted_error)
            # 等比例步长缩放（而非逐轴截断），保持严格雅可比下降方向，避免 Yaw/Pitch 耦合拒解。
            max_step = float(np.max(np.abs(dq)))
            if max_step > 0.25:
                dq = dq * (0.25 / max_step)
            improved = False
            for scale in (1.0, 0.5, 0.25, 0.1, 0.05):
                candidate = np.clip(q + scale * dq, low, high)
                candidate_error = self.pose_error(self.forward(candidate), target)
                score = float(np.linalg.norm(weights * candidate_error))
                if np.isfinite(score) and score < best_score - 1e-9:
                    q = candidate
                    best_q = candidate.copy()
                    best_score = score
                    improved = True
                    break
            if not improved:
                break
        final_error = self.pose_error(self.forward(best_q), target)
        return IKResult(best_q, False, best_score < initial_score, best_manip,
                        float(np.linalg.norm(final_error[:3])),
                        float(np.linalg.norm(final_error[3:])))


def make_pose(xyz_rpy: np.ndarray) -> np.ndarray:
    values = np.asarray(xyz_rpy, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("目标位姿必须是 xyz+rpy 六个有限数值")
    transform = np.eye(4)
    transform[:3, 3] = values[:3]
    transform[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    return transform
