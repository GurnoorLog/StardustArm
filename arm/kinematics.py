import pybullet as p
import numpy as np
from arm.robot_arm import LINK_LENGTHS


JOINT_AXES = [
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
]


def get_end_effector_pos(arm_id):
    num_joints = p.getNumJoints(arm_id)
    if num_joints == 0:
        pos, _ = p.getBasePositionAndOrientation(arm_id)
        return list(pos)
    last_revolute = num_joints - 1
    for j in range(num_joints - 1, -1, -1):
        info = p.getJointInfo(arm_id, j)
        jtype = info[2]
        if jtype == p.JOINT_REVOLUTE:
            last_revolute = j
            break
    link_state = p.getLinkState(arm_id, last_revolute)
    return list(link_state[4])


def check_joint_limits(angles, limits=None):
    if limits is None:
        limits = [(-1.57, 1.57) for _ in range(len(angles))]
    warnings = []
    for i, (angle, (low, high)) in enumerate(zip(angles, limits)):
        margin = 0.05 * (high - low)
        if angle <= low + margin or angle >= high - margin:
            warnings.append(i)
    return warnings


def detect_singularity(angles):
    total_extension = sum(abs(a) for a in angles)
    return total_extension > 2.5


def _rot_mat(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)],
    ])


def _mat_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]


def _fk_update(joint_angles):
    joint_positions = []
    link_centers = []
    link_orientations = []
    pos = np.array([0.0, 0.0, 0.0])
    R = np.eye(3)
    for i in range(6):
        axis_local = JOINT_AXES[i]
        angle = joint_angles[i]
        axis_world = R @ axis_local
        R_j = _rot_mat(axis_world, angle)
        R = R_j @ R
        joint_pos = pos.copy()
        joint_positions.append(joint_pos)
        half_len = LINK_LENGTHS[i] / 2.0
        z_local = R[:, 2]
        link_center = pos + z_local * half_len
        link_centers.append(link_center)
        link_orientations.append(_mat_to_quat(R))
        pos = pos + z_local * LINK_LENGTHS[i]
    return joint_positions, link_centers, link_orientations
