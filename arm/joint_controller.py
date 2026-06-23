import pybullet as p
import numpy as np
from config import JOINT_LIMIT_WARNING


JOINT_LIMITS = [(-1.57, 1.57) for _ in range(6)]
JOINT_LIMIT_MARGIN = 0.05 * 3.14


JOINT_FORCE = 8.0
JOINT_VELOCITY = 50.0


def apply_action(arm_id, joint_idx, delta):
    num_joints = p.getNumJoints(arm_id)
    if joint_idx < 0 or joint_idx >= num_joints:
        return
    low, high = JOINT_LIMITS[joint_idx]
    current_state = p.getJointState(arm_id, joint_idx)
    current_angle = current_state[0]
    new_angle = np.clip(current_angle + delta, low, high)
    p.resetJointState(arm_id, joint_idx, float(new_angle))
    p.setJointMotorControl2(
        arm_id, joint_idx,
        p.POSITION_CONTROL,
        targetPosition=float(new_angle),
        force=JOINT_FORCE,
        maxVelocity=JOINT_VELOCITY,
    )


def set_joint(arm_id, joint_idx, angle_rad):
    num_joints = p.getNumJoints(arm_id)
    if joint_idx < 0 or joint_idx >= num_joints:
        return
    low, high = JOINT_LIMITS[joint_idx]
    clamped = np.clip(angle_rad, low, high)
    p.resetJointState(arm_id, joint_idx, float(clamped))
    p.setJointMotorControl2(
        arm_id, joint_idx,
        p.POSITION_CONTROL,
        targetPosition=float(clamped),
        force=JOINT_FORCE,
        maxVelocity=JOINT_VELOCITY,
    )


def get_all_angles(arm_id):
    num_joints = p.getNumJoints(arm_id)
    angles = []
    for j in range(num_joints):
        info = p.getJointInfo(arm_id, j)
        if info[2] == p.JOINT_REVOLUTE:
            state = p.getJointState(arm_id, j)
            angles.append(state[0])
    return angles


def reset_arm(arm_id):
    num_joints = p.getNumJoints(arm_id)
    for j in range(num_joints):
        info = p.getJointInfo(arm_id, j)
        if info[2] == p.JOINT_REVOLUTE:
            p.resetJointState(arm_id, j, 0.0)
            p.setJointMotorControl2(
                arm_id, j,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=JOINT_FORCE,
                maxVelocity=JOINT_VELOCITY,
            )
