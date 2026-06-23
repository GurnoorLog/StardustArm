import numpy as np
import mujoco
from config import IK_DAMPING, IK_MAX_ITER, IK_TOLERANCE, JOINT_LIMITS


def solve_ik(model, data, target_pos, body_id, joint_qposadrs, joint_dofadrs,
             q_init=None, damping=IK_DAMPING, max_iter=IK_MAX_ITER,
             tol=IK_TOLERANCE):
    body_world = data.xpos[body_id].copy()
    dist = float(np.linalg.norm(body_world - target_pos))
    if dist < tol:
        return data.qpos.copy(), True

    if q_init is not None:
        for adr, val in zip(joint_qposadrs, q_init.copy()):
            data.qpos[adr] = val
        mujoco.mj_forward(model, data)

    nv = model.nv
    jac = np.zeros((3, nv))
    err = target_pos - data.xpos[body_id]

    last_err_norm = float("inf")

    for iteration in range(max_iter):
        mujoco.mj_jac(model, data, jac, None, np.zeros(3), body_id)
        J = jac[:, :nv]

        JJT = J @ J.T
        JJT_reg = JJT + damping * damping * np.eye(3)
        try:
            v = np.linalg.solve(JJT_reg, err)
        except np.linalg.LinAlgError:
            return data.qpos.copy(), False
        dq = J.T @ v

        for i in range(model.nv):
            data.qvel[i] = dq[i]

        mujoco.mj_integratePos(model, data.qpos, data.qvel, 1.0)
        mujoco.mj_forward(model, data)

        err = target_pos - data.xpos[body_id]
        err_norm = float(np.linalg.norm(err))

        if err_norm < tol:
            return data.qpos.copy(), True

        if abs(err_norm - last_err_norm) < 1e-6 and err_norm > tol * 10:
            break

        last_err_norm = err_norm

        for adr in joint_qposadrs:
            in_adr = adr - joint_qposadrs[0]
            if in_adr < 0 or in_adr >= len(JOINT_LIMITS):
                continue
            low, high = JOINT_LIMITS[in_adr]
            data.qpos[adr] = np.clip(data.qpos[adr], low, high)

    return data.qpos.copy(), err_norm < tol * 5


def solve_ik_avoid(model, data, target_pos, body_id, joint_qposadrs,
                   joint_dofadrs, obstacle_positions, obstacle_radius,
                   q_init=None, damping=IK_DAMPING, max_iter=IK_MAX_ITER,
                   tol=IK_TOLERANCE, avoid_gain=0.3):
    body_world = data.xpos[body_id].copy()
    err = target_pos - body_world
    err_norm = float(np.linalg.norm(err))
    if err_norm < tol:
        return data.qpos.copy(), True, 0.0

    if q_init is not None:
        for adr, val in zip(joint_qposadrs, q_init.copy()):
            data.qpos[adr] = val
        mujoco.mj_forward(model, data)

    nv = model.nv
    jac = np.zeros((3, nv))
    last_err_norm = float("inf")

    for iteration in range(max_iter):
        body_world = data.xpos[body_id].copy()
        err = target_pos - body_world
        err_norm = float(np.linalg.norm(err))

        mujoco.mj_jac(model, data, jac, None, np.zeros(3), body_id)
        J = jac[:, :nv]

        v_des = err * min(1.0, err_norm / 0.3)
        min_obs_dist = float("inf")

        for obs_pos in obstacle_positions:
            for link_name in ["link2", "link3", "link4", "link5", "link6",
                              "link7", "gripper_base"]:
                for i in range(model.nbody):
                    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if n == link_name:
                        link_pos = data.xpos[i]
                        to_obs = link_pos - obs_pos
                        obs_dist = float(np.linalg.norm(to_obs))
                        min_obs_dist = min(min_obs_dist, obs_dist)
                        if obs_dist < obstacle_radius * 2.0:
                            repulse = to_obs / (obs_dist + 1e-8)
                            v_des += repulse * avoid_gain * (
                                1.0 - obs_dist / (obstacle_radius * 2.0)
                            )
                        break
                else:
                    continue
                break

        JJT = J @ J.T
        JJT_reg = JJT + damping * damping * np.eye(3)
        try:
            v_solved = np.linalg.solve(JJT_reg, v_des)
        except np.linalg.LinAlgError:
            return data.qpos.copy(), False, min_obs_dist
        dq = J.T @ v_solved

        for i in range(model.nv):
            data.qvel[i] = dq[i]

        mujoco.mj_integratePos(model, data.qpos, data.qvel, 1.0)
        mujoco.mj_forward(model, data)

        for adr in joint_qposadrs:
            in_adr = adr - joint_qposadrs[0]
            if in_adr < 0 or in_adr >= len(JOINT_LIMITS):
                continue
            low, high = JOINT_LIMITS[in_adr]
            data.qpos[adr] = np.clip(data.qpos[adr], low, high)

        if err_norm < tol:
            return data.qpos.copy(), True, min_obs_dist

        if abs(err_norm - last_err_norm) < 1e-6 and err_norm > tol * 10:
            break
        last_err_norm = err_norm

    body_pos = data.xpos[body_id].copy()
    final_dist = float(np.linalg.norm(target_pos - body_pos))
    return data.qpos.copy(), final_dist < tol * 5, min_obs_dist
