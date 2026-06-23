import numpy as np
from config import TRAJECTORY_STEPS


def minimum_jerk_trajectory(start, end, steps=TRAJECTORY_STEPS):
    traj = []
    for i in range(steps + 1):
        t = i / steps
        tau = t
        s = 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5
        point = start + (end - start) * s
        traj.append(point.copy() if isinstance(point, np.ndarray) else list(point))
    return traj


def cubic_trajectory(start, end, steps=TRAJECTORY_STEPS):
    traj = []
    for i in range(steps + 1):
        t = i / steps
        s = 3.0 * t ** 2 - 2.0 * t ** 3
        point = start + (end - start) * s
        traj.append(point.copy() if isinstance(point, np.ndarray) else list(point))
    return traj


def compute_trajectory_time(distance, max_speed=0.5, max_accel=1.0):
    t_acc = max_speed / max_accel
    d_acc = 0.5 * max_accel * t_acc ** 2
    if 2 * d_acc >= distance:
        t_half = np.sqrt(distance / max_accel)
        return 2.0 * t_half
    d_cruise = distance - 2 * d_acc
    t_cruise = d_cruise / max_speed
    return 2.0 * t_acc + t_cruise
