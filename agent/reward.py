import numpy as np
from config import (
    REWARD_CLOSER, REWARD_COLLISION, REWARD_JOINT_LIMIT, REWARD_SUCCESS,
)


def calculate_reward(prev_dist, curr_dist, collisions, joint_warnings, success):
    reward = 0.0
    reasons = []

    if success:
        reward += REWARD_SUCCESS
        reasons.append(f"SUCCESS(+{REWARD_SUCCESS})")
        return reward, "; ".join(reasons)

    dist_delta = prev_dist - curr_dist
    reward += dist_delta * REWARD_CLOSER
    if dist_delta > 0:
        reasons.append(f"closer(+{dist_delta * REWARD_CLOSER:.1f})")
    else:
        reasons.append(f"farther({dist_delta * REWARD_CLOSER:.1f})")

    if collisions:
        penalty = REWARD_COLLISION * len(collisions)
        reward += penalty
        reasons.append(f"collision({penalty})")

    if joint_warnings:
        penalty = REWARD_JOINT_LIMIT * len(joint_warnings)
        reward += penalty
        reasons.append(f"joint_limit({penalty})")

    return reward, "; ".join(reasons)
