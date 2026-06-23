import numpy as np


def log_step(step, ee_pos, dist, action, reward, cumulative, qwen_used, debris_list=None):
    if step % 10 != 0:
        return
    qwen_str = "OK" if qwen_used else "FALLBACK"
    ee_str = f"({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
    action_str = f"joint={action['joint']} delta={action['delta']:+0.3f}"

    if debris_list:
        close = sum(
            1 for d in debris_list
            if np.linalg.norm(np.array(d.position) - np.array(ee_pos)) < 1.5
        )
        debris_str = f" | DEBRIS: {close}"
    else:
        debris_str = ""

    print(
        f"[STEP {step:03d}] EE: {ee_str} | "
        f"DIST: {dist:.3f} | "
        f"ACTION: {action_str} | "
        f"REWARD: {reward:+3.0f} | "
        f"CUMUL: {cumulative:.0f} | "
        f"QWEN: {qwen_str}{debris_str}",
        flush=True,
    )
