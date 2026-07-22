import sys
import os
import datetime
import time
import random
import threading
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

builtin_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    builtin_print(*args, **kwargs)

from config import (
    MAX_STEPS, SUCCESS_THRESHOLD, TARGET_POS,
    JOINT_LIMITS, GRAB_DIST, RELEASE_DIST,
    FINGER_OPEN, FINGER_CLOSED, COLLISION_MARGIN,
    TRAJECTORY_STEPS, IK_DAMPING, IK_MAX_ITER, IK_TOLERANCE,
    FLOATING_BASE, RL_ENABLED, VISION_WIDTH, VISION_HEIGHT,
    BALL_PRIORITIES, LIFT_ENABLED, LIFT_HEIGHT, LIFT_STEPS, SWING_STEPS,
)
from environment.mujoco_env import MuJoCoEnv
from agent.qwen_agent import QwenAgent
from agent.memory import Memory
from agent.reward import calculate_reward
from agent.ik_controller import solve_ik, solve_ik_avoid
from agent.trajectory import minimum_jerk_trajectory
from agent.vision import VisionSystem
from agent.rl_agent import PPOAgent, build_obs, compute_reward as rl_reward, RollingBuffer
from learning.episode_store import EpisodeStore
from learning.strategy_extractor import StrategyExtractor
from learning.failure_analyser import FailureAnalyser
from learning.knowledge_builder import KnowledgeBuilder
from telemetry.logger import log_step
from telemetry.exporter import Exporter


JOINT_LIMITS_LIST = JOINT_LIMITS

STATE_IDLE = "IDLE"
STATE_REACHING = "REACHING"
STATE_GRABBED = "GRABBED"
STATE_LIFTING = "LIFTING"
STATE_SWINGING = "SWINGING"
STATE_DONE = "DONE"


def apply_action(env, joint_idx, delta):
    angles = env.get_joint_angles()
    low, high = JOINT_LIMITS_LIST[joint_idx]
    new_angle = np.clip(angles[joint_idx] + delta, low, high)
    env.set_joint_target(joint_idx, new_angle)


def is_action_safe(angles, action, scale):
    if action["joint"] < 0 or action["joint"] >= 6:
        return False
    low, high = JOINT_LIMITS_LIST[action["joint"]]
    new_val = angles[action["joint"]] + action["delta"] * scale
    if new_val < low or new_val > high:
        return False
    return True


def ik_reach(env, target_pos, speed_scale=0.008, obstacle_pos=None, avoid=False,
             grabbed_ball=None):
    ee = np.array(env.get_ee_pos(), dtype=float)
    tgt = np.array(target_pos, dtype=float)
    direction = tgt - ee
    dist = float(np.linalg.norm(direction))
    if dist < 0.005:
        return
    step = ee + direction * min(1.0, speed_scale / max(dist, 0.001))
    wp = np.array(step, dtype=float)
    if avoid and obstacle_pos:
        joint_config, ik_ok, _ = solve_ik_avoid(
            env.model, env.data, wp,
            env.get_ee_body_id(),
            env.get_joint_qposadrs(),
            env.get_joint_dofadrs(),
            obstacle_pos, 0.08,
            q_init=None, damping=IK_DAMPING,
            max_iter=IK_MAX_ITER, tol=IK_TOLERANCE,
            avoid_gain=0.3,
        )
    else:
        joint_config, ik_ok = solve_ik(
            env.model, env.data, wp,
            env.get_ee_body_id(),
            env.get_joint_qposadrs(),
            env.get_joint_dofadrs(),
            q_init=None, damping=IK_DAMPING,
            max_iter=IK_MAX_ITER, tol=IK_TOLERANCE,
        )
    for j_idx, adr in enumerate(env.get_joint_qposadrs()):
        if j_idx < env.model.nu - 1:
            env.data.qpos[adr] = joint_config[adr]
            env.set_joint_target(j_idx, joint_config[adr])
    if grabbed_ball:
        env.sync_grabbed_ball(grabbed_ball)


def compute_joint_target(env, target_pos):
    joint_config, ik_ok = solve_ik(
        env.model, env.data, np.array(target_pos, dtype=float),
        env.get_ee_body_id(),
        env.get_joint_qposadrs(),
        env.get_joint_dofadrs(),
        q_init=None, damping=IK_DAMPING,
        max_iter=IK_MAX_ITER, tol=IK_TOLERANCE,
    )
    if not ik_ok:
        q_home = [0.0] * len(env.get_joint_qposadrs())
        joint_config, ik_ok = solve_ik(
            env.model, env.data, np.array(target_pos, dtype=float),
            env.get_ee_body_id(),
            env.get_joint_qposadrs(),
            env.get_joint_dofadrs(),
            q_init=q_home, damping=IK_DAMPING,
            max_iter=IK_MAX_ITER, tol=IK_TOLERANCE,
        )
        if ik_ok:
            print(f"[JT] IK succeeded from home position")
    return joint_config, ik_ok


def start_episode(env, memory, rl_active):
    memory.clear()
    env.set_finger_target(FINGER_OPEN)
    env.reset_arm()
    # Restore ball collisions first (they may have been disabled during previous grab)
    env.reset_ball_geom_collisions()
    # Deactivate any active grab welds from the previous episode
    for ball_name in env.get_ball_names():
        env.release_ball(ball_name)
    env.set_ball_pos(TARGET_POS, "ball_primary")
    env.set_ball_pos([-0.1, 0.2, 0.10], "ball_secondary")
    if rl_active:
        rl_agent = PPOAgent(obs_dim=13, act_dim=6)
        rl_buffer = RollingBuffer(2048, 13, 6)

    target_names = list(env.get_ball_names())
    random.shuffle(target_names)
    return target_names, 0


def main():
    print("=" * 42)
    print("   STARDANCE — Advanced MuJoCo Edition")
    print("   NASA On-Orbit Servicing Simulator")
    print("=" * 42)

    store = EpisodeStore()
    total_episodes = store.get_total_episodes()
    success_rate = store.get_success_rate()
    best_ep = store.get_best_episode()
    best_reward = best_ep.get("final_reward", 0) if best_ep else 0
    episode_id = total_episodes + 1

    knowledge_builder = KnowledgeBuilder()
    episodes = store.load_all_episodes()
    strategy_extractor = StrategyExtractor()
    failure_analyser = FailureAnalyser()

    if episodes:
        knowledge_builder.build(episodes)
    knowledge_injection = knowledge_builder.get_prompt_injection(episodes, store)

    print(f"\n[STARDANCE] Starting Episode {episode_id}")
    print(f"[MEMORY] Loaded {total_episodes} past episodes | "
          f"Success rate: {success_rate:.0f}% | Best reward: {best_reward}")

    env = MuJoCoEnv()
    env.load(floating_base=FLOATING_BASE)
    env.init_viewer()

    agent = QwenAgent(knowledge_injection, env=env)
    memory = Memory()
    exporter = Exporter()
    vision = VisionSystem(env)
    rl_agent = PPOAgent(obs_dim=13, act_dim=6) if RL_ENABLED else None
    rl_buffer = RollingBuffer(2048, 13, 6) if RL_ENABLED else None

    enter_pressed = threading.Event()
    def wait_for_terminal_enter():
        try:
            input(">>> Press ENTER in this terminal to start the episode <<<")
            enter_pressed.set()
        except (EOFError, OSError):
            pass
    threading.Thread(target=wait_for_terminal_enter, daemon=True).start()

    episode_num = 0
    step_count = 0
    et = None
    cumulative_reward = 0.0
    prev_dist = float("inf")
    success = False
    telemetry = []

    state = STATE_IDLE
    paused = True
    ik_mode = True
    use_vision = False
    floating = FLOATING_BASE
    rl_active = False
    grabbed = False
    collision_count = 0
    finger_open = True

    target_order = []
    target_idx = 0
    current_target_name = "ball_primary"
    target_pos = np.array(TARGET_POS, dtype=float)

    lift_traj = None
    lift_idx = 0
    lift_active = False
    swing_step = 0
    swing_j1_center = 0.0

    jt_target = None
    jt_traj = None
    jt_step = 0
    use_jt = False
    jt_failed = False

    ball_names = list(env.get_ball_names())

    print("[STATUS] Press ENTER (in GLFW window OR terminal) to start.\n")

    print("[CONTROLS]")
    print("  LEFT DRAG   — Orbit camera")
    print("  RIGHT DRAG  — Zoom  |  MIDDLE DRAG  — Pan")
    print("  SCROLL      — Zoom  |  ESC          — Exit")
    print("  ENTER       — Start episode  |  SPACE  — Pause")
    print("  R           — Reset balls")
    print("  I           — Toggle IK mode")
    print("  V           — Toggle vision detection")
    print("  F           — Toggle floating base\n")

    KEY_ENTER = env.get_glfw_key("ENTER")
    KEY_SPACE = env.get_glfw_key("SPACE")
    KEY_R = env.get_glfw_key("R")
    KEY_ESCAPE = env.get_glfw_key("ESCAPE")
    KEY_I = env.get_glfw_key("I")
    KEY_V = env.get_glfw_key("V")
    KEY_F = env.get_glfw_key("F")

    try:
        while env.is_running():
            # --- Terminal Enter detection (thread-based, no GLFW focus needed) ---
            if paused and enter_pressed.is_set():
                enter_pressed.clear()
                paused = False
                episode_num += 1
                step_count = 0
                cumulative_reward = 0.0
                prev_dist = float("inf")
                success = False
                telemetry = []
                collision_count = 0
                grabbed = False
                jt_target = None
                jt_failed = False
                use_jt = False
                state = STATE_REACHING
                target_order, target_idx = start_episode(env, memory, rl_active)
                current_target_name = target_order[target_idx]
                target_pos = env.get_ball_pos(current_target_name)
                print(f"[MAIN] === Episode {episode_num} ===")
                print(f"[MAIN] Targets: {target_order}")

            # --- GLFW key handlers ---
            if env.is_key_triggered(KEY_ENTER):
                if paused:
                    paused = False
                    episode_num += 1
                    step_count = 0
                    cumulative_reward = 0.0
                    prev_dist = float("inf")
                    success = False
                    telemetry = []
                    collision_count = 0
                    grabbed = False
                    jt_target = None
                    jt_failed = False
                    use_jt = False
                    state = STATE_REACHING
                    target_order, target_idx = start_episode(env, memory, rl_active)
                    current_target_name = target_order[target_idx]
                    target_pos = env.get_ball_pos(current_target_name)
                    print(f"[MAIN] === Episode {episode_num} ===")
                    print(f"[MAIN] Targets: {target_order}")

            if env.is_key_triggered(KEY_SPACE):
                paused = not paused
                print("[MAIN]", "PAUSED" if paused else "RESUMED")

            if env.is_key_triggered(KEY_R):
                env.set_ball_pos(TARGET_POS, "ball_primary")
                env.set_ball_pos([-0.1, 0.2, 0.10], "ball_secondary")
                print("[MAIN] Balls reset")

            if env.is_key_triggered(KEY_I):
                ik_mode = not ik_mode
                print(f"[MAIN] IK mode: {'ON' if ik_mode else 'OFF'}")

            if env.is_key_triggered(KEY_V):
                use_vision = not use_vision
                if use_vision:
                    print("[VISION] ON")
                    detected = vision.detect_balls()
                    if detected:
                        print(f"[VISION] {[d['name'] for d in detected]}")
                else:
                    print("[VISION] OFF")

            if env.is_key_triggered(KEY_F):
                floating = not floating
                print(f"[MAIN] Floating base: {'ON' if floating else 'OFF'}")

            if env.is_key_pressed(KEY_ESCAPE):
                break

            # --- Get current target position ---
            if use_vision:
                detected = vision.detect_balls()
                if detected:
                    current_target_name = detected[0]["name"]
                    target_pos = env.get_ball_pos(current_target_name)

            if state not in (STATE_IDLE, STATE_DONE):
                target_pos = env.get_ball_pos(current_target_name)

            if floating:
                env.stabilize_base()

            # === STATE MACHINE ===

            if paused or state == STATE_IDLE:
                env.step()
                status = "IDLE"

            elif state == STATE_REACHING:
                step_count += 1
                ee = env.get_ee_pos()
                curr_dist = float(np.linalg.norm(np.array(ee) - target_pos))

                # On first REACHING step for a new target, plan joint-space trajectory
                if not use_jt and jt_target is None and not jt_failed:
                    current_j = np.array(env.get_joint_angles())
                    jt_config_raw, jt_ok = compute_joint_target(env, target_pos)
                    if jt_ok:
                        jt_target = np.array([jt_config_raw[adr] for adr in env.get_joint_qposadrs()])
                        jt_traj = np.linspace(current_j, jt_target, TRAJECTORY_STEPS)
                        jt_step = 0
                        use_jt = True
                        ee_now = env.get_ee_pos()
                        traj_dist = np.linalg.norm(np.array(ee_now) - target_pos)
                        print(f"[JT] Joint traj: {len(jt_traj)} steps, dist={traj_dist:.3f}")
                        # Rewind to original position before trajectory starts
                        for j_idx, adr in enumerate(env.get_joint_qposadrs()):
                            if j_idx < env.model.nu - 1:
                                env.data.qpos[adr] = float(current_j[j_idx])
                        mujoco.mj_forward(env.model, env.data)
                    else:
                        use_jt = False
                        jt_failed = True
                        print(f"[JT] IK failed, fallback to incremental")

                angles_before = np.array(env.get_joint_angles())
                qwen_used = False

                # Follow joint-space trajectory
                if use_jt and jt_traj is not None and jt_step < len(jt_traj):
                    for j_idx, adr in enumerate(env.get_joint_qposadrs()):
                        if j_idx < env.model.nu - 1:
                            val = float(jt_traj[jt_step][j_idx])
                            env.data.qpos[adr] = val
                            env.set_joint_target(j_idx, val)
                    mujoco.mj_forward(env.model, env.data)
                    env.step()
                    jt_step += 1
                else:
                    if ik_mode:
                        ik_reach(env, target_pos, speed_scale=0.012 if curr_dist > 0.2 else 0.006,
                                 obstacle_pos=env.get_obstacle_positions(), avoid=bool(env.get_obstacle_positions()))
                        env.step()
                    else:
                        obs_pos = env.get_obstacle_positions()
                        history = memory.get_context()
                        action = agent.act(ee, angles_before.tolist(), curr_dist, obs_pos, history, target_pos=target_pos)
                        apply_action(env, action["joint"], action["delta"])
                        env.step()
                        qwen_used = agent.qwen_used

                # Compute action for telemetry (common to all branches)
                angles_after = np.array(env.get_joint_angles())
                diffs = angles_after - angles_before
                max_idx = int(np.argmax(np.abs(diffs))) if len(diffs) > 0 else 0
                max_diff = float(diffs[max_idx]) if len(diffs) > 0 else 0.0
                action = {"joint": max_idx, "delta": max_diff}

                # Update current state after step
                ee = env.get_ee_pos()
                curr_dist = float(np.linalg.norm(np.array(ee) - target_pos))
                angles_after = env.get_joint_angles()

                if step_count % 80 == 0:
                    print(f"[REACH] Step {step_count}, dist={curr_dist:.3f} to {current_target_name}")

                # Check for collisions / joint limits
                coll = env.check_collision(COLLISION_MARGIN)
                collisions_list = [True] if coll else []
                if coll:
                    collision_count += 1
                    if collision_count >= 3:
                        print("[MAIN] Collision — backing off")
                        collision_count = 0
                else:
                    collision_count = 0

                joint_warnings = []
                for j_idx, angle in enumerate(angles_after):
                    low, high = JOINT_LIMITS_LIST[j_idx]
                    if angle < low + 0.05 or angle > high - 0.05:
                        joint_warnings.append(j_idx)

                # Compute reward
                step_success = curr_dist < GRAB_DIST
                if prev_dist == float("inf"):
                    prev_dist = curr_dist
                step_reward, reason = calculate_reward(prev_dist, curr_dist, collisions_list, joint_warnings, step_success)
                cumulative_reward += step_reward

                # Save to memory/history for QwenAgent context
                state_dict = {
                    "step": step_count,
                    "ee_pos": [round(v, 3) for v in ee],
                    "dist": curr_dist,
                    "reward": step_reward
                }
                memory.add(state_dict, action)

                # Save telemetry
                telemetry.append({
                    "step": step_count,
                    "ee_pos": ee.tolist(),
                    "dist": curr_dist,
                    "action": action,
                    "reward": step_reward,
                    "cumulative": cumulative_reward,
                    "qwen_used": qwen_used,
                    "collision_count": 1 if coll else 0,
                    "joint_angles": angles_after,
                })

                # Log step
                log_step(
                    step=step_count,
                    ee_pos=ee,
                    dist=curr_dist,
                    action=action,
                    reward=step_reward,
                    cumulative=cumulative_reward,
                    qwen_used=qwen_used,
                    debris_list=None
                )

                prev_dist = curr_dist

                if curr_dist < GRAB_DIST:
                    grabbed = True
                    env.set_finger_target(FINGER_CLOSED)
                    env.grab_ball(current_target_name)
                    state = STATE_GRABBED
                    ee_now = env.get_ee_pos()
                    lift_target = [ee_now[0] * 0.3, ee_now[1] * 0.3, 0.48]
                    lift_traj = minimum_jerk_trajectory(
                        np.array(ee_now), np.array(lift_target), LIFT_STEPS
                    )
                    lift_idx = 0
                    lift_active = True
                    print(f"[GRAB] Caught {current_target_name}! Lifting...")

                if step_count > MAX_STEPS:
                    print(f"[MAIN] Max steps ({MAX_STEPS})")
                    state = STATE_DONE

                status = f"REACHING {current_target_name} ({curr_dist:.3f})"

            elif state == STATE_GRABBED:
                env.step()
                status = "GRABBED"

                if lift_active and lift_traj is not None and lift_idx < len(lift_traj):
                    wp = np.array(lift_traj[lift_idx], dtype=float)
                    lift_idx += 1
                    obs_pos = env.get_obstacle_positions()
                    ik_reach(env, wp, obstacle_pos=obs_pos, avoid=bool(obs_pos),
                             grabbed_ball=current_target_name)

            elif state == STATE_LIFTING:
                env.set_finger_target(FINGER_CLOSED)
                if lift_idx < len(lift_traj):
                    wp = np.array(lift_traj[lift_idx], dtype=float)
                    lift_idx += 1
                    obs_pos = env.get_obstacle_positions()
                    ik_reach(env, wp, obstacle_pos=obs_pos, avoid=bool(obs_pos),
                             grabbed_ball=current_target_name)
                    env.step()
                    status = f"LIFTING ({lift_idx}/{len(lift_traj)})"
                else:
                    lift_active = False
                    swing_step = 0
                    angles = env.get_joint_angles()
                    swing_j1_center = angles[0]
                    state = STATE_SWINGING
                    print(f"[SWING] {current_target_name} secured! Swinging...")

            elif state == STATE_SWINGING:
                swing_step += 1
                angle = swing_j1_center + 0.25 * np.sin(swing_step * 0.03)
                env.set_joint_target(0, float(angle))
                env.set_finger_target(FINGER_CLOSED)
                env.step()
                if grabbed:
                    env.sync_grabbed_ball(current_target_name)
                status = f"SWINGING {current_target_name} ({swing_step}/{SWING_STEPS})"

                if swing_step >= SWING_STEPS:
                    # Done showing this ball — release it and go to next
                    env.release_ball(current_target_name)
                    env.set_finger_target(FINGER_OPEN)
                    grabbed = False
                    target_idx += 1
                    if target_idx < len(target_order):
                        current_target_name = target_order[target_idx]
                        target_pos = env.get_ball_pos(current_target_name)
                        lift_active = False
                        lift_traj = None
                        swing_step = 0
                        jt_target = None
                        jt_failed = False
                        use_jt = False
                        prev_dist = float("inf")
                        state = STATE_REACHING
                        print(f"[NEXT] Moving to {current_target_name}...")
                    else:
                        state = STATE_DONE
                        success = True
                        print(f"[MAIN] ALL {len(target_order)} BALLS CAPTURED! Mission complete!")

            elif state == STATE_DONE:
                env.step()
                status = "MISSION COMPLETE" if success else "FAILED"

            env.render()
            env.freeze_balls(exclude=current_target_name if grabbed else None)
            env.poll_keys()

    except KeyboardInterrupt:
        print("\n[STARDANCE] Interrupted.")
    except Exception as loop_err:
        print(f"\n[LOOP] Error: {loop_err}")
        import traceback
        traceback.print_exc()
    finally:
        outcome = "SUCCESS" if success else "FAILURE"
        ep = {
            "episode_id": episode_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_steps": step_count,
            "final_reward": cumulative_reward,
            "outcome": outcome,
            "mode": "multi_ik" if ik_mode else "greedy",
            "targets_captured": target_idx,
            "total_targets": len(target_order),
            "telemetry": telemetry,
        }
        exporter.save_mission(ep)
        exporter.run_post_mission(ep, strategy_extractor, failure_analyser)
        env.close()


if __name__ == "__main__":
    main()
