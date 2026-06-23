import json
import random
import requests
import numpy as np
from config import OLLAMA_URL, MODEL_NAME


class QwenAgent:
    def __init__(self, knowledge_injection="", env=None):
        self.knowledge_injection = knowledge_injection
        self.qwen_used = False
        self._stalled_steps = 0
        self._target = np.array([0.5, 0.0, 0.8], dtype=float)
        self._env = env

    def set_knowledge(self, knowledge):
        self.knowledge_injection = knowledge

    def update_target(self, new_target_xyz):
        self._target = np.array(new_target_xyz, dtype=float)

    def _build_prompt(self, ee_pos, joint_angles, distance, debris_positions, memory_context):
        lines = []
        if self.knowledge_injection:
            lines.append(self.knowledge_injection)
            lines.append("")

        lines.append("You are controlling a 6-DOF MEPhI_ARM robotic arm (MuJoCo physics).")
        lines.append("Goal: move the gripper end-effector (EE) to the target ball and grab it.")
        lines.append("")
        lines.append("ARM BODIES (kinematic chain):")
        lines.append("  base_link → link1 (j1: Z-axis rotation, ±90°)")
        lines.append("  → link2 (j2: -X-axis shoulder pitch, ±90°)")
        lines.append("  → link3 (j3: -X-axis elbow pitch, ±90°)")
        lines.append("  → link4 (j4: Y-axis wrist yaw, ±90°)")
        lines.append("  → link5 (j5: -X-axis wrist pitch, ±90°)")
        lines.append("  → link6 (j6: Y-axis wrist roll, ±180°)")
        lines.append("  → link7 → gripper_base (EE) → finger (prismatic, 0-3.7cm)")
        lines.append("")
        lines.append("CAPABILITIES:")
        lines.append("  - EE reachable workspace: ~0.4m radius hemisphere centered at (0, 0, 0.25)")
        lines.append("  - Finger closes to 0.0mm, opens to 37mm")
        lines.append("  - PD position control (kp=8.0) on all joints")
        lines.append("  - Arm can reach the floor (z=0.10) with elbow bent down")
        lines.append("  - Base rotation (j1) lets arm reach any direction")
        lines.append("")
        lines.append("STRATEGY:")
        lines.append("  - To reach downward (low Z): increase j2 and j3 (pitch forward)")
        lines.append("  - To reach upward (high Z): decrease j2 and j3 (pitch backward/up)")
        lines.append("  - To reach far: extend arm by keeping j3 near 0")
        lines.append("  - Use j4 (yaw) and j5 (pitch) to orient the EE toward the ball")
        lines.append("  - gripper fingers will automatically close to grasp the ball when the EE is within 0.13m of the target")
        lines.append("")
        lines.append("Current state:")
        lines.append(f"- End-effector position: {[round(v, 3) for v in ee_pos]}")
        lines.append(f"- Target position: {[round(v, 3) for v in self._target]}")
        lines.append(f"- Distance to target: {distance:.3f}")
        lines.append(f"- Joint angles (radians): {[round(v, 3) for v in joint_angles]}")
        lines.append(f"- Obstacles: {[[round(v, 3) for v in d] for d in debris_positions]}")
        lines.append("")
        lines.append("Recent action history:")
        lines.append(memory_context)
        lines.append("")
        lines.append('Respond ONLY with valid JSON. Example: {"joint": 3, "delta": 0.05}')
        lines.append("joint must be integer 0-5, delta must be float between -0.1 and 0.1.")
        return "\n".join(lines)

    def _call_qwen(self, prompt):
        print(f"[QWEN-LLM] Consulting Qwen model...")
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 128},
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception:
            return ""

    def _parse_qwen_response(self, text):
        text = text.strip()
        try:
            start = text.index("{")
            end = text.rindex("}")
            obj = json.loads(text[start:end + 1])
            raw_joint = obj.get("joint", 0)
            if isinstance(raw_joint, list):
                raw_joint = raw_joint[0] if raw_joint else 0
            joint = int(raw_joint)
            delta = float(obj.get("delta", 0.0))
            joint = max(0, min(5, joint))
            delta = max(-0.1, min(0.1, delta))
            return {"joint": joint, "delta": delta}, True
        except (ValueError, json.JSONDecodeError, KeyError):
            return None, False

    def _fk_eval(self, joint_angles, action, joint_limits):
        j, d = action["joint"], action["delta"]
        low, high = joint_limits[j]
        new_angle = joint_angles[j] + d
        if new_angle < low or new_angle > high:
            return float("inf")
        if self._env is not None:
            ee = self._env.compute_ee_for_action(joint_angles, action)
            return float(np.linalg.norm(ee - self._target))
        return float("inf")

    def _all_links_above(self, threshold=0.05):
        if self._env is not None:
            return self._env.check_links_above(threshold)
        return True

    def is_action_safe(self, joint_angles, action, step_scale=1.0):
        j, d = action["joint"], action["delta"]
        test_angles = list(joint_angles)
        test_angles[j] += d * step_scale
        if self._env is not None:
            return self._env.check_links_above_for_angles(test_angles, threshold=0.05)
        return True

    def _greedy_heuristic(self, joint_angles, joint_limits):
        best_action = {"joint": 0, "delta": 0.05}
        best_dist = float("inf")

        for j_idx in range(6):
            for delta in [-0.1, -0.05, 0.05, 0.1]:
                new_joint = joint_angles[j_idx] + delta
                low, high = joint_limits[j_idx]
                if new_joint < low or new_joint > high:
                    continue
                action = {"joint": j_idx, "delta": delta}
                if not self.is_action_safe(joint_angles, action):
                    continue
                if self._env is not None:
                    ee = self._env.compute_ee_for_action(joint_angles, action)
                    dist = float(np.linalg.norm(ee - self._target))
                    if dist < best_dist:
                        best_dist = dist
                        best_action = action

        return best_action, best_dist

    def _random_search(self, joint_angles, joint_limits, n_samples=50):
        best_action = None
        best_dist = float("inf")

        for _ in range(n_samples):
            deltas = np.random.uniform(-0.15, 0.15, size=6)
            deltas *= 0.5 + 0.5 * np.random.random()
            valid = True
            for j in range(6):
                low, high = joint_limits[j]
                new_val = joint_angles[j] + deltas[j]
                if new_val < low or new_val > high:
                    valid = False
                    break
            if not valid:
                continue
            test_angles = [joint_angles[j] + deltas[j] for j in range(6)]
            if self._env is not None:
                ee = self._env.compute_ee_for_angles(test_angles)
                dist = float(np.linalg.norm(ee - self._target))
                if dist < best_dist:
                    best_dist = dist
                    primary_idx = int(np.argmax(np.abs(deltas)))
                    best_action = {"joint": primary_idx, "delta": deltas[primary_idx]}

        if best_action is None:
            return {"joint": 0, "delta": 0.05}, float("inf")
        return best_action, best_dist

    def _get_valid_actions(self, joint_angles, joint_limits):
        actions = []
        for j_idx in range(6):
            for delta in [-0.1, -0.05, 0.05, 0.1]:
                new_joint = joint_angles[j_idx] + delta
                low, high = joint_limits[j_idx]
                if low <= new_joint <= high:
                    actions.append({"joint": j_idx, "delta": delta})
        return actions

    def act(self, ee_pos, joint_angles, distance, debris_positions, memory_context, target_pos=None):
        self.qwen_used = False
        if target_pos is not None:
            self._target = np.array(target_pos, dtype=float)
        joint_limits = [(-np.pi / 2, np.pi / 2) for _ in range(6)]

        greedy_action, greedy_dist = self._greedy_heuristic(joint_angles, joint_limits)
        greedy_improvement = distance - greedy_dist

        if greedy_improvement > 0.001:
            self._stalled_steps = 0
            return greedy_action

        self._stalled_steps += 1

        if self._stalled_steps >= 5:
            prompt = self._build_prompt(
                ee_pos, joint_angles, distance,
                debris_positions, memory_context
            )
            response_text = self._call_qwen(prompt)
            parsed, ok = self._parse_qwen_response(response_text)
            if ok:
                qwen_dist = self._fk_eval(joint_angles, parsed, joint_limits)
                if qwen_dist < distance:
                    self._stalled_steps = 0
                    self.qwen_used = True
                    print(f"[QWEN-LLM] Suggests joint{parsed['joint']} delta={parsed['delta']:+.3f} "
                          f"(current dist={distance:.3f}→{qwen_dist:.3f})")
                    return parsed

        valid = self._get_valid_actions(joint_angles, joint_limits)
        if valid:
            small = [a for a in valid if abs(a["delta"]) <= 0.05]
            return random.choice(small if small else valid)

        return greedy_action
