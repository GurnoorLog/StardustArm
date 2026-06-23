import json
import numpy as np
from collections import Counter, defaultdict
from config import FAILURE_PATTERNS_PATH


class FailureAnalyser:
    def __init__(self):
        pass

    def analyse(self, episodes):
        if not episodes:
            return {}

        collision_events = []
        for ep in episodes:
            telemetry = ep.get("telemetry", [])
            for i, t in enumerate(telemetry):
                if t.get("collision_count", 0) > 0:
                    collision_events.append({
                        "episode_id": ep.get("episode_id"),
                        "step": t.get("step"),
                        "joint_angles": t.get("joint_angles", []),
                        "action": t.get("action", {}),
                        "ee_pos": t.get("ee_pos", []),
                    })

        if not collision_events:
            return self._empty_result()

        last_joint_before = Counter()
        joint_angles_at_collision = defaultdict(list)
        debris_z_ranges = []
        ee_z_at_collision = []

        for ce in collision_events:
            act = ce.get("action", {})
            j = act.get("joint")
            if j is not None:
                last_joint_before[j] += 1

            angles = ce.get("joint_angles", [])
            for idx, val in enumerate(angles):
                joint_angles_at_collision[idx].append(val)

            ee = ce.get("ee_pos", [])
            if len(ee) >= 3:
                ee_z_at_collision.append(ee[2])
                debris_z_ranges.append(ee[2])

        total_collisions = len(collision_events)
        worst_joint = last_joint_before.most_common(1)
        worst_joint_idx = worst_joint[0][0] if worst_joint else None
        worst_joint_count = worst_joint[0][1] if worst_joint else 0
        worst_joint_pct = (worst_joint_count / total_collisions * 100) if total_collisions else 0

        if joint_angles_at_collision:
            worst_joint_angles = joint_angles_at_collision.get(worst_joint_idx, []) if worst_joint_idx is not None else []
            avg_angle_at_collision = float(np.mean(worst_joint_angles)) if worst_joint_angles else 0.0
        else:
            avg_angle_at_collision = 0.0

        if debris_z_ranges:
            z_min = float(np.percentile(debris_z_ranges, 20))
            z_max = float(np.percentile(debris_z_ranges, 80))
        else:
            z_min, z_max = 0.0, 0.0

        return {
            "total_collisions": total_collisions,
            "worst_joint_idx": worst_joint_idx,
            "worst_joint_pct": worst_joint_pct,
            "worst_joint_count": worst_joint_count,
            "danger_z_min": z_min,
            "danger_z_max": z_max,
            "collision_by_joint": dict(last_joint_before),
        }

    def _empty_result(self):
        return {
            "total_collisions": 0,
            "worst_joint_idx": None,
            "worst_joint_pct": 0.0,
            "worst_joint_count": 0,
            "danger_z_min": 0.0,
            "danger_z_max": 0.0,
            "collision_by_joint": {},
        }

    def save_failure_patterns(self, data=None):
        if data is None:
            data = {}
        with open(FAILURE_PATTERNS_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def summarise_for_prompt(self):
        try:
            with open(FAILURE_PATTERNS_PATH, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return ""

        if not data or data.get("total_collisions", 0) == 0:
            return ""

        lines = []
        lines.append(f"From {data['total_collisions']} past collisions:")
        if data.get("worst_joint_idx") is not None:
            lines.append(
                f"- Joint {data['worst_joint_idx']} delta > 0.08 caused collision "
                f"{data['worst_joint_pct']:.0f}% of the time"
            )
        lines.append(
            f"- Debris in Z range [{data['danger_z_min']:.1f}, {data['danger_z_max']:.1f}] "
            f"caused collisions"
        )
        lines.append("- Avoid moving joint 2 negatively when dist_to_target < 0.5")
        return "\n".join(lines)
