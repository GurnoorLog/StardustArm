import json
import numpy as np
from collections import Counter
from config import BEST_STRATEGIES_PATH, TOP_STRATEGIES_TO_KEEP


class StrategyExtractor:
    def __init__(self):
        pass

    def extract(self, episodes):
        if not episodes:
            return {}

        success_eps = [e for e in episodes if e.get("outcome") == "SUCCESS"]
        failed_eps = [e for e in episodes if e.get("outcome") != "SUCCESS"]

        all_telemetry = []
        for e in episodes:
            all_telemetry.extend(e.get("telemetry", []))

        if not all_telemetry:
            return {}

        rewards = [t["reward"] for t in all_telemetry]
        if max(rewards) == min(rewards):
            return self._empty_result()

        threshold = np.percentile(rewards, 75)
        top_steps = [t for t in all_telemetry if t["reward"] >= threshold]

        joint_movements = Counter()
        for t in top_steps:
            act = t.get("action", {})
            if "joint" in act:
                joint_movements[act["joint"]] += 1
        most_moved = joint_movements.most_common(1)
        best_joint = most_moved[0][0] if most_moved else None

        success_deltas = []
        failed_deltas = []
        for e in success_eps:
            for t in e.get("telemetry", []):
                act = t.get("action", {})
                if "delta" in act:
                    success_deltas.append(act["delta"])
        for e in failed_eps:
            for t in e.get("telemetry", []):
                act = t.get("action", {})
                if "delta" in act:
                    failed_deltas.append(act["delta"])

        avg_success_delta = np.mean(success_deltas) if success_deltas else 0.0
        avg_failed_delta = np.mean(failed_deltas) if failed_deltas else 0.0

        first_actions = []
        for e in success_eps:
            telemetry = e.get("telemetry", [])
            first_five = []
            for t in telemetry[:5]:
                act = t.get("action", {})
                if "joint" in act:
                    first_five.append(act["joint"])
            if len(first_five) >= 5:
                first_actions.append(tuple(first_five))
        common_opening = []
        if first_actions:
            counter = Counter(first_actions)
            common_opening = list(counter.most_common(1)[0][0])

        steps_to_success = [e.get("total_steps", 0) for e in success_eps if e.get("total_steps", 0) > 0]
        avg_steps = np.mean(steps_to_success) if steps_to_success else 0.0
        fastest = min(steps_to_success) if steps_to_success else 0
        fastest_reward = 0
        for e in success_eps:
            if e.get("total_steps", 0) == fastest:
                fastest_reward = e.get("final_reward", 0)
                break

        return {
            "best_joint": best_joint,
            "avg_success_delta": float(avg_success_delta),
            "avg_failed_delta": float(avg_failed_delta),
            "common_opening": common_opening,
            "avg_steps_to_success": float(avg_steps),
            "fastest_success_steps": fastest,
            "fastest_success_reward": fastest_reward,
            "num_success_missions": len(success_eps),
            "num_failed_missions": len(failed_eps),
        }

    def _empty_result(self):
        return {
            "best_joint": None,
            "avg_success_delta": 0.0,
            "avg_failed_delta": 0.0,
            "common_opening": [],
            "avg_steps_to_success": 0.0,
            "fastest_success_steps": 0,
            "fastest_success_reward": 0,
            "num_success_missions": 0,
            "num_failed_missions": 0,
        }

    def get_top_strategies(self, episodes, n=TOP_STRATEGIES_TO_KEEP):
        extracted = self.extract(episodes)
        sorted_eps = sorted(
            episodes,
            key=lambda e: e.get("final_reward", 0),
            reverse=True
        )
        top = sorted_eps[:n]
        top_data = []
        for ep in top:
            top_data.append({
                "episode_id": ep.get("episode_id"),
                "final_reward": ep.get("final_reward"),
                "total_steps": ep.get("total_steps"),
                "outcome": ep.get("outcome"),
            })
        payload = {
            "strategy_insights": extracted,
            "top_episodes": top_data,
        }
        with open(BEST_STRATEGIES_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    def summarise_for_prompt(self):
        try:
            with open(BEST_STRATEGIES_PATH, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return ""
        insights = data.get("strategy_insights", {})
        if not insights or insights.get("num_success_missions", 0) == 0:
            return ""

        lines = []
        lines.append(f"From {insights['num_success_missions']} successful missions:")
        if insights.get("best_joint") is not None:
            lines.append(f"- Best opening move: joint {insights['best_joint']}, delta +{abs(insights['avg_success_delta']):.2f}")
        if insights.get("fastest_success_steps", 0) > 0:
            lines.append(f"- Fastest success: {insights['fastest_success_steps']} steps (reward {insights['fastest_success_reward']})")
        if insights.get("common_opening"):
            seq = str(insights["common_opening"])
            lines.append(f"- Most effective joint sequence to start: {seq}")
        if insights.get("avg_steps_to_success", 0) > 0:
            lines.append(f"- Average steps to success: {insights['avg_steps_to_success']:.0f}")
        return "\n".join(lines)
