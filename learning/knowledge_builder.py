from learning.strategy_extractor import StrategyExtractor
from learning.failure_analyser import FailureAnalyser
from config import MIN_EPISODES_BEFORE_LEARNING, KNOWLEDGE_MAX_TOKENS


class KnowledgeBuilder:
    def __init__(self):
        self.strategy_extractor = StrategyExtractor()
        self.failure_analyser = FailureAnalyser()

    def build(self, episodes):
        strategy_data = self.strategy_extractor.extract(episodes)
        failure_data = self.failure_analyser.analyse(episodes)
        self.strategy_extractor.get_top_strategies(episodes)
        self.failure_analyser.save_failure_patterns(failure_data)
        return strategy_data, failure_data

    def get_prompt_injection(self, episodes, store=None):
        total = len(episodes)
        if total < MIN_EPISODES_BEFORE_LEARNING:
            return ""

        if store is None:
            from learning.episode_store import EpisodeStore
            store = EpisodeStore()

        best_ep = store.get_best_episode()
        best_reward = best_ep.get("final_reward", 0) if best_ep else 0
        best_steps = best_ep.get("total_steps", 0) if best_ep else 0
        best_id = best_ep.get("episode_id", "?") if best_ep else "?"

        success_rate = store.get_success_rate()
        successes = sum(1 for e in episodes if e.get("outcome") == "SUCCESS")

        strategy_summary = self.strategy_extractor.summarise_for_prompt()
        failure_summary = self.failure_analyser.summarise_for_prompt()

        lines = []
        lines.append(f"=== LEARNED KNOWLEDGE FROM {total} PAST MISSIONS ===")
        lines.append("")
        lines.append("WHAT WORKS:")
        lines.append(strategy_summary if strategy_summary else "  (insufficient data)")
        lines.append("")
        lines.append("WHAT TO AVOID:")
        lines.append(failure_summary if failure_summary else "  (insufficient data)")
        lines.append("")
        lines.append(f"PERSONAL BEST: {best_reward} reward in {best_steps} steps (Episode {best_id})")
        lines.append(f"TOTAL MISSIONS FLOWN: {total}")
        lines.append(f"SUCCESS RATE: {success_rate:.0f}%")
        lines.append("")
        lines.append("Use this knowledge to make better decisions than your previous runs.")
        lines.append("===")

        result = "\n".join(lines)

        if len(result) > KNOWLEDGE_MAX_TOKENS * 4:
            lines = lines[:6] + lines[-5:]
            result = "\n".join(lines)

        return result
