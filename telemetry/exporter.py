import json
import os
from learning.episode_store import EpisodeStore


class Exporter:
    def __init__(self):
        self.store = EpisodeStore()

    def save_mission(self, episode):
        self.store.save_episode(episode)
        total = self.store.get_total_episodes()
        success_rate = self.store.get_success_rate()
        best = self.store.get_best_episode()
        best_reward = best.get("final_reward", 0) if best else 0
        print(
            f"[MEMORY] Episode {episode['episode_id']} saved. "
            f"Total episodes: {total}. "
            f"Success rate: {success_rate:.0f}%. "
            f"Best reward ever: {best_reward:.0f}.",
            flush=True,
        )

    def run_post_mission(self, episode, strategy_extractor, failure_analyser):
        episodes = self.store.load_all_episodes()
        strategy_extractor.get_top_strategies(episodes)
        failure_data = failure_analyser.analyse(episodes)
        failure_analyser.save_failure_patterns(failure_data)
