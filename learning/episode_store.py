import os
import json
import datetime
from config import (
    MEMORY_DIR, EPISODE_STORE_PATH, BEST_STRATEGIES_PATH, FAILURE_PATTERNS_PATH,
)


class EpisodeStore:
    def __init__(self):
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        for path in [EPISODE_STORE_PATH, BEST_STRATEGIES_PATH, FAILURE_PATTERNS_PATH]:
            if not os.path.exists(path):
                with open(path, "w") as f:
                    json.dump([], f)

    def save_episode(self, episode):
        episodes = self.load_all_episodes()
        episodes.append(episode)
        with open(EPISODE_STORE_PATH, "w") as f:
            json.dump(episodes, f, indent=2)

    def load_all_episodes(self):
        try:
            with open(EPISODE_STORE_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def get_total_episodes(self):
        return len(self.load_all_episodes())

    def get_best_episode(self):
        episodes = self.load_all_episodes()
        if not episodes:
            return None
        return max(episodes, key=lambda e: e.get("final_reward", 0))

    def get_recent_episodes(self, n=10):
        episodes = self.load_all_episodes()
        return episodes[-n:] if len(episodes) >= n else episodes[:]

    def get_success_rate(self):
        episodes = self.load_all_episodes()
        if not episodes:
            return 0.0
        successes = sum(1 for e in episodes if e.get("outcome") == "SUCCESS")
        return successes / len(episodes) * 100.0
