from collections import deque
from config import MEMORY_WINDOW


class Memory:
    def __init__(self, window=MEMORY_WINDOW):
        self.window = window
        self.history = deque(maxlen=window)

    def add(self, state, action):
        summary = (
            f"Step {state.get('step', '?')}: "
            f"EE={state.get('ee_pos', '?')} "
            f"dist={state.get('dist', 0):.3f} "
            f"action=joint{action['joint']} delta={action['delta']:+.3f} "
            f"reward={state.get('reward', 0):+.0f}"
        )
        self.history.append(summary)

    def get_context(self):
        if not self.history:
            return "  (no prior actions yet)"
        lines = list(self.history)
        return "\n".join(f"  {s}" for s in lines)

    def clear(self):
        self.history.clear()
