import pybullet as p
import numpy as np
from config import (
    PANEL_MISSION_POS, PANEL_ENV_POS, PANEL_ARM_POS, PANEL_AI_POS,
)


class EnvPanels:
    def __init__(self):
        self._text_ids = []
        self._panel_ids = []
        self._create_panels()

    def _make_panel(self, pos, color):
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.45, 0.02, 0.35],
            rgbaColor=color,
        )
        pid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        self._panel_ids.append(pid)

    def _create_panels(self):
        self._make_panel(PANEL_MISSION_POS, [0.1, 0.6, 0.1, 0.85])
        self._make_panel(PANEL_ENV_POS, [0.1, 0.2, 0.7, 0.85])
        self._make_panel(PANEL_ARM_POS, [0.7, 0.1, 0.1, 0.85])
        self._make_panel(PANEL_AI_POS, [0.5, 0.1, 0.7, 0.85])

    def _text(self, text, pos, size, color):
        pid = p.addUserDebugText(
            text, pos, textColorRGB=color,
            textSize=size, lifeTime=0,
        )
        self._text_ids.append(pid)

    def _line_y(self, panel_pos, index, offset=0.15):
        return [
            panel_pos[0],
            panel_pos[1] + offset,
            panel_pos[2] + 0.28 - index * 0.14,
        ]

    def update(self, state):
        for tid in self._text_ids:
            try:
                p.removeUserDebugItem(tid)
            except Exception:
                pass
        self._text_ids = []

        ep = state.get("episode", "?")
        step = state.get("step", 0)
        reward = state.get("reward", 0.0)
        cumul = state.get("cumulative_reward", 0.0)
        status = state.get("status", "IDLE")
        dc = state.get("debris_count", 0)
        grav = state.get("gravity_z", 0.0)
        speed = state.get("sim_speed", 1.0)
        tpos = state.get("target_pos", [0.0, 0.0, 0.0])
        dist = state.get("dist_to_target", 0.0)
        ee = state.get("ee_pos", [0.0, 0.0, 0.0])
        ja = state.get("joint_angles", [0.0] * 6)
        jw = state.get("joint_warnings", [])
        qs = state.get("qwen_status", "WAITING")
        br = state.get("best_reward", 0.0)
        be = state.get("best_ep", "?")
        sr = state.get("success_rate", 0.0)

        tx, ty, tz = tpos[0], tpos[1], tpos[2]
        ex, ey, ez = ee[0], ee[1], ee[2]
        j1, j2, j3, j4, j5, j6 = (
            ja[0] if len(ja) > 0 else 0,
            ja[1] if len(ja) > 1 else 0,
            ja[2] if len(ja) > 2 else 0,
            ja[3] if len(ja) > 3 else 0,
            ja[4] if len(ja) > 4 else 0,
            ja[5] if len(ja) > 5 else 0,
        )
        ws = ", ".join(f"J{j}" for j in jw[:4]) if jw else "NONE"

        # Panel 1 — MISSION CONTROL
        p1 = PANEL_MISSION_POS
        self._text("MISSION CONTROL", self._line_y(p1, -0.5), 1.6, [0.3, 1.0, 0.3])
        self._text(f"EP: {ep}  STEP: {step}", self._line_y(p1, 0), 1.15, [1, 1, 1])
        self._text(f"REWARD: {reward:+.0f}", self._line_y(p1, 1), 1.15, [1, 1, 1])
        self._text(f"CUMUL: {cumul:.0f}", self._line_y(p1, 2), 1.15, [1, 1, 1])
        self._text(f"STATUS: {status}", self._line_y(p1, 3), 1.15, [1, 1, 1])

        # Panel 2 — ENVIRONMENT
        p2 = PANEL_ENV_POS
        self._text("ENVIRONMENT", self._line_y(p2, -0.5), 1.6, [0.4, 0.6, 1.0])
        self._text(f"DEBRIS: {dc}  GRAV: {grav:.1f}", self._line_y(p2, 0), 1.15, [1, 1, 1])
        self._text(f"SPEED: {speed:.1f}x", self._line_y(p2, 1), 1.15, [1, 1, 1])
        self._text(f"TARGET: ({tx:.2f}, {ty:.2f}, {tz:.2f})", self._line_y(p2, 2), 1.15, [1, 1, 1])
        self._text(f"DIST: {dist:.3f} m", self._line_y(p2, 3), 1.15, [1, 1, 1])

        # Panel 3 — ARM STATUS
        p3 = PANEL_ARM_POS
        self._text("ARM STATUS", self._line_y(p3, -0.5), 1.6, [1.0, 0.4, 0.4])
        self._text(f"EE: ({ex:.2f}, {ey:.2f}, {ez:.2f})", self._line_y(p3, 0), 1.15, [1, 1, 1])
        self._text(f"J1:{j1:.2f}  J2:{j2:.2f}  J3:{j3:.2f}", self._line_y(p3, 1), 1.15, [1, 1, 1])
        self._text(f"J4:{j4:.2f}  J5:{j5:.2f}  J6:{j6:.2f}", self._line_y(p3, 2), 1.15, [1, 1, 1])
        self._text(f"WARNINGS: {ws}", self._line_y(p3, 3), 1.15, [1, 1, 1])

        # Panel 4 — AI BRAIN
        p4 = PANEL_AI_POS
        self._text("AI BRAIN", self._line_y(p4, -0.5), 1.6, [0.8, 0.4, 1.0])
        self._text("MODEL: qwen2.5:3b", self._line_y(p4, 0), 1.15, [1, 1, 1])
        self._text(f"QWEN: {qs}", self._line_y(p4, 1), 1.15, [1, 1, 1])
        self._text(f"BEST: {br:.0f} (EP {be})", self._line_y(p4, 2), 1.15, [1, 1, 1])
        self._text(f"SUCCESS RATE: {sr:.0f}%", self._line_y(p4, 3), 1.15, [1, 1, 1])
