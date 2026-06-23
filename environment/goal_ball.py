import pybullet as p
import numpy as np
from config import TARGET_POS, GOAL_BALL_RADIUS


class GoalBall:
    def __init__(self):
        self._radius = GOAL_BALL_RADIUS
        x, y, z = TARGET_POS
        self._create_ball([x, y, z], self._radius)

    def _create_ball(self, pos, radius):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius,
            rgbaColor=[0.78, 0.80, 0.85, 1.0],
        )
        self.ball_id = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        p.changeVisualShape(self.ball_id, -1, specularColor=[0.9, 0.9, 0.9])

    def set_radius(self, new_radius):
        if abs(new_radius - self._radius) < 0.001:
            return
        self._radius = new_radius
        pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        p.removeBody(self.ball_id)
        self._create_ball(pos, new_radius)

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        return list(pos)

    def set_position(self, xyz):
        p.resetBasePositionAndOrientation(self.ball_id, list(xyz), [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0])

    def enable_mouse_drag(self):
        print("[BALL] Mouse drag enabled — grab the ball to set the target")

    def update(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        return list(ball_pos)
