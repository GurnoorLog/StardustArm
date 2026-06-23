import pybullet as p
from config import ARM_MOUNT_OFFSET


class BasePlatform:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.body_id = None
        self._build()

    def _build(self):
        mount_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.20, 0.20, 0.03],
            rgbaColor=[0.15, 0.15, 0.18, 1.0]
        )
        mount_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.20, 0.20, 0.03]
        )
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=mount_col,
            baseVisualShapeIndex=mount_vis,
            basePosition=[0.0, 0.0, -0.03],
        )

        floor_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[12.0, 12.0, 0.01]
        )
        floor_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[12.0, 12.0, 0.01],
            rgbaColor=[0.50, 0.52, 0.55, 1.0]
        )
        self._floor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_col,
            baseVisualShapeIndex=floor_vis,
            basePosition=[0.0, 0.0, -0.01],
        )

    def get_arm_mount_position(self):
        return [
            self.position[0] + ARM_MOUNT_OFFSET[0],
            self.position[1] + ARM_MOUNT_OFFSET[1],
            self.position[2] + ARM_MOUNT_OFFSET[2],
        ]

    def get_base_id(self):
        return self.body_id
