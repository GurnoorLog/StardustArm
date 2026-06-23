import pybullet as p
import random
import math
import numpy as np
from config import DEBRIS_VELOCITY_DEFAULT, DEBRIS_SPIN_DEFAULT


class Debris:
    def __init__(self, position, velocity, spin, radius, body_id, debris_type):
        self.position = list(position)
        self.velocity = list(velocity)
        self.spin = list(spin)
        self.radius = radius
        self.body_id = body_id
        self.debris_type = debris_type


class DebrisManager:
    def __init__(self):
        self.debris_list = []

    def spawn_debris(self, n, velocity_scale=None, spin_scale=None):
        if velocity_scale is None:
            velocity_scale = DEBRIS_VELOCITY_DEFAULT
        if spin_scale is None:
            spin_scale = DEBRIS_SPIN_DEFAULT
        new_debris = []
        spawn_radius = 1.0
        center = [1.8, 0.0, 0.5]
        types = ["rock", "panel", "cylinder", "fragment"]
        for _ in range(n):
            debris_type = random.choice(types)
            angle = random.uniform(0, 2 * math.pi)
            r = spawn_radius * math.sqrt(random.uniform(0.3, 1.0))
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle) * 0.6
            z = center[2] + random.uniform(-0.4, 0.4)
            pos = [x, y, z]

            speed = velocity_scale * random.uniform(0.5, 1.5)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(-0.5, 0.5)
            vel = [
                speed * math.cos(theta) * math.cos(phi),
                speed * math.sin(theta) * math.cos(phi),
                speed * math.sin(phi),
            ]
            ang_vel = [
                spin_scale * random.uniform(-2.0, 2.0),
                spin_scale * random.uniform(-2.0, 2.0),
                spin_scale * random.uniform(-2.0, 2.0),
            ]
            body_id = self._create_debris_body(pos, debris_type)
            p.resetBaseVelocity(body_id, linearVelocity=vel, angularVelocity=ang_vel)
            radius = self._get_type_radius(debris_type)
            d = Debris(pos, vel, ang_vel, radius, body_id, debris_type)
            new_debris.append(d)
        self.debris_list.extend(new_debris)
        return new_debris

    def _get_type_radius(self, debris_type):
        sizes = {
            "rock": random.uniform(0.04, 0.09),
            "panel": random.uniform(0.03, 0.06),
            "cylinder": random.uniform(0.03, 0.07),
            "fragment": random.uniform(0.02, 0.05),
        }
        return sizes.get(debris_type, 0.05)

    def _create_debris_body(self, pos, debris_type):
        radius = self._get_type_radius(debris_type)
        grey = 0.4 + random.random() * 0.4
        if debris_type == "rock":
            sx = radius * random.uniform(0.8, 1.2)
            sy = radius * random.uniform(0.8, 1.2)
            sz = radius * random.uniform(0.8, 1.2)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, sz])
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[sx, sy, sz],
                rgbaColor=[grey, grey * 0.9, grey * 0.8, 0.9]
            )
        elif debris_type == "panel":
            sx = radius * 3
            sy = radius * 0.5
            sz = radius * 2
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, sz])
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[sx, sy, sz],
                rgbaColor=[grey * 0.5, grey * 0.6, grey, 0.8]
            )
        elif debris_type == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=radius * 3)
            vis = p.createVisualShape(
                p.GEOM_CYLINDER, radius=radius, length=radius * 3,
                rgbaColor=[grey * 0.7, grey * 0.7, grey, 0.9]
            )
        else:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=radius,
                rgbaColor=[grey + 0.1, grey, grey - 0.1, 0.85]
            )
        body_id = p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        return body_id

    def update_positions(self):
        for d in self.debris_list:
            pos, orn = p.getBasePositionAndOrientation(d.body_id)
            d.position = list(pos)
            vel, ang_vel = p.getBaseVelocity(d.body_id)
            d.velocity = list(vel)
            d.spin = list(ang_vel)

    def update_debris_velocity(self, debris_id, new_velocity):
        for d in self.debris_list:
            if d.body_id == debris_id:
                d.velocity = list(new_velocity)
                p.resetBaseVelocity(d.body_id, linearVelocity=new_velocity)
                return True
        return False

    def update_all_velocities(self, velocity_scale):
        for d in self.debris_list:
            speed = velocity_scale * random.uniform(0.5, 1.5)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(-0.5, 0.5)
            vel = [
                speed * math.cos(theta) * math.cos(phi),
                speed * math.sin(theta) * math.cos(phi),
                speed * math.sin(phi),
            ]
            d.velocity = vel
            p.resetBaseVelocity(d.body_id, linearVelocity=vel)

    def add_debris(self, position, velocity, spin):
        debris_type = random.choice(["rock", "panel", "cylinder", "fragment"])
        body_id = self._create_debris_body(position, debris_type)
        p.resetBaseVelocity(body_id, linearVelocity=velocity, angularVelocity=spin)
        radius = self._get_type_radius(debris_type)
        d = Debris(position, velocity, spin, radius, body_id, debris_type)
        self.debris_list.append(d)
        return d

    def remove_debris(self, debris_id):
        for i, d in enumerate(self.debris_list):
            if d.body_id == debris_id:
                p.removeBody(debris_id)
                self.debris_list.pop(i)
                return True
        return False

    def clear_all(self):
        for d in self.debris_list:
            p.removeBody(d.body_id)
        self.debris_list = []

    def check_collisions(self, arm_id):
        colliding = []
        arm_body_ids = self._get_arm_body_ids(arm_id)
        for d in self.debris_list:
            pts = p.getContactPoints(bodyA=d.body_id, bodyB=-1)
            for pt in pts:
                other = pt[2]
                if other in arm_body_ids:
                    colliding.append({
                        "debris_body": d.body_id,
                        "debris_type": d.debris_type,
                        "other_body": other,
                        "contact_pos": list(pt[6]) if pt[6] else None,
                    })
        arm_bodies = arm_body_ids if isinstance(arm_body_ids, list) else [arm_body_ids]
        for ab in arm_bodies:
            pts = p.getContactPoints(bodyA=ab, bodyB=-1)
            for pt in pts:
                other = pt[2]
                for d in self.debris_list:
                    if other == d.body_id:
                        already = any(x["debris_body"] == d.body_id for x in colliding)
                        if not already:
                            colliding.append({
                                "debris_body": d.body_id,
                                "debris_type": d.debris_type,
                                "other_body": ab,
                                "contact_pos": list(pt[6]) if pt[6] else None,
                            })
        return colliding

    def _get_arm_body_ids(self, arm_id):
        if arm_id is None:
            return []
        if isinstance(arm_id, dict):
            ids = [arm_id.get("body_id", -1)]
            ids.extend(arm_id.get("links", []))
            return ids
        return [arm_id]

    def get_all_debris_positions(self):
        return [list(d.position) for d in self.debris_list]

    def get_count(self):
        return len(self.debris_list)
