import numpy as np
import mujoco
from config import VISION_WIDTH, VISION_HEIGHT, BALL_COLORS, BALL_PRIORITIES


class VisionSystem:
    def __init__(self, env):
        self.env = env
        self.cam_id = -1
        self.scene = None
        self.context = None
        self.rgb_buffer = np.zeros((VISION_HEIGHT, VISION_WIDTH, 3), dtype=np.uint8)
        self.depth_buffer = np.zeros((VISION_HEIGHT, VISION_WIDTH), dtype=np.float32)

    def _init_camera(self):
        model = self.env.model
        for i in range(model.ncam):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if n == "vision_cam":
                self.cam_id = i
                break
        if self.cam_id < 0:
            return False
        self.scene = mujoco.MjvScene(model, maxgeom=1000)
        self.context = mujoco.MjrContext(model, int(
            mujoco.mjtFontScale.mjFONTSCALE_150
        ))
        return True

    def capture(self):
        if self.cam_id < 0:
            if not self._init_camera():
                return None
        model = self.env.model
        data = self.env.data
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.fixedcamid = self.cam_id
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        mujoco.mjv_updateScene(model, data, self.env.opt, None, cam,
                                mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        viewport = mujoco.MjrRect(0, 0, VISION_WIDTH, VISION_HEIGHT)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_readPixels(self.rgb_buffer, self.depth_buffer,
                               viewport, self.context)
        return self.rgb_buffer.copy()

    def detect_balls(self):
        rgb = self.capture()
        if rgb is None:
            return []
        results = []
        for name, (r_tgt, g_tgt, b_tgt) in BALL_COLORS.items():
            priority = BALL_PRIORITIES.get(name, 0)
            mask = (
                (rgb[:, :, 0] > r_tgt - 40) & (rgb[:, :, 0] < r_tgt + 40) &
                (rgb[:, :, 1] > g_tgt - 40) & (rgb[:, :, 1] < g_tgt + 40) &
                (rgb[:, :, 2] > b_tgt - 40) & (rgb[:, :, 2] < b_tgt + 40)
            )
            if np.any(mask):
                ys, xs = np.where(mask)
                cy = int(np.mean(ys))
                cx = int(np.mean(xs))
                area = int(np.sum(mask))
                results.append({
                    "name": name,
                    "color": (r_tgt, g_tgt, b_tgt),
                    "pixel_center": (cx, cy),
                    "area": area,
                    "priority": priority,
                })
        results.sort(key=lambda r: -r["priority"])
        return results

    def pixel_to_world_ray(self, px, py):
        model = self.env.model
        cam_pos = model.cam_pos[self.cam_id].copy()
        forward = model.cam_pos[self.cam_id].copy()
        forward[0] = 0
        forward[1] = 0
        forward[2] = 1
        w = VISION_WIDTH
        h = VISION_HEIGHT
        fov = 45.0
        aspect = w / h
        tan_fov = np.tan(np.radians(fov / 2.0))
        nx = 2.0 * px / w - 1.0
        ny = 1.0 - 2.0 * py / h
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)
        rn = np.linalg.norm(right)
        if rn > 1e-8:
            right /= rn
        up = np.cross(right, forward)
        un = np.linalg.norm(up)
        if un > 1e-8:
            up /= un
        ray = nx * aspect * tan_fov * right + ny * tan_fov * up + forward
        rn = np.linalg.norm(ray)
        if rn > 1e-8:
            ray /= rn
        return cam_pos, ray
