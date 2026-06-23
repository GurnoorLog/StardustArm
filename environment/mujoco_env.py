import mujoco
import glfw
import numpy as np
import os
import time

from config import (
    VISION_WIDTH, VISION_HEIGHT, VISION_CAM_NAME,
    BALL_PRIORITIES, FLOATING_BASE, BASE_THRUST_KP, BASE_THRUST_MAX,
    COLLISION_MARGIN, JOINT_LIMITS, TRAJECTORY_STEPS,
)

_SCENE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "robots", "mephi_arm", "scene.xml"
)


class MuJoCoEnv:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.window = None
        self.model = None
        self.data = None
        self.scene = None
        self.context = None
        self.cam = None
        self.opt = None
        self.pert = None

        self._mouse_left = False
        self._mouse_left_camera = False
        self._mouse_right = False
        self._mouse_middle = False
        self._last_mx = 0.0
        self._last_my = 0.0
        self._keys = {}
        self._keys_prev = {}

        self._ball_bodies = {}
        self._ee_body_id = -1
        self._gripper_body_id = -1
        self._finger_body_id = -1
        self._obstacle_ids = []
        self._joint_qposadrs = []
        self._joint_dofadrs = []
        self._finger_adr = -1
        self._finger_dof = -1
        self._base_jnt_adr = None
        self._base_dof_adr = None

        self._trajectory = None
        self._traj_idx = 0
        self._ik_target = None
        self._selected_target = "primary"

        self._vision_scene = None
        self._vision_ctx = None
        self._vision_cam_id = -1

        self._grab_eq_ids = {}
        self._ball_geom_ids = {}

    def _get_body_id(self, name):
        for i in range(self.model.nbody):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if n == name:
                return i
        return -1

    def _init_caches(self):
        for name in ["ball_primary", "ball_secondary"]:
            bid = self._get_body_id(name)
            if bid >= 0:
                self._ball_bodies[name] = bid
        self._gripper_body_id = self._get_body_id("gripper_base")
        self._finger_body_id = self._get_body_id("finger")

        for oname in ["obstacle1", "obstacle2", "obstacle3"]:
            bid = self._get_body_id(oname)
            if bid >= 0:
                self._obstacle_ids.append(bid)

        for jn in ["j1", "j2", "j3", "j4", "j5", "j6"]:
            for i in range(self.model.njnt):
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if n == jn:
                    self._joint_qposadrs.append(self.model.jnt_qposadr[i])
                    self._joint_dofadrs.append(self.model.jnt_dofadr[i])
                    break

        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n == "joint_finger":
                self._finger_adr = self.model.jnt_qposadr[i]
                self._finger_dof = self.model.jnt_dofadr[i]
                break

        for i in range(self.model.ncam):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if n == VISION_CAM_NAME:
                self._vision_cam_id = i
                break

        for i in range(self.model.neq):
            eq_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if eq_name == "grab_weld_primary":
                self._grab_eq_ids["ball_primary"] = i
            elif eq_name == "grab_weld_secondary":
                self._grab_eq_ids["ball_secondary"] = i

        for name in ["ball_primary", "ball_secondary"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            if gid >= 0:
                self._ball_geom_ids[name] = gid

    def load(self, floating_base=FLOATING_BASE):
        old_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(_SCENE_PATH))
            self.model = mujoco.MjModel.from_xml_path(
                os.path.basename(_SCENE_PATH)
            )
        finally:
            os.chdir(old_cwd)
        self.data = mujoco.MjData(self.model)

        self._init_caches()

        if floating_base:
            self._enable_floating_base()

        mujoco.mj_forward(self.model, self.data)
        return self

    def _enable_floating_base(self):
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                self._base_jnt_adr = self.model.jnt_qposadr[i]
                self._base_dof_adr = self.model.jnt_dofadr[i]
                self.model.jnt_solref[i] = [0.0, 1.0]
                return

    def init_viewer(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.FOCUSED, glfw.TRUE)
        self.window = glfw.create_window(
            self.width, self.height, "Stardance — MuJoCo", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        self.cam.distance = 3.5
        self.cam.azimuth = 45
        self.cam.elevation = 30
        self.cam.lookat[:] = [0.0, 0.0, 0.3]

        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        self.context = mujoco.MjrContext(
            self.model, int(mujoco.mjtFontScale.mjFONTSCALE_150)
        )

        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)

        if self._vision_cam_id >= 0:
            self._vision_scene = mujoco.MjvScene(self.model, maxgeom=1000)
            self._vision_ctx = mujoco.MjrContext(
                self.model, int(mujoco.mjtFontScale.mjFONTSCALE_150)
            )

    def _compute_camera_pos(self):
        az = np.radians(self.cam.azimuth)
        el = np.radians(self.cam.elevation)
        d = self.cam.distance
        return self.cam.lookat + d * np.array([
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
            np.sin(el)
        ])

    def _screen_to_ray(self, mx, my):
        cam_pos = self._compute_camera_pos()
        w, h = glfw.get_window_size(self.window)
        if w == 0 or h == 0:
            return cam_pos, np.array([0.0, 0.0, -1.0])
        nx = 2.0 * mx / w - 1.0
        ny = 1.0 - 2.0 * my / h
        forward = self.cam.lookat - cam_pos
        fwd_norm = np.linalg.norm(forward)
        if fwd_norm < 1e-8:
            return cam_pos, np.array([0.0, 0.0, -1.0])
        forward /= fwd_norm
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= r_norm
        up = np.cross(right, forward)
        u_norm = np.linalg.norm(up)
        if u_norm < 1e-8:
            up = np.array([0.0, 1.0, 0.0])
        else:
            up /= u_norm
        fov = 45.0
        aspect = w / h
        tan_fov = np.tan(np.radians(fov / 2.0))
        ray_cam = np.array([nx * aspect * tan_fov, ny * tan_fov, -1.0])
        rc_norm = np.linalg.norm(ray_cam)
        if rc_norm > 1e-8:
            ray_cam /= rc_norm
        ray_world = (ray_cam[0] * right + ray_cam[1] * up
                     - ray_cam[2] * forward)
        rw_norm = np.linalg.norm(ray_world)
        if rw_norm > 1e-8:
            ray_world /= rw_norm
        else:
            ray_world = -forward
        return cam_pos, ray_world

    def _ray_plane_intersect(self, origin, direction, plane_z):
        if abs(direction[2]) < 1e-8:
            return None
        t = (plane_z - origin[2]) / direction[2]
        if t < 0:
            return None
        return origin + t * direction

    def _on_mouse_button(self, window, button, action, mods):
        x, y = glfw.get_cursor_pos(window)
        self._last_mx = x
        self._last_my = y
        ctrl = (mods & glfw.MOD_CONTROL) != 0

        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_left = (action == glfw.PRESS)
            self._mouse_left_camera = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_right = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._mouse_middle = (action == glfw.PRESS)

    def _on_mouse_move(self, window, x, y):
        dx = x - self._last_mx
        dy = y - self._last_my
        self._last_mx = x
        self._last_my = y
        if self._mouse_left_camera:
            self.cam.azimuth += dx * 0.3
            self.cam.elevation = np.clip(
                self.cam.elevation - dy * 0.3, -89.0, 89.0
            )
        elif self._mouse_right:
            self.cam.distance += dy * 0.05
            self.cam.distance = np.clip(self.cam.distance, 0.5, 20.0)
        elif self._mouse_middle:
            forward = self.cam.lookat - self._compute_camera_pos()
            fwd_norm = np.linalg.norm(forward)
            if fwd_norm > 1e-8:
                forward /= fwd_norm
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(world_up, forward)
            rn = np.linalg.norm(right)
            if rn > 1e-8:
                right /= rn
            up = np.cross(right, forward)
            un = np.linalg.norm(up)
            if un > 1e-8:
                up /= un
            speed = self.cam.distance * 0.002
            self.cam.lookat -= right * dx * speed
            self.cam.lookat += up * dy * speed

    def _on_scroll(self, window, x_offset, y_offset):
        self.cam.distance -= y_offset * 0.3
        self.cam.distance = np.clip(self.cam.distance, 0.5, 20.0)

    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self._keys[key] = True
        elif action == glfw.RELEASE:
            self._keys[key] = False

    def is_key_pressed(self, key):
        return self._keys.get(key, False)

    def is_key_triggered(self, key):
        return self._keys.get(key, False) and not self._keys_prev.get(key, False)

    def poll_keys(self):
        self._keys_prev = dict(self._keys)

    def get_glfw_key(self, name):
        return getattr(glfw, f"KEY_{name}", None)

    def step(self):
        mujoco.mj_step(self.model, self.data)
        for i, adr in enumerate(self._joint_qposadrs):
            val = self.data.qpos[adr]
            low, high = JOINT_LIMITS[i]
            if np.isnan(val) or np.isinf(val):
                val = (low + high) * 0.5
            self.data.qpos[adr] = np.clip(val, low, high)

    def render(self):
        if self.window is None:
            return
        vp_w, vp_h = glfw.get_framebuffer_size(self.window)
        if vp_w == 0 or vp_h == 0:
            return
        viewport = mujoco.MjrRect(0, 0, vp_w, vp_h)
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, self.pert, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def is_running(self):
        if self.window is None:
            return False
        return not glfw.window_should_close(self.window)

    def should_close(self):
        return not self.is_running()

    def close(self):
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def get_ee_pos(self):
        return self.data.xpos[self._gripper_body_id].copy()

    def get_ee_body_id(self):
        return self._gripper_body_id

    def get_joint_angles(self):
        return [self.data.qpos[adr] for adr in self._joint_qposadrs]

    def get_joint_qposadrs(self):
        return self._joint_qposadrs

    def get_joint_dofadrs(self):
        return self._joint_dofadrs

    def set_joint_target(self, joint_idx, angle_rad):
        if joint_idx < 0 or joint_idx >= self.model.nu - 1:
            return
        self.data.ctrl[joint_idx] = float(angle_rad)

    def set_finger_target(self, pos):
        self.data.ctrl[self.model.nu - 1] = float(pos)

    def get_finger_pos(self):
        if self._finger_adr < 0:
            return 0.0
        return float(self.data.qpos[self._finger_adr])

    def grab_ball(self, ball_name):
        eq_id = self._grab_eq_ids.get(ball_name)
        if eq_id is None:
            return
        
        # Teleport the ball to gripper_base center to avoid sudden force jump
        g_id = self._gripper_body_id
        bid = self._ball_bodies.get(ball_name)
        if bid is not None:
            ball_jnt_id = -1
            for i in range(self.model.njnt):
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if n == f"{ball_name}_free":
                    ball_jnt_id = i
                    break
            
            if ball_jnt_id >= 0:
                q_adr = self.model.jnt_qposadr[ball_jnt_id]
                v_adr = self.model.jnt_dofadr[ball_jnt_id]
                
                # Teleport: Position = gripper_base position
                self.data.qpos[q_adr : q_adr+3] = self.data.xpos[g_id]
                
                # Teleport: Orientation = gripper_base orientation quaternion
                quat_g = np.zeros(4)
                mujoco.mju_mat2Quat(quat_g, self.data.xmat[g_id].flatten())
                self.data.qpos[q_adr+3 : q_adr+7] = quat_g
                
                # Zero ALL velocities on the ball so there is no velocity mismatch
                self.data.qvel[v_adr : v_adr+6] = 0.0
                
                mujoco.mj_forward(self.model, self.data)

        # Disable ball geom collisions so the contact solver cannot fight the weld
        gid = self._ball_geom_ids.get(ball_name)
        if gid is not None:
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0

        self.model.eq_data[eq_id, 0:3] = [0.0, 0.0, 0.0]
        self.model.eq_data[eq_id, 3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.eq_active[eq_id] = 1

    def release_ball(self, ball_name):
        eq_id = self._grab_eq_ids.get(ball_name)
        if eq_id is None:
            return
        # Zero ball velocity so it doesn't carry gripper momentum on release
        bid = self._ball_bodies.get(ball_name)
        if bid is not None:
            for i in range(self.model.njnt):
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if n == f"{ball_name}_free":
                    v_adr = self.model.jnt_dofadr[i]
                    self.data.qvel[v_adr : v_adr+6] = 0.0
                    break
        self.data.eq_active[eq_id] = 0
        # NOTE: collision stays DISABLED (contype=0) after release so the ball
        # cannot slam into the floor and cause QACC instability. Collisions are
        # restored at the start of the next episode via reset_ball_geom_collisions().

    def reset_ball_geom_collisions(self):
        """Restore physics collisions on all target balls. Call at episode start."""
        for ball_name, gid in self._ball_geom_ids.items():
            self.model.geom_contype[gid] = 1
            self.model.geom_conaffinity[gid] = 1

    def reset_arm(self):
        for adr in self._joint_qposadrs:
            self.data.qpos[adr] = 0.0
        for adr in self._joint_dofadrs:
            self.data.qvel[adr] = 0.0
        self.set_finger_target(0.037)
        for _ in range(10):
            self.step()
        mujoco.mj_forward(self.model, self.data)

    def compute_ee_for_angles(self, joint_angles):
        saved = self.data.qpos.copy()
        for i, adr in enumerate(self._joint_qposadrs):
            if i < len(joint_angles):
                self.data.qpos[adr] = joint_angles[i]
        mujoco.mj_forward(self.model, self.data)
        ee = self.data.xpos[self._gripper_body_id].copy()
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return ee

    def check_links_above(self, threshold=0.05):
        for name in ["link2", "link3", "link4", "link5", "link6", "link7",
                      "gripper_base", "finger"]:
            bid = self._get_body_id(name)
            if bid >= 0 and self.data.xpos[bid][2] < threshold:
                return False
        return True

    def check_links_above_for_angles(self, test_angles, threshold=0.05):
        saved = self.data.qpos.copy()
        for i, adr in enumerate(self._joint_qposadrs):
            if i < len(test_angles):
                self.data.qpos[adr] = test_angles[i]
        mujoco.mj_forward(self.model, self.data)
        ok = self.check_links_above(threshold)
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return ok

    def compute_ee_for_action(self, joint_angles, action):
        j, d = action["joint"], action["delta"]
        test = list(joint_angles)
        test[j] += d
        return self.compute_ee_for_angles(test)

    # --- Ball handling ---
    def get_ball_names(self):
        return sorted(self._ball_bodies.keys())

    def get_ball_pos(self, name="ball_primary"):
        bid = self._ball_bodies.get(name)
        if bid is None:
            bid = self._ball_bodies.get("ball_primary")
        if bid is None:
            return np.array([0.0, 0.0, 0.06])
        return self.data.xpos[bid].copy()

    def get_primary_ball_pos(self):
        return self.get_ball_pos("ball_primary")

    def set_ball_pos(self, pos, name="ball_primary"):
        bid = self._ball_bodies.get(name)
        if bid is None:
            return
        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n == f"{name}_free":
                adr = self.model.jnt_qposadr[i]
                self.data.qpos[adr:adr+3] = pos
                self.data.qvel[self.model.jnt_dofadr[i]:
                               self.model.jnt_dofadr[i]+3] = 0.0
                return

    def get_all_ball_positions_with_priority(self):
        results = []
        for bname, bid in self._ball_bodies.items():
            pos = self.data.xpos[bid].copy()
            priority = BALL_PRIORITIES.get(bname, 0)
            results.append({
                "name": bname, "pos": pos,
                "priority": priority,
            })
        results.sort(key=lambda r: -r["priority"])
        return results

    def select_target_by_priority(self, ee_pos, min_dist=0.0):
        balls = self.get_all_ball_positions_with_priority()
        for b in balls:
            dist = float(np.linalg.norm(ee_pos - b["pos"]))
            if dist > min_dist:
                self._selected_target = b["name"]
                return b["name"], b["pos"]
        return balls[0]["name"], balls[0]["pos"]

    def get_selected_target_name(self):
        return self._selected_target

    # --- Obstacles ---
    def get_obstacle_positions(self):
        return [self.data.xpos[oid].copy() for oid in self._obstacle_ids]

    def get_obstacle_radius(self):
        return 0.08

    def check_collision(self, margin=COLLISION_MARGIN):
        for oid in self._obstacle_ids:
            obs_pos = self.data.xpos[oid]
            for link_name in ["link2", "link3", "link4", "link5", "link6",
                              "link7", "gripper_base", "finger"]:
                lid = self._get_body_id(link_name)
                if lid < 0:
                    continue
                dist = float(np.linalg.norm(
                    self.data.xpos[lid] - obs_pos
                ))
                if dist < margin:
                    return True
        return False

    def check_collision_for_angles(self, test_angles, margin=COLLISION_MARGIN):
        saved = self.data.qpos.copy()
        for i, adr in enumerate(self._joint_qposadrs):
            if i < len(test_angles):
                self.data.qpos[adr] = test_angles[i]
        mujoco.mj_forward(self.model, self.data)
        collision = self.check_collision(margin)
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return collision

    # --- Trajectory ---
    def set_trajectory(self, waypoints):
        self._trajectory = waypoints
        self._traj_idx = 0

    def has_trajectory(self):
        return self._trajectory is not None and self._traj_idx < len(self._trajectory)

    def advance_trajectory(self):
        if self._trajectory is None or self._traj_idx >= len(self._trajectory):
            return None
        pt = self._trajectory[self._traj_idx]
        self._traj_idx += 1
        return pt

    def trajectory_progress(self):
        if self._trajectory is None:
            return 1.0
        return self._traj_idx / max(len(self._trajectory), 1)

    # --- Vision ---
    def get_vision_image(self):
        if self._vision_cam_id < 0 or self._vision_scene is None:
            return None
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.fixedcamid = self._vision_cam_id
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, self._vision_scene
        )
        viewport = mujoco.MjrRect(0, 0, VISION_WIDTH, VISION_HEIGHT)
        mujoco.mjr_render(viewport, self._vision_scene, self._vision_ctx)
        rgb = np.zeros((VISION_HEIGHT, VISION_WIDTH, 3), dtype=np.uint8)
        depth = np.zeros((VISION_HEIGHT, VISION_WIDTH), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, viewport, self._vision_ctx)
        return rgb.copy()

    # --- Floating base ---
    def apply_base_thrust(self, force_xyz):
        if self._base_dof_adr is None:
            return
        for i in range(3):
            adr = self._base_dof_adr + i
            if adr < self.model.nv:
                self.data.qfrc_applied[adr] = np.clip(
                    force_xyz[i], -BASE_THRUST_MAX, BASE_THRUST_MAX
                )

    def stabilize_base(self, target_xy=(0.0, 0.0)):
        if self._base_jnt_adr is None:
            return
        dx = target_xy[0] - self.data.qpos[self._base_jnt_adr]
        dy = target_xy[1] - self.data.qpos[self._base_jnt_adr + 1]
        self.apply_base_thrust([
            dx * BASE_THRUST_KP,
            dy * BASE_THRUST_KP,
            -self.data.qvel[self._base_dof_adr + 2] * BASE_THRUST_KP * 0.5,
        ])

    def set_floating_base_pose(self, pos, quat=(1, 0, 0, 0)):
        if self._base_jnt_adr is not None:
            self.data.qpos[self._base_jnt_adr:self._base_jnt_adr+3] = pos
            self.data.qpos[self._base_jnt_adr+3:self._base_jnt_adr+7] = quat
            for i in range(6):
                if self._base_dof_adr + i < self.model.nv:
                    self.data.qvel[self._base_dof_adr + i] = 0.0
