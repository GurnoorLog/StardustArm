import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os
import xml.etree.ElementTree as ET
import uuid

SCENE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "robots", "mephi_arm", "scene.xml"
)

JOINT_LIMITS = [
    (-1.5708, 1.5708), (-1.5708, 1.5708), (-1.5708, 1.5708),
    (-1.5708, 1.5708), (-1.5708, 1.5708), (-3.1415, 3.1415),
]

FINGER_OPEN = 0.037
FINGER_CLOSED = 0.0
GRAB_DIST = 0.15
MAX_STEPS = 5000
TARGET_NAMES = ["ball_primary", "ball_secondary"]


class SpaceshipArmEnv(gym.Env):

    def __init__(self, floating_base=False, space_gravity=True, render_mode=None):
        super().__init__()
        self.floating_base = floating_base
        self.render_mode = render_mode
        self._window = None

        xml_path = self._prepare_xml(floating_base, space_gravity)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._cache_ids()

        # HER-compatible Dict observation
        obs_dim = 6 + 6 + 3 + 1 + 7  # joints + vel + ee + finger + ctrl
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        ))

        n_act = 7 if not floating_base else 13
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)

        self._current_target = None
        self._step_count = 0
        self._grabbed = False
        self._prev_ctrl = np.zeros(7, dtype=np.float32)
        self._desired_goal = np.zeros(3, dtype=np.float32)

        if render_mode == "human":
            self._init_viewer()

    def _prepare_xml(self, floating_base, space_gravity):
        tree = ET.parse(SCENE_PATH)
        root = tree.getroot()
        option = root.find("option")
        if space_gravity:
            option.set("gravity", "0 0 0")
        else:
            option.set("gravity", "0 0 -9.81")

        if floating_base:
            worldbody = root.find("worldbody")
            arm_body = worldbody.find("body[@name='link1']")
            if arm_body is not None:
                arm_body.set("pos", "0 0 0")
                fw = arm_body.find("joint[@name='free_joint']")
                if fw is None:
                    free_jnt = ET.SubElement(arm_body, "joint")
                    free_jnt.set("name", "free_joint")
                    free_jnt.set("type", "free")
            actuator = root.find("actuator")
            if actuator is not None:
                for axis, vec in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]:
                    thr = ET.SubElement(actuator, "motor")
                    thr.set("name", f"thruster_{axis}"); thr.set("joint", "free_joint")
                    thr.set("axis", vec); thr.set("gear", "50.0"); thr.set("forcerange", "-100 100")
                for axis, vec in [("rx", "1 0 0"), ("ry", "0 1 0"), ("rz", "0 0 1")]:
                    thr = ET.SubElement(actuator, "motor")
                    thr.set("name", f"thruster_{axis}"); thr.set("joint", "free_joint")
                    thr.set("axis", vec); thr.set("gear", "10.0"); thr.set("forcerange", "-20 20")
        tag = uuid.uuid4().hex[:8]
        out = os.path.join(os.path.dirname(SCENE_PATH), f"scene_rl_{tag}.xml")
        tree.write(out)
        return out

    def _cache_ids(self):
        self._gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self._target_ids = {}
        for name in TARGET_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._target_ids[name] = bid
        self._target_names = list(self._target_ids.keys())
        self._ball_geom_ids = {}
        for name in TARGET_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            if gid >= 0:
                self._ball_geom_ids[name] = gid

        self._joint_qposadrs = []
        self._joint_dofadrs = []
        for jn in ["j1", "j2", "j3", "j4", "j5", "j6"]:
            for i in range(self.model.njnt):
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if n == jn:
                    self._joint_qposadrs.append(self.model.jnt_qposadr[i])
                    self._joint_dofadrs.append(self.model.jnt_dofadr[i])
                    break

        self._finger_adr = self._finger_dof = None
        self._finger_geom_id = None
        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n == "joint_finger":
                self._finger_adr = self.model.jnt_qposadr[i]
                self._finger_dof = self.model.jnt_dofadr[i]
        self._finger_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "finger_vis")
        if self._finger_geom_id >= 0:
            self.model.geom_rgba[self._finger_geom_id] = [1.0, 0.2, 0.2, 1.0]
        self._gripper_base_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper_base_vis")

        self._grab_eq_ids = {}
        for i in range(self.model.neq):
            eq_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if eq_name == "grab_weld_primary":
                self._grab_eq_ids["ball_primary"] = i
            elif eq_name == "grab_weld_secondary":
                self._grab_eq_ids["ball_secondary"] = i

        self._free_joint_adr = self._free_dof_adr = None
        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n == "free_joint":
                self._free_joint_adr = self.model.jnt_qposadr[i]
                self._free_dof_adr = self.model.jnt_dofadr[i]
                break

    def _find_ball_free_joint(self, ball_name):
        for i in range(self.model.njnt):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) == f"{ball_name}_free":
                return i
        return -1

    def _set_ball_position(self, ball_name, pos):
        jnt_id = self._find_ball_free_joint(ball_name)
        if jnt_id < 0:
            return
        q_adr = self.model.jnt_qposadr[jnt_id]
        self.data.qpos[q_adr:q_adr+3] = pos
        self.data.qpos[q_adr+3:q_adr+7] = [1.0, 0.0, 0.0, 0.0]

    def _get_state(self):
        return np.concatenate([
            np.array([self.data.qpos[adr] for adr in self._joint_qposadrs], dtype=np.float32),
            np.array([self.data.qvel[dof] for dof in self._joint_dofadrs], dtype=np.float32),
            self.data.xpos[self._gripper_id].copy().astype(np.float32),
            np.array([self.data.qpos[self._finger_adr]], dtype=np.float32) if self._finger_adr is not None else np.zeros(1, dtype=np.float32),
            self._prev_ctrl,
        ])

    def _make_obs(self):
        ee = self.data.xpos[self._gripper_id].copy().astype(np.float32)
        return {
            "observation": self._get_state(),
            "desired_goal": self._desired_goal.copy(),
            "achieved_goal": ee,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -d

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        for name in self._ball_geom_ids:
            gid = self._ball_geom_ids[name]
            self.model.geom_contype[gid] = 1
            self.model.geom_conaffinity[gid] = 1
            self.model.geom_rgba[gid] = [0.9, 0.1, 0.1, 1.0]
        if self._gripper_base_geom_id >= 0:
            self.model.geom_rgba[self._gripper_base_geom_id] = [0.0, 0.0, 0.8, 1.0]
        if self._finger_geom_id is not None and self._finger_geom_id >= 0:
            self.model.geom_rgba[self._finger_geom_id] = [1.0, 0.2, 0.2, 1.0]
        for eq_id in self._grab_eq_ids.values():
            self.data.eq_active[eq_id] = 0

        for i, (low, high) in enumerate(JOINT_LIMITS):
            self.data.qpos[self._joint_qposadrs[i]] = self.np_random.uniform(low * 0.05, high * 0.05)
        if self._finger_adr is not None:
            self.data.qpos[self._finger_adr] = FINGER_OPEN

        if self._free_joint_adr is not None:
            self.data.qpos[self._free_joint_adr:self._free_joint_adr+3] = [0.0, 0.0, 0.5]
            self.data.qpos[self._free_joint_adr+3:self._free_joint_adr+7] = [1.0, 0.0, 0.0, 0.0]

        target_name = self._target_names[int(self.np_random.choice(len(self._target_names)))]
        self._current_target = target_name

        for name in TARGET_NAMES:
            gid = self._ball_geom_ids.get(name)
            if gid is not None:
                if name == target_name:
                    self.model.geom_rgba[gid][3] = 1.0
                else:
                    self.model.geom_rgba[gid][3] = 0.0

        theta = self.np_random.uniform(0, 2 * np.pi)
        r = self.np_random.uniform(0.1, 0.3)
        z = self.np_random.uniform(0.10, 0.20)
        target_pos = np.array([r * np.cos(theta), r * np.sin(theta), z])
        self._desired_goal = target_pos.copy().astype(np.float32)
        self._set_ball_position(target_name, target_pos)
        for name in TARGET_NAMES:
            if name != target_name:
                hidden_pos = np.array([99.0, 99.0, 99.0])
                self._set_ball_position(name, hidden_pos)

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._grabbed = False
        self._prev_ctrl.fill(0.0)

        return self._make_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        delta_scale = 0.05

        for i in range(6):
            low, high = JOINT_LIMITS[i]
            self._prev_ctrl[i] += action[i] * delta_scale * (high - low)
            self._prev_ctrl[i] = np.clip(self._prev_ctrl[i], low, high)
            self.data.ctrl[i] = self._prev_ctrl[i]

        if self._grabbed and self._finger_adr is not None:
            self.data.qpos[self._finger_adr] = FINGER_CLOSED
            self._prev_ctrl[6] = 0.0
        else:
            delta_finger = action[6] * delta_scale * FINGER_OPEN
            self._prev_ctrl[6] = np.clip(self._prev_ctrl[6] + delta_finger, 0.0, FINGER_OPEN)
        self.data.ctrl[6] = self._prev_ctrl[6]

        if self.floating_base and self._free_joint_adr is not None:
            for j in range(6):
                self.data.ctrl[7 + j] = action[7 + j] * 50.0 if j < 3 else action[7 + j] * 20.0

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        for name in TARGET_NAMES:
            jnt_id = self._find_ball_free_joint(name)
            if jnt_id >= 0:
                q_adr = self.model.jnt_qposadr[jnt_id]
                for k in range(3):
                    if not np.isfinite(self.data.qpos[q_adr + k]):
                        self.data.qpos[q_adr + k] = 0.0
                self.data.qpos[q_adr + 3:q_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        ee_pos = self.data.xpos[self._gripper_id]
        target_pos = self.data.xpos[self._target_ids[self._current_target]]
        dist = float(np.linalg.norm(ee_pos - target_pos))

        grabbed_now = dist < GRAB_DIST and not self._grabbed
        if grabbed_now:
            self._grabbed = True
            eq_id = self._grab_eq_ids.get(self._current_target)
            if eq_id is not None:
                self.data.eq_active[eq_id] = 1
            gid = self._ball_geom_ids.get(self._current_target)
            if gid is not None:
                self.model.geom_rgba[gid] = [0.0, 1.0, 0.0, 1.0]
            if self._gripper_base_geom_id >= 0:
                self.model.geom_rgba[self._gripper_base_geom_id] = [0.0, 1.0, 0.0, 1.0]
            if self._finger_geom_id is not None and self._finger_geom_id >= 0:
                self.model.geom_rgba[self._finger_geom_id] = [0.0, 1.0, 0.0, 1.0]

        truncated = self._step_count >= MAX_STEPS
        terminated = self._grabbed

        info = {
            "dist": dist,
            "grabbed": self._grabbed,
            "is_success": float(self._grabbed),
            "target": self._current_target,
            "step": self._step_count,
        }

        obs = self._make_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._window is None:
            self._init_viewer()
        import glfw
        if glfw.window_should_close(self._window):
            return
        mujoco.mjv_updateScene(self.model, self.data, self._opt, self._pert, self._cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self._scene)
        vp_w, vp_h = glfw.get_framebuffer_size(self._window)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, vp_w, vp_h), self._scene, self._context)
        glfw.swap_buffers(self._window)
        glfw.poll_events()

    def is_running(self):
        import glfw
        if self._window is None:
            return True
        return not glfw.window_should_close(self._window)

    def _init_viewer(self):
        import glfw
        glfw.init()
        self._window = glfw.create_window(800, 600, "Spaceship Arm", None, None)
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)
        self._opt = mujoco.MjvOption()
        self._pert = mujoco.MjvPerturb()
        self._cam = mujoco.MjvCamera()
        self._scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self._context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjv_defaultFreeCamera(self.model, self._cam)
        self._cam.distance = 0.6
        self._cam.lookat[:] = [0.0, 0.0, 0.15]
        self._cam.elevation = -20
        self._cam.azimuth = 90

        self._mouse_left = False
        self._mouse_middle = False
        self._mouse_right = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        def mouse_callback(w, button, action, mods):
            self._last_mouse_x, self._last_mouse_y = glfw.get_cursor_pos(self._window)
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._mouse_left = action == glfw.PRESS
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                self._mouse_middle = action == glfw.PRESS
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self._mouse_right = action == glfw.PRESS

        def scroll_callback(w, xoffset, yoffset):
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, yoffset * 0.05, self._scene, self._cam)

        def move_callback(w, xpos, ypos):
            dx = xpos - self._last_mouse_x
            dy = ypos - self._last_mouse_y
            self._last_mouse_x = xpos
            self._last_mouse_y = ypos
            if self._mouse_left:
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx, dy, self._scene, self._cam)
            elif self._mouse_middle:
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx, dy, self._scene, self._cam)
            elif self._mouse_right:
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, dy * 0.05, self._scene, self._cam)

        glfw.set_mouse_button_callback(self._window, mouse_callback)
        glfw.set_scroll_callback(self._window, scroll_callback)
        glfw.set_cursor_pos_callback(self._window, move_callback)

    def close(self):
        if self._window is not None:
            import glfw
            glfw.destroy_window(self._window)
            glfw.terminate()
            self._window = None
