import os
import sys
import time
import threading
import concurrent.futures
os.environ["MUJOCO_GL"] = "egl"
import gradio as gr
import numpy as np
import mujoco
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from agent.ik_controller import solve_ik
from agent.trajectory import minimum_jerk_trajectory

SCENE_PATH = os.path.join(ROOT, "assets", "robots", "mephi_arm", "scene.xml")
BALL_NAMES = ["ball_primary", "ball_secondary"]
GRIPPER_NAME = "gripper_base"

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
OBSTACLE_NAMES = ["obstacle1", "obstacle2", "obstacle3"]

STATE_IDLE = "IDLE"
STATE_REACHING = "REACHING"
STATE_GRABBED = "GRABBED"
STATE_LIFTING = "LIFTING"
STATE_SWINGING = "SWINGING"
STATE_DONE = "DONE"

BALL_PRESETS = {
    "Center": [0.0, 0.45, 0.30],
    "Close": [0.0, 0.30, 0.20],
    "Far": [0.15, 0.65, 0.40],
    "High": [0.0, 0.35, 0.55],
    "Low": [0.0, 0.60, 0.12],
}

DEFAULT_CAM = dict(azimuth=45.0, elevation=30.0, distance=3.5)
JOINT_LIMITS = getattr(config, "JOINT_LIMITS", [(-1.5708, 1.5708)] * 5 + [(-3.1415, 3.1415)])


class SimEnv:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self._cache_ids()

    def _cache_ids(self):
        self._gripper_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_NAME
        )
        self._ball_ids = {}
        for name in BALL_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._ball_ids[name] = bid
        self._joint_qposadrs = []
        self._joint_dofadrs = []
        for jn in JOINT_NAMES:
            for i in range(self.model.njnt):
                n = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, i
                )
                if n == jn:
                    self._joint_qposadrs.append(self.model.jnt_qposadr[i])
                    self._joint_dofadrs.append(self.model.jnt_dofadr[i])
                    break
        self._finger_adr = -1
        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, i
            )
            if n == "joint_finger":
                self._finger_adr = self.model.jnt_qposadr[i]
                break
        self._obstacle_ids = []
        for oname in OBSTACLE_NAMES:
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, oname
            )
            if bid >= 0:
                self._obstacle_ids.append(bid)
        self._ball_geom_ids = {}
        for name in BALL_NAMES:
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom"
            )
            if gid >= 0:
                self._ball_geom_ids[name] = gid
        self._grab_eq_ids = {}
        for i in range(self.model.neq):
            eq_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, i
            )
            if eq_name == "grab_weld_primary":
                self._grab_eq_ids["ball_primary"] = i
            elif eq_name == "grab_weld_secondary":
                self._grab_eq_ids["ball_secondary"] = i

    def get_ee_pos(self):
        return self.data.xpos[self._gripper_id].copy()

    def get_ee_body_id(self):
        return self._gripper_id

    def get_joint_angles(self):
        return [self.data.qpos[adr] for adr in self._joint_qposadrs]

    def get_joint_qposadrs(self):
        return self._joint_qposadrs

    def get_joint_dofadrs(self):
        return self._joint_dofadrs

    def get_finger_qposadr(self):
        return self._finger_adr

    def set_joint_target(self, joint_idx, angle_rad):
        if 0 <= joint_idx < self.model.nu - 1:
            self.data.ctrl[joint_idx] = float(angle_rad)

    def set_finger_target(self, pos):
        self.data.ctrl[self.model.nu - 1] = float(pos)

    def set_joints_instant(self, angles):
        for j_idx, adr in enumerate(self._joint_qposadrs):
            if j_idx < self.model.nu - 1:
                self.data.qpos[adr] = float(angles[j_idx])
                self.set_joint_target(j_idx, float(angles[j_idx]))
        mujoco.mj_forward(self.model, self.data)

    def set_finger_instant(self, pos):
        if self._finger_adr >= 0:
            self.data.qpos[self._finger_adr] = float(pos)
        self.set_finger_target(float(pos))
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_ball_names(self):
        return list(self._ball_ids.keys())

    def get_ball_pos(self, name="ball_primary"):
        bid = self._ball_ids.get(name, next(iter(self._ball_ids.values()), -1))
        if bid < 0:
            return np.array([0.0, 0.3, 0.10])
        return self.data.xpos[bid].copy()

    def set_ball_pos(self, pos, name="ball_primary"):
        bid = self._ball_ids.get(name)
        if bid is None:
            return
        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, i
            )
            if n == f"{name}_free":
                adr = self.model.jnt_qposadr[i]
                self.data.qpos[adr : adr + 3] = pos
                v_adr = self.model.jnt_dofadr[i]
                self.data.qvel[v_adr : v_adr + 3] = 0.0
                return

    def get_obstacle_positions(self):
        return [self.data.xpos[oid].copy() for oid in self._obstacle_ids]

    def reset_arm(self):
        for adr in self._joint_qposadrs:
            self.data.qpos[adr] = 0.0
        for adr in self._joint_dofadrs:
            self.data.qvel[adr] = 0.0
        self.set_finger_target(config.FINGER_OPEN)
        for _ in range(10):
            self.step()
        mujoco.mj_forward(self.model, self.data)

    def grab_ball(self, ball_name):
        eq_id = self._grab_eq_ids.get(ball_name)
        if eq_id is None:
            return
        g_id = self._gripper_id
        bid = self._ball_ids.get(ball_name)
        if bid is not None:
            ball_jnt_id = -1
            for i in range(self.model.njnt):
                n = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, i
                )
                if n == f"{ball_name}_free":
                    ball_jnt_id = i
                    break
            if ball_jnt_id >= 0:
                q_adr = self.model.jnt_qposadr[ball_jnt_id]
                v_adr = self.model.jnt_dofadr[ball_jnt_id]
                self.data.qpos[q_adr : q_adr + 3] = self.data.xpos[g_id]
                quat_g = np.zeros(4)
                mujoco.mju_mat2Quat(quat_g, self.data.xmat[g_id].flatten())
                self.data.qpos[q_adr + 3 : q_adr + 7] = quat_g
                self.data.qvel[v_adr : v_adr + 6] = 0.0
                mujoco.mj_forward(self.model, self.data)
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
        self.data.eq_active[eq_id] = 0

    def sync_grabbed_ball(self, ball_name):
        eq_id = self._grab_eq_ids.get(ball_name)
        if eq_id is None or self.data.eq_active[eq_id] != 1:
            return
        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n == f"{ball_name}_free":
                q_adr = self.model.jnt_qposadr[i]
                v_adr = self.model.jnt_dofadr[i]
                self.data.qpos[q_adr:q_adr+3] = self.data.xpos[self._gripper_id]
                quat_g = np.zeros(4)
                mujoco.mju_mat2Quat(quat_g, self.data.xmat[self._gripper_id].flatten())
                self.data.qpos[q_adr+3:q_adr+7] = quat_g
                self.data.qvel[v_adr:v_adr+6] = 0.0
                break
        mujoco.mj_forward(self.model, self.data)

    def freeze_balls(self, exclude=None):
        for bname in self._ball_ids:
            if bname == exclude:
                continue
            for i in range(self.model.njnt):
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if n == f"{bname}_free":
                    v_adr = self.model.jnt_dofadr[i]
                    self.data.qvel[v_adr:v_adr+6] = 0.0
                    break

    def reset_ball_geom_collisions(self):
        for gid in self._ball_geom_ids.values():
            self.model.geom_contype[gid] = 1
            self.model.geom_conaffinity[gid] = 1


def ik_reach(env, target_pos, speed_scale=0.008, obstacle_pos=None, avoid=False,
             grabbed_ball=None):
    ee = np.array(env.get_ee_pos(), dtype=float)
    tgt = np.array(target_pos, dtype=float)
    direction = tgt - ee
    dist = float(np.linalg.norm(direction))
    if dist < 0.005:
        return
    step = ee + direction * min(1.0, speed_scale / max(dist, 0.001))
    wp = np.array(step, dtype=float)
    joint_config, ik_ok = solve_ik(
        env.model, env.data, wp,
        env.get_ee_body_id(),
        env.get_joint_qposadrs(),
        env.get_joint_dofadrs(),
        q_init=None, damping=config.IK_DAMPING,
        max_iter=config.IK_MAX_ITER, tol=config.IK_TOLERANCE,
    )
    for j_idx, adr in enumerate(env.get_joint_qposadrs()):
        if j_idx < env.model.nu - 1:
            env.data.qpos[adr] = joint_config[adr]
            env.set_joint_target(j_idx, joint_config[adr])
    if grabbed_ball:
        env.sync_grabbed_ball(grabbed_ball)


class Scene:
    # one model/data/env for everyone. rendering is pinned to a single worker
    # thread because mujoco's egl context can only be current on one thread at
    # a time, and gradio fires events off on a thread pool — without this we
    # got EGL_BAD_ACCESS on every other call.

    def __init__(self):
        self.lock = threading.Lock()
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        self.env = SimEnv(self.model, self.data)
        # blue (secondary) ball is retired — hide its geoms instead of touching the scene
        for i in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname and gname.startswith("ball_secondary"):
                self.model.geom_rgba[i][3] = 0.0
        self.camera = mujoco.MjvCamera()
        self.camera.distance = DEFAULT_CAM["distance"]
        self.camera.azimuth = DEFAULT_CAM["azimuth"]
        self.camera.elevation = DEFAULT_CAM["elevation"]
        self.camera.lookat[:] = [0.0, 0.0, 0.3]
        self._renderer = None
        mujoco.mj_forward(self.model, self.data)

    def apply_camera(self, azimuth, elevation, distance):
        self.camera.azimuth = float(azimuth)
        self.camera.elevation = float(elevation)
        self.camera.distance = float(distance)

    def render(self):
        def _do():
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=360, width=480)
            self._renderer.update_scene(self.data, self.camera)
            return self._renderer.render().copy()

        return _RENDER_EXECUTOR.submit(_do).result()


_RENDER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="renderer"
)


_SCENE = None
_SCENE_LOCK = threading.Lock()


def _get_scene():
    global _SCENE
    if _SCENE is None:
        with _SCENE_LOCK:
            if _SCENE is None:
                _SCENE = Scene()
    return _SCENE


_LAST_FRAMES = []
_FRAME_LOCK = threading.Lock()
_RUNNING = False
_RUNNING_LOCK = threading.Lock()


def _remember_frame(frame):
    small = frame[::2, ::2]
    with _FRAME_LOCK:
        _LAST_FRAMES.append(small)
        if len(_LAST_FRAMES) > 1500:
            del _LAST_FRAMES[: len(_LAST_FRAMES) - 1500]


def _set_running(v):
    global _RUNNING
    with _RUNNING_LOCK:
        _RUNNING = bool(v)


def _is_running():
    with _RUNNING_LOCK:
        return _RUNNING


def _render_pose(ball=(0.0, 0.3, 0.10), joints=None, finger=config.FINGER_OPEN,
                 cam=DEFAULT_CAM):
    scene = _get_scene()
    with scene.lock:
        scene.apply_camera(cam["azimuth"], cam["elevation"], cam["distance"])
        scene.env.set_ball_pos(list(ball), "ball_primary")
        scene.env.reset_ball_geom_collisions()
        if joints is not None:
            scene.env.set_joints_instant(list(joints))
            scene.env.set_finger_instant(finger)
        else:
            scene.env.reset_arm()
            scene.env.set_finger_instant(finger)
        return scene.render()


def initial_frame():
    return _render_pose()


def pose_render(j1, j2, j3, j4, j5, j6, finger, bx, by, bz, azimuth, elevation, distance):
    if _is_running():
        return gr.skip()
    return _render_pose(
        ball=(bx, by, bz),
        joints=[j1, j2, j3, j4, j5, j6],
        finger=finger,
        cam=dict(azimuth=azimuth, elevation=elevation, distance=distance),
    )


def apply_preset(preset):
    if _is_running():
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    ball = BALL_PRESETS.get(preset, BALL_PRESETS["Center"])
    frame = _render_pose(ball=ball)
    return ball[0], ball[1], ball[2], frame


def random_ball():
    if _is_running():
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    rng = np.random.default_rng()
    x = float(rng.uniform(-0.35, 0.35))
    y = float(rng.uniform(0.20, 0.70))
    z = float(rng.uniform(0.08, 0.50))
    ball = [x, y, z]
    frame = _render_pose(ball=ball)
    return x, y, z, frame


def run_demo(bx, by, bz, azimuth, elevation, distance, running):
    scene = _get_scene()
    _set_running(True)
    with scene.lock:
        scene.apply_camera(azimuth, elevation, distance)
        model, data, env = scene.model, scene.data, scene.env

        env.set_finger_target(config.FINGER_OPEN)
        env.reset_arm()
        env.reset_ball_geom_collisions()
        for ball_name in env.get_ball_names():
            env.release_ball(ball_name)
        env.set_ball_pos([float(bx), float(by), float(bz)], "ball_primary")

        target_names = list(env.get_ball_names())
        current_target_name = target_names[0]
        target_pos = env.get_ball_pos(current_target_name)

        state = STATE_REACHING
        step_count = 0
        lift_traj = None
        lift_idx = 0
        swing_step = 0
        swing_j1_center = 0.0
        jt_target = None
        jt_traj = None
        jt_step = 0
        use_jt = False
        jt_failed = False
        frame_counter = 0
        t0 = time.time()

        max_steps = getattr(config, 'MAX_STEPS', 3000)
        lift_steps = getattr(config, 'LIFT_STEPS', 120)
        swing_steps = getattr(config, 'SWING_STEPS', 120)
        traj_steps = getattr(config, 'TRAJECTORY_STEPS', 250)
        grab_dist = getattr(config, 'GRAB_DIST', 0.13)

        while state != STATE_DONE and step_count < max_steps:
            if state == STATE_REACHING:
                step_count += 1
                ee = env.get_ee_pos()
                curr_dist = float(np.linalg.norm(np.array(ee) - target_pos))

                if not use_jt and jt_target is None and not jt_failed:
                    current_j = np.array(env.get_joint_angles())
                    joint_config, ik_ok = solve_ik(
                        model, data, np.array(target_pos, dtype=float),
                        env.get_ee_body_id(),
                        env.get_joint_qposadrs(),
                        env.get_joint_dofadrs(),
                        q_init=None, damping=config.IK_DAMPING,
                        max_iter=config.IK_MAX_ITER, tol=config.IK_TOLERANCE,
                    )
                    if ik_ok:
                        jt_target = np.array([
                            joint_config[adr] for adr in env.get_joint_qposadrs()
                        ])
                        jt_traj = np.linspace(current_j, jt_target, traj_steps)
                        jt_step = 0
                        use_jt = True
                        env.set_joints_instant(current_j)
                    else:
                        use_jt = False
                        jt_failed = True

                if use_jt and jt_traj is not None and jt_step < len(jt_traj):
                    env.set_joints_instant(jt_traj[jt_step])
                    env.step()
                    jt_step += 1
                else:
                    speed = 0.012 if curr_dist > 0.2 else 0.006
                    ik_reach(
                        env, target_pos, speed_scale=speed,
                        obstacle_pos=env.get_obstacle_positions(),
                        avoid=bool(env.get_obstacle_positions()),
                    )
                    env.step()

                ee = env.get_ee_pos()
                curr_dist = float(np.linalg.norm(np.array(ee) - target_pos))

                if curr_dist < grab_dist:
                    env.set_finger_target(config.FINGER_CLOSED)
                    env.grab_ball(current_target_name)
                    state = STATE_GRABBED
                    ee_now = env.get_ee_pos()
                    lift_target = [ee_now[0] * 0.3, ee_now[1] * 0.3, 0.48]
                    lift_traj = minimum_jerk_trajectory(
                        np.array(ee_now), np.array(lift_target), lift_steps
                    )
                    lift_idx = 0

            elif state == STATE_GRABBED:
                env.step()
                if lift_traj is not None and lift_idx < len(lift_traj):
                    wp = np.array(lift_traj[lift_idx], dtype=float)
                    lift_idx += 1
                    ik_reach(env, wp,
                             obstacle_pos=env.get_obstacle_positions(),
                             avoid=bool(env.get_obstacle_positions()),
                             grabbed_ball=current_target_name)
                    env.set_finger_target(config.FINGER_CLOSED)
                    env.step()
                    state = STATE_LIFTING

            elif state == STATE_LIFTING:
                env.set_finger_target(config.FINGER_CLOSED)
                if lift_idx < len(lift_traj):
                    wp = np.array(lift_traj[lift_idx], dtype=float)
                    lift_idx += 1
                    ik_reach(env, wp,
                             obstacle_pos=env.get_obstacle_positions(),
                             avoid=bool(env.get_obstacle_positions()),
                             grabbed_ball=current_target_name)
                    env.step()
                else:
                    swing_step = 0
                    angles = env.get_joint_angles()
                    swing_j1_center = angles[0]
                    state = STATE_SWINGING

            elif state == STATE_SWINGING:
                swing_step += 1
                angle = swing_j1_center + 0.25 * np.sin(swing_step * 0.03)
                env.set_joint_target(0, float(angle))
                env.set_finger_target(config.FINGER_CLOSED)
                env.step()
                env.sync_grabbed_ball(current_target_name)
                if swing_step >= swing_steps:
                    env.release_ball(current_target_name)
                    env.set_finger_target(config.FINGER_OPEN)
                    state = STATE_DONE

            elif state == STATE_IDLE:
                env.step()

            env.freeze_balls(
                exclude=current_target_name if state in (STATE_GRABBED, STATE_LIFTING, STATE_SWINGING) else None
            )

            frame_counter += 1
            if frame_counter % 2 == 0:
                frame = scene.render()
                _remember_frame(frame)
                msg = f"Step {step_count}/{max_steps} — {state}"
                yield [frame, f"{msg}  ({time.time() - t0:.0f}s)", False, 0, True]

        frame = scene.render()
        _remember_frame(frame)
        elapsed = time.time() - t0
        with _FRAME_LOCK:
            n_frames = len(_LAST_FRAMES)
        if state == STATE_DONE:
            note = f"Done — {step_count} steps, state={state} ({elapsed:.0f}s)."
        else:
            note = f"Stopped at max steps — {step_count} steps, state={state} ({elapsed:.0f}s)."
        _set_running(False)
        yield [frame, note, True, 0, False]


def view_frame(i):
    if i is None:
        i = 0
    with _FRAME_LOCK:
        n_frames = len(_LAST_FRAMES)
    if n_frames == 0:
        return _render_pose(), "No run yet — press Run reach & grab first.", False
    idx = max(0, min(int(i), n_frames - 1))
    with _FRAME_LOCK:
        img = _LAST_FRAMES[idx].copy()
    return img, f"Frame {idx + 1}/{n_frames} (scrubbed)", False


def autoplay(playing, idx):
    with _FRAME_LOCK:
        n_frames = len(_LAST_FRAMES)
    if not playing:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    if n_frames == 0:
        return gr.skip(), "No run yet — press Run reach & grab first.", False, 0
    i = idx % n_frames
    with _FRAME_LOCK:
        img = _LAST_FRAMES[i].copy()
    return img, f"Replay — frame {i + 1}/{n_frames}", True, (i + 1) % n_frames


def play_pause(playing):
    return not playing


def stop_run(running):
    _set_running(False)
    return False, False, "Stopped — replay paused."


def toggle_controls(running):
    val = {"interactive": not running}
    return [val] * 15


def download_gif():
    import imageio
    with _FRAME_LOCK:
        frames = list(_LAST_FRAMES)
    if not frames:
        return None, "No run yet — run one first, then download."
    out_path = os.path.join(
        tempfile.gettempdir(), f"stardance_{int(time.time())}.gif"
    )
    imageio.mimsave(out_path, frames, fps=15)
    return out_path, f"GIF ready ({len(frames)} frames)"


def save_video():
    import imageio
    with _FRAME_LOCK:
        frames = list(_LAST_FRAMES)
    if not frames:
        return None, "No run yet — run one first, then download."
    out_path = os.path.join(
        tempfile.gettempdir(), f"stardance_{int(time.time())}.mp4"
    )
    imageio.mimsave(out_path, frames, fps=20, codec="libx264")
    return out_path, f"MP4 ready ({len(frames)} frames)"


with gr.Blocks(title="Stardance — NASA On-Orbit Servicing Simulator") as demo:
    gr.HTML("""
    <div style="text-align:center; padding:1rem">
        <h1>🛰️ Stardance</h1>
        <p style="font-size:1.1rem; color:#666">
            NASA On-Orbit Servicing Simulator — MuJoCo Robotic Arm
        </p>
        <p style="font-size:0.85rem; color:#aaa; max-width:600px; margin:0.3rem auto 0; line-height:1.4">
            🎮 <b>Interactive demo (v2).</b> Drop a ball anywhere, twiddle the joints, spin the camera,
            then hit <b>run</b> and watch the arm reach &amp; grab it freshly simulated — no replayed GIF.
            It renders on a small Hack Club box, so the frame rate is kinda humble. 😄
            Grabbing full control? Git clone the repo and check the README.
        </p>
    </div>
    """)

    gr.Markdown("""**How to use** (every slider works with arrow keys too, no mouse needed):
1. Place the ball — X/Y/Z sliders, a preset, or 🎲 random.
2. Optional: pose the arm with the joint sliders, or spin the camera.
3. Hit **Run reach & grab** and watch it go live.
4. When it finishes, the replay **auto-plays like a video** — use ⏯/Pause, or drag **Scrub** for any frame, then download GIF or MP4.""")

    with gr.Row():
        with gr.Column(scale=2):
            sim = gr.Image(
                label="Live Simulation", type="numpy", interactive=False,
                height=440, show_label=False,
            )
        with gr.Column(scale=1):
            status = gr.Textbox(
                label="Status",
                value="Ready — place the ball (sliders/preset/random) and press Run, or pose the arm manually.",
                lines=3,
            )
            with gr.Row():
                run_btn = gr.Button("▶ Run reach & grab", variant="primary")
                stop_btn = gr.Button("Stop")
            with gr.Row():
                play_btn = gr.Button("⏯ Play / Pause replay", variant="secondary")
                export_gif_btn = gr.Button("💾 Download GIF")
                export_mp4_btn = gr.Button("⬇ MP4")
            export = gr.File(label="Exported clip (fills after clicking GIF or MP4)")
            play_state = gr.State(False)
            idx_state = gr.State(0)
            running = gr.State(False)
            scrub = gr.Slider(
                0, 1500, value=0, step=1, label="Scrub — step through the last run",
            )
            preset = gr.Dropdown(
                choices=list(BALL_PRESETS.keys()), value="Center", label="Ball preset"
            )
            random_btn = gr.Button("🎲 Random ball")
            bx = gr.Slider(-0.5, 0.5, value=0.0, step=0.01, label="Ball X (ship-ward →)")
            by = gr.Slider(0.10, 0.8, value=0.3, step=0.01, label="Ball Y (out from base)")
            bz = gr.Slider(0.05, 0.7, value=0.10, step=0.01, label="Ball Z (height)")
            with gr.Accordion("Manual arm pose (live)", open=False):
                j1 = gr.Slider(*JOINT_LIMITS[0], value=0.0, step=0.01, label="Joint 1")
                j2 = gr.Slider(*JOINT_LIMITS[1], value=0.0, step=0.01, label="Joint 2")
                j3 = gr.Slider(*JOINT_LIMITS[2], value=0.0, step=0.01, label="Joint 3")
                j4 = gr.Slider(*JOINT_LIMITS[3], value=0.0, step=0.01, label="Joint 4")
                j5 = gr.Slider(*JOINT_LIMITS[4], value=0.0, step=0.01, label="Joint 5")
                j6 = gr.Slider(*JOINT_LIMITS[5], value=0.0, step=0.01, label="Joint 6")
                finger = gr.Slider(
                    config.FINGER_CLOSED, config.FINGER_OPEN, value=config.FINGER_OPEN,
                    step=0.0005, label="Finger grip (0 = closed)",
                )
            with gr.Accordion("Camera", open=False):
                az_slider = gr.Slider(-180, 180, value=DEFAULT_CAM["azimuth"], step=1, label="Azimuth")
                el_slider = gr.Slider(-90, 90, value=DEFAULT_CAM["elevation"], step=1, label="Elevation")
                dist_slider = gr.Slider(1.0, 8.0, value=DEFAULT_CAM["distance"], step=0.1, label="Distance")

    pose_inputs = [j1, j2, j3, j4, j5, j6, finger, bx, by, bz, az_slider, el_slider, dist_slider]
    run_inputs = [bx, by, bz, az_slider, el_slider, dist_slider]
    locked_controls = [
        bx, by, bz, preset, random_btn,
        j1, j2, j3, j4, j5, j6,
        finger, az_slider, el_slider, dist_slider,
    ]

    demo.load(fn=initial_frame, outputs=sim)
    run_ev = run_btn.click(
        fn=run_demo, inputs=[*run_inputs, running],
        outputs=[sim, status, play_state, idx_state, running],
    )
    stop_btn.click(
        fn=stop_run, inputs=running,
        outputs=[running, play_state, status],
        cancels=[run_ev],
    )

    for ctrl in [j1, j2, j3, j4, j5, j6]:
        ctrl.input(fn=pose_render, inputs=pose_inputs, outputs=sim)
    finger.input(fn=pose_render, inputs=pose_inputs, outputs=sim)
    for ctrl in [bx, by, bz, az_slider, el_slider, dist_slider]:
        ctrl.change(fn=pose_render, inputs=pose_inputs, outputs=sim)

    running.change(fn=toggle_controls, inputs=running, outputs=locked_controls)
    preset.change(fn=apply_preset, inputs=preset, outputs=[bx, by, bz, sim])
    random_btn.click(fn=random_ball, outputs=[bx, by, bz, sim])
    scrub.change(fn=view_frame, inputs=scrub, outputs=[sim, status, play_state])
    export_gif_btn.click(fn=download_gif, outputs=[export, status])
    export_mp4_btn.click(fn=save_video, outputs=[export, status])
    play_btn.click(fn=play_pause, inputs=play_state, outputs=play_state)
    replay_timer = gr.Timer(0.06)
    replay_timer.tick(
        fn=autoplay,
        inputs=[play_state, idx_state],
        outputs=[sim, status, play_state, idx_state],
    )


import gradio.http_server
class _FixScheme:
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope["scheme"] = "https"
            hdrs = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"x-forwarded-proto"]
            hdrs.append((b"x-forwarded-proto", b"https"))
            scope["headers"] = hdrs
        await self.app(scope, receive, send)


_orig_start = gradio.http_server.start_server
def _patched_start(app, server_name=None, server_port=None,
                    ssl_keyfile=None, ssl_certfile=None,
                    ssl_keyfile_password=None):
    return _orig_start(_FixScheme(app), server_name=server_name,
                       server_port=server_port, ssl_keyfile=ssl_keyfile,
                       ssl_certfile=ssl_certfile,
                       ssl_keyfile_password=ssl_keyfile_password)
gradio.http_server.start_server = _patched_start

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())