import pybullet as p
import numpy as np
import math
from config import GOAL_BALL_RADIUS


class ControlPanel:
    def __init__(self):
        self.items = {}
        self._button_latches = {}
        self.target_marker_id = None
        self.glow_marker_id = None
        self._target_pos = [0.0, 0.4, 0.06]
        self._mouse_grabbed = False
        self._is_grabbed = False
        self._grab_cam = None
        self._grab_target = None
        self._create_all()

    def _add_slider(self, name, range_min, range_max, default):
        uid = p.addUserDebugParameter(name, range_min, range_max, default)
        self.items[name] = uid
        return uid

    def _add_button(self, name):
        uid = p.addUserDebugParameter(name, 0, 1, 0)
        self.items[name] = uid
        self._button_latches[name] = False
        return uid

    def _add_header(self, name):
        uid = p.addUserDebugParameter(name, 0, 1, 0)
        self.items[name] = uid
        return uid

    def _create_all(self):
        self._add_header("=== DEBRIS ===")
        self._add_slider("Debris Count", 0, 20, 0)
        self._add_slider("Debris Velocity", 0.0, 2.0, 0.2)
        self._add_slider("Debris Spin", 0.0, 1.0, 0.1)
        self._add_button("Apply All")
        self._add_button("Add One")
        self._add_button("Clear All")

        self._add_header("=== CONTROL ===")
        self._add_slider("Sim Speed", 0.1, 5.0, 1.0)
        self._add_slider("Gravity", -20.0, 0.0, -9.81)
        self._add_button("Pause AI")

        self._add_header("=== GOAL BALL ===")
        self._add_slider("Ball X", -5, 5, 0.0)
        self._add_slider("Ball Y", -5, 5, 0.4)
        self._add_slider("Ball Z", -5, 5, 0.06)
        self._add_slider("Ball Size", 0.02, 0.3, GOAL_BALL_RADIUS)
        self._add_button("Teleport Ball")
        self._add_button("Reset Ball")

        self._add_header("=== ARM BEHAVIOUR ===")
        self._add_slider("Step Size", 0.01, 0.2, 0.05)
        self._add_slider("Success Threshold", 0.01, 0.2, 0.05)

        self._add_header("=== ACTION ===")
        self._add_button("  START  ")

    def read_all_controls(self):
        result = {}
        for name, uid in self.items.items():
            try:
                result[name] = p.readUserDebugParameter(uid)
            except Exception:
                pass

        result["start_pressed"] = self._check_button("  START  ")
        result["apply_debris_pressed"] = self._check_button("Apply All")
        result["add_debris_pressed"] = self._check_button("Add One")
        result["clear_debris_pressed"] = self._check_button("Clear All")
        result["pause_ai_pressed"] = self._check_button("Pause AI")

        result["debris_count"] = int(round(result.get("Debris Count", 0)))
        result["sim_speed"] = result.get("Sim Speed", 1.0)

        result["teleport_ball_pressed"] = self._check_button("Teleport Ball")
        result["reset_ball_pressed"] = self._check_button("Reset Ball")
        result["ball_size"] = result.get("Ball Size", GOAL_BALL_RADIUS)
        result["step_size"] = result.get("Step Size", 0.05)
        result["success_threshold"] = result.get("Success Threshold", 0.05)
        result["ball_xyz"] = [
            result.get("Ball X", 0.0),
            result.get("Ball Y", 0.4),
            result.get("Ball Z", 0.15),
        ]
        return result

    def _check_button(self, name):
        try:
            val = p.readUserDebugParameter(self.items[name])
        except Exception:
            return False
        latched = self._button_latches.get(name, False)
        if val > 0.5 and not latched:
            self._button_latches[name] = True
            return True
        if val < 0.5:
            self._button_latches[name] = False
        return False

    def get_target_position(self):
        return list(self._target_pos)

    def set_target_position(self, xyz):
        self._target_pos = list(xyz)

    def move_target_to(self, xyz):
        self._target_pos = list(xyz)

    def set_grabbed(self, grabbed):
        self._is_grabbed = grabbed

    def is_grabbed(self):
        return self._is_grabbed

    def _camera_delta_move(self, cam_info):
        dyaw = math.radians(cam_info[8] - self._grab_cam['yaw'])
        dpitch = math.radians(cam_info[9] - self._grab_cam['pitch'])

        g_yaw_r = math.radians(self._grab_cam['yaw'])
        g_pitch_r = math.radians(self._grab_cam['pitch'])
        g_dist = self._grab_cam['dist']
        ct = self._grab_cam['target']

        cam_pos = np.array([
            ct[0] + g_dist * math.cos(g_pitch_r) * math.sin(g_yaw_r),
            ct[1] + g_dist * math.cos(g_pitch_r) * math.cos(g_yaw_r),
            ct[2] + g_dist * math.sin(g_pitch_r),
        ])
        ball_pos = np.array(self._grab_target)
        depth = max(float(np.linalg.norm(ball_pos - cam_pos)), 0.1)

        view_dir = np.array([
            -math.cos(g_pitch_r) * math.sin(g_yaw_r),
            -math.cos(g_pitch_r) * math.cos(g_yaw_r),
            -math.sin(g_pitch_r),
        ])
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(view_dir, world_up)
        rn = np.linalg.norm(right)
        if rn > 1e-8:
            right = right / rn
        else:
            right = np.array([1.0, 0.0, 0.0])
        up = np.cross(right, view_dir)

        sensitivity = depth * 1.0
        delta = -dyaw * sensitivity * right + dpitch * sensitivity * up

        new_pos = np.array(self._grab_target) + delta
        self.move_target_to([float(new_pos[0]), float(new_pos[1]), float(new_pos[2])])

    def handle_mouse(self):
        cam_info = p.getDebugVisualizerCamera()
        width = float(cam_info[0])
        height = float(cam_info[1])
        view_mat = cam_info[2]
        proj_mat = cam_info[3]

        yaw_r = math.radians(cam_info[8])
        pitch_r = math.radians(cam_info[9])
        dist = cam_info[10]
        ct = cam_info[11]
        cam_pos = np.array([
            ct[0] + dist * math.cos(pitch_r) * math.sin(yaw_r),
            ct[1] + dist * math.cos(pitch_r) * math.cos(yaw_r),
            ct[2] + dist * math.sin(pitch_r),
        ])

        vp = np.array(proj_mat, dtype=float).reshape(4, 4).T @ np.array(view_mat, dtype=float).reshape(4, 4).T
        try:
            vp_inv = np.linalg.inv(vp)
        except np.linalg.LinAlgError:
            return

        def screen_to_ray(px, py):
            ndc_x = 2.0 * px / width - 1.0
            ndc_y = 1.0 - 2.0 * py / height
            far_h = vp_inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
            far_pt = (far_h / far_h[3])[:3]
            dir_v = far_pt - cam_pos
            n = np.linalg.norm(dir_v)
            if n < 1e-8:
                return cam_pos, cam_pos + np.array([0, 0, -1])
            dir_v = dir_v / n
            return cam_pos, cam_pos + dir_v * 15.0

        def ray_plane_intersect(r_from, r_to, plane_z):
            d = r_to - r_from
            dn = np.linalg.norm(d)
            if dn < 1e-8 or abs(d[2]) < 1e-8:
                return None
            d = d / dn
            t = (plane_z - r_from[2]) / d[2]
            if t < 0:
                return None
            return np.array(r_from + t * d)

        try:
            events = p.getMouseEvents()
        except Exception:
            events = []

        for ev in events:
            ev_type = ev[0]
            mx, my = float(ev[1]), float(ev[2])
            btn = ev[3]
            state = ev[4]

            if btn != 0:
                continue

            if state == 1:
                r_from, r_to = screen_to_ray(mx, my)
                hit = ray_plane_intersect(r_from, r_to, self._target_pos[2])
                if hit is not None:
                    dx = hit[0] - self._target_pos[0]
                    dy = hit[1] - self._target_pos[1]
                    if np.sqrt(dx * dx + dy * dy) < 1.0:
                        self._mouse_grabbed = True
                        self._grab_cam = {
                            'yaw': cam_info[8],
                            'pitch': cam_info[9],
                            'dist': cam_info[10],
                            'target': list(cam_info[11]),
                        }
                        self._grab_target = list(self._target_pos)
                        print(f"[MOUSE] Grab at {self._grab_target}")
            elif state == 0 and self._mouse_grabbed:
                self._mouse_grabbed = False
                self._grab_cam = None
                self._grab_target = None
                print(f"[MOUSE] Release at {self._target_pos}")

        if self._mouse_grabbed and self._grab_cam is not None:
            self._camera_delta_move(cam_info)

    def update_target_marker(self, xyz):
        pass
