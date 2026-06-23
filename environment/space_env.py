import pybullet as p
import time


_connected = False
_gui_mode = False


def connect():
    global _connected, _gui_mode
    try:
        p.connect(p.GUI, options="--width=800 --height=600")
        _gui_mode = True
        _connected = True
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        for flag in [1, 2, 8]:
            try:
                p.configureDebugVisualizer(flag, 0)
            except Exception:
                pass
        p.setPhysicsEngineParameter(
            fixedTimeStep=1/240.0,
            numSolverIterations=50,
            enableConeFriction=0,
        )
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(
            cameraDistance=3.5, cameraYaw=45,
            cameraPitch=-30, cameraTargetPosition=[0.0, 0.0, 0.3],
        )
    except Exception as e:
        print(f"[ENV] GUI failed ({e}) — falling back to DIRECT")
        for _ in range(3):
            try:
                p.disconnect()
            except Exception:
                pass
            try:
                p.connect(p.DIRECT)
                break
            except Exception:
                continue
        _gui_mode = False
        _connected = True
        p.setPhysicsEngineParameter(
            fixedTimeStep=1/240.0,
            numSolverIterations=50,
            enableConeFriction=0,
        )
        p.setGravity(0, 0, 0)


def finalize_rendering():
    if _gui_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        try:
            p.stepSimulation()
        except Exception:
            pass
        time.sleep(0.3)


def refresh_gui():
    if _gui_mode:
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        except Exception:
            pass


def is_connected():
    try:
        p.getNumBodies()
        return True
    except Exception:
        return False


def is_gui():
    return _gui_mode


def step_physics():
    try:
        p.stepSimulation()
        refresh_gui()
    except Exception:
        pass
