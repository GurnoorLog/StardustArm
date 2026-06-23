import os
import sys
import tempfile
import pybullet as p
import numpy as np
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from config import URDF_PATH, URDF_FALLBACK


LINK_LENGTHS = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
LINK_RADIUS = 0.03
JOINT_LIMITS_RAD = 1.57


def build_arm(mount_position):
    if os.path.exists(URDF_PATH):
        try:
            arm_id = load_urdf_arm(mount_position)
            p.stepSimulation()
            p.getNumJoints(arm_id)
            print(f"[ARM] Loaded from URDF: {URDF_PATH}")
            return arm_id
        except Exception as e:
            print(f"[ARM] URDF load failed ({e}) — falling back to programmatic arm")
    if URDF_FALLBACK:
        arm_id = build_programmatic_arm(mount_position)
        print("[ARM] Using programmatic arm (fallback)")
        return arm_id
    raise FileNotFoundError(f"URDF not found at {URDF_PATH} and fallback disabled")


def load_urdf_arm(mount_position):
    arm_id = p.loadURDF(
        URDF_PATH,
        basePosition=list(mount_position),
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=1,
        globalScaling=1.0,
    )
    _setup_arm_joints(arm_id)
    num_joints = p.getNumJoints(arm_id)
    revolute_count = 0
    for j in range(num_joints):
        info = p.getJointInfo(arm_id, j)
        jtype = info[2]
        jname = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
        if jtype == p.JOINT_REVOLUTE:
            revolute_count += 1
        type_str = "revolute" if jtype == p.JOINT_REVOLUTE else "prismatic" if jtype == p.JOINT_PRISMATIC else "fixed"
        print(f"[ARM] Joint {j}: {jname} ({type_str})")
    print(f"[ARM] Loaded {revolute_count} revolute joints from URDF")
    return arm_id


def build_programmatic_arm(mount_position):
    urdf_str = _generate_fallback_urdf()
    tmp_path = os.path.join(tempfile.gettempdir(), "stardance_fallback_arm.urdf")
    with open(tmp_path, "w") as f:
        f.write(urdf_str)
    arm_id = p.loadURDF(
        tmp_path,
        basePosition=list(mount_position),
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=1,
    )
    _setup_arm_joints(arm_id)
    return arm_id


def _setup_arm_joints(arm_id):
    num_joints = p.getNumJoints(arm_id)
    for j in range(num_joints):
        joint_info = p.getJointInfo(arm_id, j)
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            p.resetJointState(arm_id, j, 0.0)
            p.setJointMotorControl2(
                arm_id, j,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=8.0,
                maxVelocity=50.0,
            )


def verify_urdf():
    import pybullet as p
    import pybullet_data
    tmp_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        arm_id = p.loadURDF(
            URDF_PATH,
            basePosition=[0, 0, 0],
            useFixedBase=1,
            globalScaling=1.0,
        )
        num_joints = p.getNumJoints(arm_id)
        print(f"[VERIFY] URDF loaded OK — {num_joints} joints found")
        revolute_count = 0
        for j in range(num_joints):
            info = p.getJointInfo(arm_id, j)
            jname = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
            jtype = info[2]
            type_str = "revolute" if jtype == p.JOINT_REVOLUTE else "fixed"
            print(f"  Joint {j}: {jname} type={jtype} ({type_str})")
            if jtype == p.JOINT_REVOLUTE:
                revolute_count += 1
        print(f"[VERIFY] Revolute joints: {revolute_count}")
        p.disconnect()
        return True
    except Exception as e:
        print(f"[VERIFY] URDF load failed: {e}")
        p.disconnect()
        return False


if __name__ == "__main__":
    verify_urdf()


def _generate_fallback_urdf():
    link_lengths = LINK_LENGTHS
    link_radii = [LINK_RADIUS] * 6
    link_masses = [0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    link_colors = [
        [0.85, 0.2, 0.2, 1.0],
        [0.2, 0.85, 0.2, 1.0],
        [0.2, 0.2, 0.85, 1.0],
        [0.85, 0.85, 0.1, 1.0],
        [0.85, 0.1, 0.85, 1.0],
        [0.1, 0.85, 0.85, 1.0],
    ]
    joint_axes = [
        "0 0 1",
        "0 1 0",
        "0 1 0",
        "0 0 1",
        "0 1 0",
        "0 0 1",
    ]
    parts = []
    parts.append('<?xml version="1.0"?>')
    parts.append('<robot name="stardance_arm">')
    parts.append("""  <link name="base_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry><box size="0.16 0.16 0.08"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <material name="base_mat"><color rgba="0.3 0.3 0.4 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.16 0.16 0.08"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
  </link>""")
    for i in range(6):
        length = link_lengths[i]
        radius = link_radii[i]
        mass = link_masses[i]
        r, g, b, a = link_colors[i]
        inertia_val = 0.001
        parts.append(f"""  <link name="link_{i+1}">
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="{inertia_val}" ixy="0" ixz="0" iyy="{inertia_val}" iyz="0" izz="{inertia_val}"/>
    </inertial>
    <visual>
      <geometry><cylinder length="{length}" radius="{radius}"/></geometry>
      <origin xyz="0 0 {length/2}" rpy="0 0 0"/>
      <material name="link{i+1}_mat"><color rgba="{r} {g} {b} {a}"/></material>
    </visual>
    <collision>
      <geometry><cylinder length="{length}" radius="{radius}"/></geometry>
      <origin xyz="0 0 {length/2}" rpy="0 0 0"/>
    </collision>
  </link>""")
    z_offset = 0.04
    for i in range(6):
        length = link_lengths[i]
        parent = "base_link" if i == 0 else f"link_{i}"
        child = f"link_{i+1}"
        parts.append(f"""  <joint name="joint_{i+1}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="0 0 {z_offset}" rpy="0 0 0"/>
    <axis xyz="{joint_axes[i]}"/>
    <limit lower="-{JOINT_LIMITS_RAD}" upper="{JOINT_LIMITS_RAD}" effort="1.0" velocity="1.0"/>
  </joint>""")
        z_offset = length if i == 0 else length
    parts.append('</robot>')
    return "\n".join(parts)
