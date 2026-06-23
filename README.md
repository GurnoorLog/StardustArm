# StardustArm 🚀

**Reinforcement learning for a space robotic arm using MuJoCo + Stable-Baselines3 (SAC + HER).**

A 6-DOF robotic arm mounted on a spaceship learns to reach and grab floating target spheres in zero-gravity through trial and error — no hand-coded IK, no trajectory planning, just pure RL.

---

## What This Does

A MuJoCo-simulated MEPhi arm (6 revolute joints + 1 prismatic gripper) is attached to a static or free-floating spaceship base. Floating target spheres appear in the arm's workspace. The arm must:

1. **Reach** — bring its gripper within 15cm of the target
2. **Grab** — close the finger and activate a weld constraint
3. **Repeat** — reset and try a new target position

The agent learns entirely from experience using **Hindsight Experience Replay (HER)**, which relabels failed trajectories as if the reached position *was* the goal, turning "misses" into learning opportunities.

---

## How It Was Built

### Tech Stack

| Component | Library |
|-----------|---------|
| Physics simulation | [MuJoCo](https://mujoco.org/) 3.x |
| RL algorithm | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) SAC |
| Goal-conditioned RL | HER (Hindsight Experience Replay) |
| Neural network | PyTorch 2.x (CUDA) |
| Viewer | MuJoCo native GLFW renderer |

### Architecture

```
train_sac.py          →  Starts training, opens viewer window
spaceship_env.py      →  Gymnasium environment (MuJoCo physics + reward)
watch.py              →  Standalone viewer to replay trained checkpoints
config.py             →  Constants (joint limits, grab dist, finger range)
assets/robots/mephi_arm/  →  URDF, STL meshes, MuJoCo scene XML
```

### The Environment (`spaceship_env.py`)

- **Observation space** (Dict): `observation` (23-dim: joint pos/vel, EE pos, finger state, prev control) + `desired_goal` (3-dim: target position) + `achieved_goal` (3-dim: gripper position)
- **Action space**: 7 continuous values [-1, 1] — delta-position for 6 arm joints + finger
- **Reward**: negative distance from gripper to target (`-||ee_pos - target_pos||`)
- **Termination**: when gripper is within 15cm of target (grab) OR 5000 steps elapsed
- **Reset**: random target position (hemisphere in front of arm), random initial joint positions

### Training (`train_sac.py`)

- SAC (Soft Actor-Critic) with automatic entropy tuning
- HER replay buffer with `future` goal selection strategy and 4 sampled goals per transition
- MultiInputPolicy (handles Dict observations)
- Training runs saved to `training_runs/<run_name>/` with checkpoints every 50K steps
- Viewer window opens automatically during training

### The Viewer (`watch.py`)

- Lists all available training runs
- Lets you pick a run and checkpoint to replay
- Plays episodes in a loop with deterministic actions
- ESC to exit

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, but recommended)
- Git

### Setup

```bash
# Clone
git clone https://github.com/GurnoorLog/StardustArm.git
cd StardustArm

# Install dependencies
pip install mujoco gymnasium stable-baselines3 sb3-contrib tensorboard glfw PyOpenGL
```

### Train a New Model

```bash
python C:\path\to\train_sac.py --run my_first_run --timesteps 500000
```

Opens a viewer window showing the arm learning in real-time.

**Camera controls in the viewer:**
- Left drag → orbit
- Right drag / scroll → zoom
- Middle drag → pan
- ESC → exit

### Watch a Trained Model

```bash
python watch.py
```

Select a run and checkpoint from the list, then watch episodes play automatically.

### View Training Metrics

```bash
tensorboard --logdir training_runs
```

Opens TensorBoard at `http://localhost:6006` showing reward curves, success rate, episode length, and more.

---

## Project Structure

```
├── spaceship_env.py         # Gymnasium RL environment (MuJoCo + reward)
├── train_sac.py             # SAC+HER training script
├── watch.py                 # Checkpoint viewer with run selection
├── config.py                # Shared constants
├── training_runs/           # Per-run folders with models, logs, configs
├── assets/robots/mephi_arm/ # Robot URDF, STL meshes, MuJoCo scene
├── agent/                   # Legacy hand-coded agents (being replaced)
├── environment/             # Legacy env components
├── arm/                     # Legacy arm kinematics
├── learning/                # Legacy learning modules
├── main.py                  # Legacy hand-coded state machine
└── ui/                      # Legacy UI components
```

---

## Key Design Decisions

**Why HER instead of shaped rewards?**  
In 6D continuous control, random exploration almost never discovers the "grab" event. HER relabels failed trajectories so the agent learns from every episode, not just the rare successes. This is the standard approach for robotic goal-reaching tasks.

**Why SAC instead of PPO?**  
SAC is more sample-efficient for continuous control and handles the exploration-exploitation trade-off automatically via entropy regularization.

**Why delta-position control instead of absolute position?**  
Delta-position actions (±5% joint range per step) create a random-walk exploration pattern that covers the joint space, unlike absolute position commands which would need precise targeting from the start.

**Why a single finger instead of parallel jaws?**  
The MEPhi arm model has a single prismatic finger that clamps objects against the gripper base. When the finger retracts to 0mm, the ball is pinched between the finger pad and the base.

---

## Customizing

### Change the Robot
Replace the URDF and STL files in `assets/robots/mephi_arm/` and update `JOINT_LIMITS` in `spaceship_env.py`.

### Change the Environment
- `GRAB_DIST` — how close the gripper must be to grab
- `MAX_STEPS` — max steps per episode
- `FINGER_OPEN` / `FINGER_CLOSED` — finger range
- `floating_base=True` — enable free-floating base with thrusters

### Change the Algorithm
Edit `train_sac.py`:
- `learning_rate`, `buffer_size`, `batch_size` in SAC constructor
- `n_sampled_goal` — how many HER relabelings per transition
- `goal_selection_strategy` — "future", "final", or "episode"
- Switch to PPO by replacing SAC with PPO (change `MultiInputPolicy` to `MlpPolicy` and remove HER)

---

## Results

After ~30K timesteps with SAC+HER, the agent achieves ~50% success rate. By 100K-200K steps, it reliably reaches and grabs the target in most episodes.

With the old hand-coded approach (IK + trajectory planning), the arm could reach but required precise target information. With RL, the arm learns to reach *without* explicit inverse kinematics — it develops its own reaching strategy through trial and error.

---

## Built With

- [MuJoCo](https://mujoco.org/) by Google DeepMind
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) by DLR-RM
- [Gymnasium](https://gymnasium.farama.org/) by Farama Foundation
- PyTorch, GLFW, NumPy
