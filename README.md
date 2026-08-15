# StardustArm

yo this is my project for the nasa on-orbit servicing challenge thing. basically a robot arm in space that learns to grab floating balls by itself. no hand-coded bs, just pure reinforcement learning. i still need to add the feuture of letting the user to place the ball..... but u can code it in if u git clone it......it's only v1 rn..more updates will be coming.......stay tuned

## what it do

there's a 6-DOF mephi arm bolted onto a spaceship (or free-floating if you want). little target spheres pop up in front of it and the arm has to:

1. **reach** — get gripper within 15cm of the ball
2. **grab** — clamp down on it
3. **repeat** — new ball, new position

the AI learns entirely from trial and error using HER (hindsight experience replay). basically when it misses, instead of recording a failure it says "what if i *meant* to reach that spot instead?" — turns every miss into a win. magic.

## how it works

### stack

| part | lib |
|------|-----|
| physics | mujoco 3.x |
| rl algorithm | stable-baselines3 SAC |
| goal replay | HER |
| neural net | pytorch 2.x |
| viewer | mujoco's glfw thing |

### files

```
train_sac.py         →  trains the model, pops open a viewer
spaceship_env.py     →  the gym environment (physics + reward)
watch.py             →  replay a trained model
config.py            →  constants n stuff
assets/robots/mephi_arm/ →  urdf, stl meshes, scene xml
```

### the env (spaceship_env.py)

- **observation**: 23-dim (joint positions/velocities, gripper pos, finger state, prev action) + 3-dim goal (target pos) + 3-dim achieved goal (where gripper actually is)
- **action**: 7 continuous values [-1, 1] — 6 arm joints + finger
- **reward**: negative distance to target (-||gripper - target||)
- **done**: when gripper is within 15cm of target OR 5000 steps
- **reset**: random target position in front of arm, random joint positions

### training (train_sac.py)

SAC with auto entropy tuning + HER replay buffer with `future` goal sampling (4 per transition). checkpoints saved every 50k steps to `training_runs/<name>/`. viewer pops up so you can watch it learn in real-time which is honestly pretty sick.

### viewer (watch.py)

lists all training runs, pick one, pick a checkpoint, it loops episodes forever. esc to exit.

## setup

```bash
git clone https://github.com/GurnoorLog/StardustArm.git
cd StardustArm
pip install mujoco gymnasium stable-baselines3 sb3-contrib tensorboard glfw PyOpenGL
```

## training

```bash
python train_sac.py --run my_cool_run --timesteps 500000
```

**viewer controls:**
- left drag → orbit
- right drag / scroll → zoom
- middle drag → pan
- esc → exit

## watch a trained model

```bash
python watch.py
```

## tensorboard

```bash
tensorboard --logdir training_runs
```

go to `http://localhost:6006` to see reward curves n stuff.

## why this way

**HER not shaped rewards?** in 6dof continuous control random flailing almost never grabs anything. HER relabels failures so the agent learns from *everything*, not just the one-in-a-million success.

**SAC not PPO?** SAC's more sample-efficient for continuous control and handles the explore-vs-exploit balance automatically.

**delta-position control?** moving in small increments (±5% joint range per step) creates random-walk exploration that covers the whole joint space. absolute positioning would need the model to already know where to go.

**one finger not parallel jaws?** the mephi model has a single prismatic finger that clamps balls against the gripper base. when it retracts to 0mm, ball's pinched between the finger and base.

## results

~30k steps → ~50% success rate. 100k-200k steps → reliably grabs every time. the old hand-coded inverse kinematics worked but needed exact target info. with RL the arm develops its own reaching strategy — no IK needed.

## built with

- [mujoco](https://mujoco.org/) by google deepmind
- [stable-baselines3](https://stable-baselines3.readthedocs.io/) by dlr-rm
- [gymnasium](https://gymnasium.farama.org/) by farama foundation
- pytorch, glfw, numpy
