import sys, os, glob, json, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.spaceship_env import SpaceshipArmEnv
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE, "training_runs")

class ViewerCallback(BaseCallback):
    def _on_step(self):
        try:
            env = self.training_env.envs[0]
            env.render()
            alive = env.is_running()
            if not alive:
                print("\nViewer closed. Stopping training.", flush=True)
            return alive
        except:
            return True

def create_run(run_name=None):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_name:
        run_name = f"sac_her_{ts}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    config = {
        "run_name": run_name,
        "created": ts,
        "algorithm": "SAC",
        "replay_buffer": "HerReplayBuffer",
        "learning_starts": 10000,
        "buffer_size": 200_000,
        "batch_size": 256,
        "n_sampled_goal": 4,
        "goal_selection": "future",
        "max_steps": 5000,
        "grab_dist": 0.15,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return run_dir, config

def make_env_fn(run_dir, rank):
    log_dir = os.path.join(run_dir, "logs")
    def _init():
        env = SpaceshipArmEnv(space_gravity=True, render_mode="human")
        return Monitor(env, filename=os.path.join(log_dir, f"mon_{rank}"))
    return _init

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    args = parser.parse_args()

    from stable_baselines3.common.vec_env import DummyVecEnv
    run_dir, config = create_run(args.run)
    print(f"Run: {config['run_name']}", flush=True)

    model_dir = os.path.join(run_dir, "models")
    log_dir = os.path.join(run_dir, "logs")
    env = DummyVecEnv([make_env_fn(run_dir, 0)])

    model = SAC(
        "MultiInputPolicy", env,
        learning_starts=10000, learning_rate=3e-4,
        buffer_size=200_000, batch_size=256,
        tau=0.005, gamma=0.99,
        ent_coef="auto_0.5", target_entropy=-7,
        verbose=1, tensorboard_log=log_dir, device="cuda",
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
    )

    checkpoint = CheckpointCallback(save_freq=50_000, save_path=model_dir, name_prefix="sac_arm")
    viewer = ViewerCallback()

    print("Viewer: LMB=orbit, RMB/scroll=zoom, MMB=pan, ESC=exit", flush=True)
    print(f"Saving to: {run_dir}", flush=True)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint, viewer],
        tb_log_name="sac_arm_run",
        log_interval=10,
    )

    model.save(os.path.join(model_dir, "sac_arm_final.zip"))
    env.close()
    print(f"Done! Run saved in: {run_dir}", flush=True)
