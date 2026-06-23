import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from spaceship_env import SpaceshipArmEnv

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(rank=0, floating_base=False):
    def _init():
        env = SpaceshipArmEnv(floating_base=floating_base, space_gravity=True)
        env = Monitor(env, filename=os.path.join(LOG_DIR, f"monitor_{rank}"))
        return env
    return _init


def test_policy(model_path, episodes=5):
    env = SpaceshipArmEnv(space_gravity=True, render_mode="human")
    model = PPO.load(model_path)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()
        print(f"Episode {ep + 1}: reward={total_reward:.1f}, dist={info['dist']:.3f}, grabbed={info['grabbed']}")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--model", type=str, default=None, help="Path to model for testing")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--floating", action="store_true", help="Enable floating base")
    args = parser.parse_args()

    if args.mode == "test":
        if args.model is None:
            print("Provide --model path to test")
        else:
            test_policy(args.model)
    else:
        env = SubprocVecEnv([make_env(i, floating_base=args.floating) for i in range(4)])

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            verbose=1,
            tensorboard_log=LOG_DIR,
        )

        checkpoint = CheckpointCallback(
            save_freq=50_000,
            save_path=MODEL_DIR,
            name_prefix="ppo_arm",
        )

        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint,
            tb_log_name="ppo_arm_run",
        )

        model.save(os.path.join(MODEL_DIR, "ppo_arm_final.zip"))
        env.close()
        print(f"Training done. Model saved to {MODEL_DIR}/ppo_arm_final.zip")
