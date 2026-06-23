import sys, os, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spaceship_env import SpaceshipArmEnv
from stable_baselines3 import SAC
from gymnasium import spaces

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE, "training_runs")

def list_runs():
    if not os.path.isdir(RUNS_DIR):
        return []
    return sorted([
        d for d in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, d))
    ])

def list_checkpoints(run_dir):
    zips = sorted(glob.glob(os.path.join(run_dir, "models", "*.zip")))
    result = []
    for z in zips:
        size_mb = os.path.getsize(z) / (1024 * 1024)
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(z)))
        result.append((z, os.path.basename(z), size_mb, mtime))
    return result

def load_config(run_dir):
    cf = os.path.join(run_dir, "config.json")
    if os.path.isfile(cf):
        with open(cf) as f:
            return json.load(f)
    return {}

def pick(options, title):
    if not options:
        print(f"No {title} found.", flush=True)
        return None
    print(f"\n{'='*50}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*50}", flush=True)
    for i, (_, label, *rest) in enumerate(options):
        extra = f"  ({rest[0]:.1f}MB, {rest[1]})" if rest else ""
        print(f"  [{i+1}] {label}{extra}", flush=True)
    print(f"  [0] Quit", flush=True)
    print(f"{'='*50}", flush=True)
    try:
        inp = input("Select: ").strip()
        idx = int(inp) - 1
        if 0 <= idx < len(options):
            return options[idx][0]
    except:
        pass
    return None

def main():
    print("=" * 50, flush=True)
    print("  Stardance Viewer", flush=True)
    print("  ESC to exit viewer window", flush=True)
    print("=" * 50, flush=True)

    if not os.path.isdir(RUNS_DIR):
        print("No training_runs/ directory found. Train first!", flush=True)
        return

    runs = list_runs()
    if not runs:
        print("No training runs found.", flush=True)
        return

    run_opts = [(os.path.join(RUNS_DIR, r), r) for r in runs]
    run_path = pick(run_opts, "Select Training Run")
    if run_path is None:
        return

    config = load_config(run_path)
    if config:
        print(f"  Config: {json.dumps(config, indent=2)}", flush=True)

    ckpts = list_checkpoints(run_path)
    ckpt_path = pick(ckpts, "Select Checkpoint")
    if ckpt_path is None:
        return

    print(f"\nLoading {os.path.basename(ckpt_path)}...", flush=True)
    env = SpaceshipArmEnv(space_gravity=True, render_mode="human")
    model = SAC.load(ckpt_path, env=env, device="cuda")
    print("Loaded. Playing episodes...", flush=True)

    ep = 0
    try:
        while env.is_running():
            obs, _ = env.reset()
            total_reward = 0.0
            ep += 1

            while env.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                env.render()
                if info.get("grabbed"):
                    print(f"  GRAB at step {info['step']}!", flush=True)
                if terminated or truncated:
                    break

            if not env.is_running():
                break
            dist = info.get("dist", 1.0)
            print(f"Episode {ep}: reward={total_reward:.0f}, dist={dist:.3f}, "
                  f"grabbed={info.get('grabbed', False)}, steps={info.get('step', 0)}", flush=True)
    finally:
        env.close()

if __name__ == "__main__":
    main()
