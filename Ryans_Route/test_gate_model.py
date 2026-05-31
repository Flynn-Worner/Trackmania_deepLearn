"""
Deterministic rollout tester for the gate-based TMNF bridge model.

Usage:
    python test_gate_model.py
    python test_gate_model.py --model-path models/tmnf_ppo_final.zip --port 9000
    python test_gate_model.py --episodes 5 --delay 0.05
"""

from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO

from env import TrackmaniaEnv


DEFAULT_MODEL_PATH = "models/tmnf_ppo_final.zip"
DEFAULT_PORT = 9000


def run_test(model_path: str, port: int, episodes: int, delay: float):
    print("=" * 60)
    print("  TMNF Gate Model Test")
    print("=" * 60)
    print(f"  Model   : {model_path}")
    print(f"  Port    : {port}")
    print(f"  Episodes: {episodes}")
    print(f"  Delay   : {delay:.2f}s")
    print()

    model = PPO.load(model_path)
    env = TrackmaniaEnv(port=port)
    env.connect()

    total_reward = 0.0
    total_gates = 0

    try:
        for episode_idx in range(1, episodes + 1):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_gates = 0
            done = False

            print(f"\n[Episode {episode_idx}] starting")
            print(f"  Initial obs: {obs}")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += float(reward)
                episode_steps += 1
                episode_gates = int(info.get("gates_crossed", episode_gates))

                print(
                    f"  step={episode_steps:4d} | action={action} | "
                    f"reward={reward:+7.3f} | gates={episode_gates} | "
                    f"reason={info.get('termination_reason', '')}"
                )

                done = bool(terminated or truncated)
                if delay > 0:
                    time.sleep(delay)

            total_reward += episode_reward
            total_gates += episode_gates

            print(
                f"[Episode {episode_idx}] ended | steps={episode_steps} | "
                f"reward={episode_reward:+.2f} | gates={episode_gates} | "
                f"reason={info.get('termination_reason', '')}"
            )

        print("\n" + "=" * 60)
        print("  Summary")
        print("=" * 60)
        print(f"  Total reward : {total_reward:+.2f}")
        print(f"  Total gates   : {total_gates}")
        print(f"  Avg reward    : {total_reward / max(1, episodes):+.2f}")
        print(f"  Avg gates     : {total_gates / max(1, episodes):.1f}")
        print("=" * 60)

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Deterministic TMNF PPO rollout tester")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    run_test(
        model_path=args.model_path,
        port=args.port,
        episodes=args.episodes,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
