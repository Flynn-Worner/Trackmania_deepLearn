"""
Entry point for training the TMNF PPO agent.

Run from the repo root:
    python main.py                          # single instance, default port 8483
    python main.py --ports 8483 8484        # two instances
    python main.py --debug                  # fast TensorBoard updates (small n_steps)
    python main.py --new                    # ignore saved model, start fresh

TensorBoard:
    tensorboard --logdir tensorboard
"""

import argparse
import os
import sys


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train TMNF PPO driving agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=[8483],
        help="TMInterface port(s) – one per running TM window (default: 8483)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--tb-dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: <repo_root>/tensorboard)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/saved/ppo_trackmania_final",
        help="Save/load path without .zip (default: models/saved/ppo_trackmania_final)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: n_steps=128 so TB updates every ~1 min instead of waiting for a full rollout",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Force creating a new model, ignoring any existing saved checkpoint",
    )
    return parser.parse_args()


def main():
    # Always run from repo root so relative paths (tensorboard/, models/) resolve correctly.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    sys.path.insert(0, repo_root)

    args = _parse_args()

    tb_dir = args.tb_dir or os.path.join(repo_root, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    print("=" * 56)
    print("  TMNF PPO Training")
    print("=" * 56)
    print(f"  Repo root  : {repo_root}")
    print(f"  Ports      : {args.ports}")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  TB dir     : {tb_dir}")
    print(f"  Model path : {args.model_path}")
    print(f"  Debug mode : {args.debug}")
    print()
    print("  TensorBoard command:")
    print(f'    tensorboard --logdir "{tb_dir}"')
    print()
    if len(args.ports) > 1:
        print("  Multi-instance checklist:")
        for i, p in enumerate(args.ports):
            print(f"    [{i+1}] Open TmForever + TMInterface  →  set custom_port {p}  →  load map")
        print()
    else:
        print("  Ensure TmForever + TMInterface is running on port", args.ports[0])
        print()

    from training.train import run_training

    run_training(
        ports=args.ports,
        total_timesteps=args.timesteps,
        tensorboard_dir=tb_dir,
        model_path=args.model_path,
        debug=args.debug,
        force_new=args.new,
    )


if __name__ == "__main__":
    main()
