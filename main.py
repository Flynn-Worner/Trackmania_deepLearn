"""
Entry point for the TMNF PPO agent -- training + recording.

Modes:
    python main.py                              # train (default)
    python main.py --record                     # record waypoints (drive at 0.1x speed)
    python main.py --record --spacing 5         # record with 5 m waypoint spacing
    python main.py --ports 8483 8484            # multi-instance training
    python main.py --new                        # fresh model, ignore checkpoint
    python main.py --debug                      # small n_steps for fast TB feedback

After recording, run:
    python generate_gates.py                    # convert waypoints -> gates

TensorBoard:
    tensorboard --logdir tensorboard
"""

import argparse
import os
import sys


def _parse_args():
    parser = argparse.ArgumentParser(
        description="TMNF PPO agent -- train or record",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # -- Mode --
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record mode: drive the ideal racing line at 0.1x speed. "
             "Saves waypoints to data/waypoints.json.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=10.0,
        help="(Record mode) Min metres between waypoints (default: 10)",
    )
    # -- Training args --
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=[8483],
        help="TMInterface port(s) - one per running TM window (default: 8483)",
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
        help="Debug mode: n_steps=128 so TB updates every ~1 min",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Force creating a new model, ignoring any existing saved checkpoint",
    )
    return parser.parse_args()


def main():
    # Always run from repo root so relative paths resolve correctly.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    sys.path.insert(0, repo_root)

    args = _parse_args()

    # ======================================================================
    # RECORD MODE
    # ======================================================================
    if args.record:
        print("=" * 56)
        print("  TMNF Waypoint Recorder")
        print("=" * 56)
        print(f"  Port     : {args.ports[0]}")
        print(f"  Spacing  : {args.spacing:.1f} m")
        print(f"  Speed    : 0.1x (slow for accurate driving)")
        print()
        print("  Drive the car down the MIDDLE of the track.")
        print("  Recording stops when you cross the finish line")
        print("  or press Ctrl+C.")
        print()

        from record_waypoints import record
        record(port=args.ports[0], spacing_m=args.spacing)
        return

    # ======================================================================
    # TRAINING MODE
    # ======================================================================
    tb_dir = args.tb_dir or os.path.join(repo_root, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    # Check that gates.json exists
    gates_path = os.path.join(repo_root, "data", "gates.json")
    if not os.path.exists(gates_path):
        print("=" * 56)
        print("  ERROR: data/gates.json not found!")
        print("=" * 56)
        print()
        print("  You need to record the racing line first:")
        print("    1. python main.py --record")
        print("    2. python generate_gates.py")
        print()
        print("  Then run training again.")
        sys.exit(1)

    print("=" * 56)
    print("  TMNF PPO Training (Gate-Based)")
    print("=" * 56)
    print(f"  Repo root  : {repo_root}")
    print(f"  Ports      : {args.ports}")
    print(f"  Timesteps  : {args.timesteps:,}")
    print(f"  TB dir     : {tb_dir}")
    print(f"  Model path : {args.model_path}")
    print(f"  Debug mode : {args.debug}")
    print(f"  Gates file : {gates_path}")
    print()
    print("  TensorBoard command:")
    print(f'    tensorboard --logdir "{tb_dir}"')
    print()
    if len(args.ports) > 1:
        print("  Multi-instance checklist:")
        for i, p in enumerate(args.ports):
            print(f"    [{i+1}] Open TmForever + TMInterface  ->  set custom_port {p}  ->  load map")
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
