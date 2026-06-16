"""PaCT experiment entry point.

Delegates all logic to experiment.run_experiment. Run examples:
  python main.py
  python main.py --datasets Helpdesk.xes.gz SEPSIS.xes.gz
  python main.py --prefix_patch_size 4
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiment import run_experiment

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PaCT experiment")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--min_prefix_len", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--prefix_patch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--prefix_mode_override", choices=["auto", "w_attr", "wo_attr"], default="auto")
    args = parser.parse_args()

    run_experiment(
        datasets=args.datasets,
        n_splits=args.n_splits,
        random_state=args.random_state,
        min_prefix_len=args.min_prefix_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
        prefix_patch_size=args.prefix_patch_size,
        eval_batch_size=args.eval_batch_size,
        prefix_mode_override=args.prefix_mode_override,
    )