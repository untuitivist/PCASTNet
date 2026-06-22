"""Command line interface for the reset-local PCASTNet project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ExperimentConfig
from .workflow import dump_resolved_config, run_stage, validate_environment


DEFAULT_CONFIG = "configs/experiments/cwru_bjtu.json"


def _load_config(path: str) -> ExperimentConfig:
    return ExperimentConfig.load(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcastnet",
        description="PCASTNet reproduction CLI for cross-machine small-sample fault diagnosis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="Validate config, paths, imports, and dataset availability.")
    dry.add_argument("--config", default=DEFAULT_CONFIG, help="Path to experiment config JSON.")

    show = sub.add_parser("show-config", help="Print resolved config.")
    show.add_argument("--config", default=DEFAULT_CONFIG, help="Path to experiment config JSON.")

    run = sub.add_parser("run", help="Run a workflow stage. This can start long training.")
    run.add_argument("--config", default=DEFAULT_CONFIG, help="Path to experiment config JSON.")
    run.add_argument(
        "--stage",
        choices=["encoder", "style-transfer", "generate", "classifier", "all"],
        default="all",
        help="Workflow stage to run.",
    )
    run.add_argument("--dry-run", action="store_true", help="Validate only; do not train/generate.")

    return parser


def main(argv: list[str] | None = None) -> int:
    # Long experiments are usually launched in the background with stdout/stderr
    # redirected to a file. Force line buffering so STC's detailed print/tqdm
    # progress reaches the log promptly instead of sitting in Python buffers.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    config = _load_config(args.config)

    if args.command == "dry-run":
        print(json.dumps(validate_environment(config), indent=2, ensure_ascii=False))
        return 0

    if args.command == "show-config":
        print(json.dumps(dump_resolved_config(config), indent=2, ensure_ascii=False))
        return 0

    if args.command == "run":
        run_stage(config, args.stage, dry_run=args.dry_run)
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
