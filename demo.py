#!/usr/bin/env python3
"""Single Python entrypoint for the reset-local PCASTNet demo.

Examples:
    python demo.py --dry-run
    python demo.py --stage encoder
    python demo.py --stage all
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG = "configs/experiments/cwru_bjtu.json"
RUN_PREFIX = "pcastnet_cwru_bjtu"
ENCODER_RUN_PREFIX = "pcastnet_encoder_cwru_bjtu"


def bootstrap() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


def timestamp_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PCASTNet CWRU->BJTU demo from one Python file.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Source config JSON.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["encoder", "style-transfer", "generate", "classifier", "all"],
        help="Stage to run. Long stages write stdout/stderr to train.log.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate paths only; do not train or generate.")
    parser.add_argument("--run-id", default=None, help="Run directory name under experiments/.")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory. Overrides --run-id.")
    return parser


def choose_run_prefix(stage: str) -> str:
    return ENCODER_RUN_PREFIX if stage == "encoder" else RUN_PREFIX


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return resolve_path(args.run_dir).resolve()
    run_id = args.run_id or timestamp_id(choose_run_prefix(args.stage))
    return (PROJECT_ROOT / "experiments" / run_id).resolve()


def materialize_effective_config(source_config: Path, run_dir: Path) -> tuple[Path, Path, dict]:
    cfg = read_json(source_config)
    flow1_dir = run_dir / "flow1"
    cfg["encoder-save-dir"] = str(flow1_dir)
    encoder_manifest_path = run_dir / "encoder_manifest.json"
    if encoder_manifest_path.exists():
        encoder_manifest = read_json(encoder_manifest_path)
        selected_encoder_path = encoder_manifest.get("selected_encoder_path")
        if selected_encoder_path:
            cfg["encoder-path"] = selected_encoder_path
    cfg["save-dir-style-transfer"] = str(run_dir / "flow2")
    cfg["style-transfer-dataset-dir"] = str(run_dir / "generated")
    cfg["save-dir-classifier"] = str(run_dir / "downstream_cnn")

    run_dir.mkdir(parents=True, exist_ok=True)
    original_config = run_dir / "config.json"
    effective_config = run_dir / "effective_config.json"
    original_config.write_text(source_config.read_text(encoding="utf-8-sig"), encoding="utf-8")
    write_json(effective_config, cfg)
    return original_config, effective_config, cfg


def write_status(run_dir: Path, state: str, stage: str, message: str, exit_code: int = 0) -> None:
    write_json(
        run_dir / "status.json",
        {
            "state": state,
            "stage": stage,
            "message": message,
            "exit_code": exit_code,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "run_dir": str(run_dir),
            "log": str(run_dir / "train.log"),
            "effective_config": str(run_dir / "effective_config.json"),
        },
    )


@contextlib.contextmanager
def redirect_to_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            yield


def load_config(config_path: Path):
    from pcastnet.config import ExperimentConfig

    return ExperimentConfig.load(config_path)


def dry_run(config_path: Path, stage: str, run_dir: Path) -> int:
    from pcastnet.assets import validate_environment

    _original_config, effective_config_path, _effective_cfg = materialize_effective_config(config_path, run_dir)
    config = load_config(effective_config_path)
    print(json.dumps(validate_environment(config), indent=2, ensure_ascii=False))
    print(f"[dry-run] Stage requested: {stage}. No training or generation was started.", flush=True)
    print(f"[dry-run] Run directory prepared for validation only: {run_dir}", flush=True)
    return 0


def run_encoder(config) -> None:
    from pcastnet.workflow import train_encoder

    print("[PCASTNet] Training encoder.", flush=True)
    train_encoder(config)


def run_style_transfer(config) -> None:
    from pcastnet.workflow import train_style_transfer

    print("[PCASTNet] Training style-transfer.", flush=True)
    train_style_transfer(config)


def run_generate(config) -> None:
    from pcastnet.workflow import generate_dataset

    print("[PCASTNet] Generating style-transfer dataset.", flush=True)
    generate_dataset(config)


def run_classifier(config) -> None:
    from pcastnet.workflow import train_classifier

    print("[PCASTNet] Training downstream classifier.", flush=True)
    train_classifier(config)


def selected_stages(stage: str) -> list[tuple[str, callable]]:
    stages: list[tuple[str, callable]] = []
    if stage in {"encoder", "all"}:
        stages.append(("encoder", run_encoder))
    if stage in {"style-transfer", "all"}:
        stages.append(("style-transfer", run_style_transfer))
    if stage in {"generate", "all"}:
        stages.append(("generate", run_generate))
    if stage in {"classifier", "all"}:
        stages.append(("classifier", run_classifier))
    return stages


def write_manifest(run_dir: Path, run_id: str, effective_cfg: dict, stage: str) -> None:
    encoder_manifest_path = run_dir / "encoder_manifest.json"
    encoder_manifest = read_json(encoder_manifest_path) if encoder_manifest_path.exists() else {}
    write_json(
        run_dir / "pipeline_manifest.json",
        {
            "exp_name": RUN_PREFIX,
            "ts": run_id,
            "stage_selector": stage,
            "exp_dir": str(run_dir),
            "paper_protocol": "CWRU->BJTU-RAO: 500 reference train samples + 50 monitored train samples; encoder uses only 50 monitored samples.",
            "encoder_runtime_split": (
                f"in-memory per-class random monitored-only n={effective_cfg['encoder-sample-scale']}, "
                f"{effective_cfg['encoder-train-ratio']:.2f}:{1 - effective_cfg['encoder-train-ratio']:.2f} "
                "from style/train only"
            ),
            "encoder_path": encoder_manifest.get("selected_encoder_path", effective_cfg.get("encoder-path")),
            "encoder_manifest": str(encoder_manifest_path) if encoder_manifest else None,
            "encoder_save_prefix": effective_cfg["encoder-save-dir"],
            "flow2_dir": effective_cfg["save-dir-style-transfer"],
            "generated_dir": effective_cfg["style-transfer-dataset-dir"],
            "downstream_dir": effective_cfg["save-dir-classifier"],
            "log": str(run_dir / "train.log"),
            "note": "Single-entry demo.py run. Encoder uses 50 monitored style/train samples, then samples train/valid split in memory; no split files and no canonical encoder copy are written.",
        },
    )


def run(args: argparse.Namespace) -> int:
    source_config = resolve_path(args.config).resolve()
    if args.dry_run:
        return dry_run(source_config, args.stage, make_run_dir(args))

    run_dir = make_run_dir(args)
    run_id = args.run_id or run_dir.name
    log_path = run_dir / "train.log"
    _original_config, effective_config_path, effective_cfg = materialize_effective_config(source_config, run_dir)
    write_json(
        run_dir / "meta.json",
        {
            "experiment_name": RUN_PREFIX,
            "timestamp": run_id,
            "project_root": str(PROJECT_ROOT),
            "run_dir": str(run_dir),
            "cfg_path": str(effective_config_path),
            "stage_selector": args.stage,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        },
    )
    write_status(run_dir, "running", args.stage, "Demo run started.")

    try:
        config = load_config(effective_config_path)
        with redirect_to_log(log_path):
            print(f"[PCASTNet] run_dir={run_dir}", flush=True)
            print(f"[PCASTNet] effective_config={effective_config_path}", flush=True)
            for stage_name, func in selected_stages(args.stage):
                write_status(run_dir, "running", stage_name, f"Running {stage_name}.")
                print("", flush=True)
                print("=" * 88, flush=True)
                print(f"[PCASTNet] START stage={stage_name}", flush=True)
                print("=" * 88, flush=True)
                func(config)
                print("=" * 88, flush=True)
                print(f"[PCASTNet] DONE stage={stage_name}", flush=True)
                print("=" * 88, flush=True)
        write_manifest(run_dir, run_id, effective_cfg, args.stage)
        write_status(run_dir, "complete", args.stage, "Demo run completed.")
        print(f"[PCASTNet] Completed: {run_dir}", flush=True)
        print(f"[PCASTNet] Log: {log_path}", flush=True)
        return 0
    except Exception as exc:
        with redirect_to_log(log_path):
            print("[PCASTNet][ERROR]", repr(exc), flush=True)
            traceback.print_exc()
        write_status(run_dir, "failed", args.stage, str(exc), 1)
        print(f"[PCASTNet] Failed: {run_dir}", flush=True)
        print(f"[PCASTNet] Log: {log_path}", flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    bootstrap()
    args = make_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
