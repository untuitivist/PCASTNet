"""Asset inspection and dataset split preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .paths import PROJECT_ROOT, project_path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def count_files(path: Path, suffixes: tuple[str, ...] = IMAGE_SUFFIXES) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)


def _path_report(path: Path, include_size: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if include_size:
        report["size_bytes"] = path.stat().st_size if path.exists() and path.is_file() else 0
    else:
        report["jpg_count"] = count_files(path)
        manifests = sorted(path.rglob("manifest.json")) if path.exists() and path.is_dir() else []
        if manifests:
            manifest_counts = {}
            for manifest in manifests:
                with manifest.open("r", encoding="utf-8-sig") as fh:
                    data = json.load(fh)
                manifest_counts[str(manifest.relative_to(path))] = len(data.get("entries", []))
            report["manifest_entry_counts"] = manifest_counts
    return report


def validate_environment(config: ExperimentConfig) -> dict[str, Any]:
    content_dir = project_path(config.content_dataset_dir)
    style_dir = project_path(config.style_dataset_dir)
    transfer_dir = project_path(config.style_transfer_dataset_dir)
    vgg_pretrained_path = project_path(config.vgg_pretrained_path)
    encoder_path = project_path(config.encoder_path)
    required = [content_dir, style_dir, transfer_dir, vgg_pretrained_path, encoder_path]
    if any(path is None for path in required):
        raise ValueError("Config paths may not be None")

    return {
        "project_root": str(PROJECT_ROOT),
        "content_dataset_dir": _path_report(content_dir),
        "style_dataset_dir": _path_report(style_dir),
        "style_transfer_dataset_dir": {
            **_path_report(transfer_dir),
            "role": "generated stage output and classifier input; demo.py overrides this to the run directory",
        },
        "vgg_pretrained_path": {
            **_path_report(vgg_pretrained_path, include_size=True),
            "role": "initial encoder weights for the encoder/classifier pretraining stage",
        },
        "encoder_path": _path_report(encoder_path, include_size=True),
        "encoder_runtime_split": {
            "source": str(style_dir / "train"),
            "paper_protocol": "Only 50 monitored samples for encoder; 500 reference train samples + 50 monitored train samples for generation/classifier.",
            "policy": (
                f"in-memory per-class random monitored-only n={config.encoder_sample_scale}, "
                f"{config.encoder_train_ratio:.2f}:{1 - config.encoder_train_ratio:.2f}; "
                "no split files are written"
            ),
        },
    }


def print_environment_report(config: ExperimentConfig) -> None:
    print(json.dumps(validate_environment(config), indent=2, ensure_ascii=False))
