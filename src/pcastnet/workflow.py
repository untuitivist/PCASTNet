"""Compatibility facade for the PCASTNet workflow API."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .assets import print_environment_report, validate_environment
from .config import ExperimentConfig
from .paths import project_path
from .runner import generate_dataset, run_stage, train_classifier, train_encoder, train_style_transfer


def dump_resolved_config(config: ExperimentConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in [
        "content_dataset_dir",
        "style_dataset_dir",
        "style_transfer_dataset_dir",
        "encoder_path",
    ]:
        data[key + "_resolved"] = str(project_path(data[key]))
    return data
