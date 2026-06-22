"""Configuration loading for reproducible PCASTNet runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import project_path


def _normalize_key(key: str) -> str:
    return key.replace("-", "_")


@dataclass(slots=True)
class ExperimentConfig:
    experiment: str = "reproduce"
    model_name: str = "CWRU-BJTU"
    num_classes: int = 4

    content_dataset_dir: str = "data/datasets/machines/CWRU/cwts"
    style_dataset_dir: str = "data/datasets/machines/BJTU/cwts"
    style_transfer_dataset_dir: str = "experiments/generated/CWRU-BJTU_adailn_s"
    vgg_pretrained_path: str = "src/models/vgg/vgg.pth"
    encoder_path: str = (
        "experiments/pretrained_encoders/CWRU-BJTU/CWRU-BJTU_encoder.pth.tar"
    )
    encoder_sample_scale: int = 50
    encoder_train_ratio: float = 0.8
    encoder_max_iter: int = 1000
    encoder_batch_size: int = 32
    encoder_preheat: int = 30
    encoder_realy_stop: int = 50
    encoder_save_dir: str = "experiments/pretrained_encoders/CWRU-BJTU"

    content_train_scale: int = 500
    content_valid_scale: int = 0
    content_test_scale: int = 0
    style_train_scale: int = 50
    style_valid_scale: int = 100
    style_test_scale: int = 500

    max_iter_style_transfer: int = 10000
    max_iter_classifier: int = 200
    batch_size_style_transfer: int = 16
    batch_size_classifier: int = 32
    lr: float = 1e-4
    lr_decay: float = 5e-5
    preheat_style_transfer: int = 200
    preheat_classifier: int = 20
    realy_stop_style_transfer: int = 500
    realy_stop_classifier: int = 50

    content_weight: float = 1.0
    style_weight: float = 10.0
    perceptual_weight: float = 0.0
    tv_weight: float = 0.0
    energy_weight: float = 1.0
    rho_c: float | None = 1.0
    rho_s: float | None = None

    folder_label: str = "CWRU-BJTU"
    use_train_datasets: list[str] = field(default_factory=lambda: ["style", "style_transfer"])
    use_val_datasets: list[str] = field(default_factory=lambda: ["style"])
    use_test_datasets: list[str] = field(default_factory=lambda: ["style"])

    save_dir_style_transfer: str | None = None
    save_dir_classifier: str = "experiments/CNN_adailn_s"

    seed: int = 42

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ExperimentConfig":
        normalized = {_normalize_key(k): v for k, v in values.items()}
        field_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = sorted(set(normalized) - field_names)
        if unknown:
            raise ValueError(f"Unknown config field(s): {', '.join(unknown)}")
        return cls(**normalized)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        resolved = project_path(path)
        if resolved is None:
            raise ValueError("Config path is required")
        with resolved.open("r", encoding="utf-8-sig") as fh:
            return cls.from_mapping(json.load(fh))

    def to_legacy_stc_kwargs(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "content_dataset_dir": str(project_path(self.content_dataset_dir)),
            "style_dataset_dir": str(project_path(self.style_dataset_dir)),
            "style_transfer_dataset_dir": str(project_path(self.style_transfer_dataset_dir)),
            "content_train_dataset_scale": self.content_train_scale,
            "content_valid_dataset_scale": self.content_valid_scale,
            "content_test_dataset_scale": self.content_test_scale,
            "style_train_dataset_scale": self.style_train_scale,
            "style_valid_dataset_scale": self.style_valid_scale,
            "style_test_dataset_scale": self.style_test_scale,
            "test_content_size": 512,
            "test_style_size": 512,
            "test_crop": False,
        }

    def to_encoder_stc_kwargs(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name + "_encoder",
            "num_classes": self.num_classes,
            "content_dataset_dir": str(project_path(self.content_dataset_dir)),
            "style_dataset_dir": str(project_path(self.style_dataset_dir)),
            "style_transfer_dataset_dir": str(project_path(self.style_transfer_dataset_dir)),
            "content_train_dataset_scale": self.content_train_scale,
            "content_valid_dataset_scale": self.content_valid_scale,
            "content_test_dataset_scale": self.content_test_scale,
            "style_train_dataset_scale": self.style_train_scale,
            "style_valid_dataset_scale": self.style_valid_scale,
            "style_test_dataset_scale": self.style_test_scale,
            "test_content_size": 512,
            "test_style_size": 512,
            "test_crop": False,
        }
