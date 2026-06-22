"""Execution stages for the PCASTNet demo pipeline."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path

from .assets import print_environment_report
from .config import ExperimentConfig
from .legacy import assert_no_parent_repo_imports, import_legacy
from .paths import project_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[warn] Seed setup partially failed: {exc}", flush=True)


def _log(message: str = "") -> None:
    print(message, flush=True)


def _stage_banner(title: str) -> None:
    line = "=" * 88
    _log()
    _log(line)
    _log(f"[PCASTNet] {title}")
    _log(line)


def _summarize_config(config: ExperimentConfig) -> None:
    _stage_banner("Resolved experiment configuration")
    rows = {
        "experiment": config.experiment,
        "model_name": config.model_name,
        "content_dataset_dir": project_path(config.content_dataset_dir),
        "style_dataset_dir": project_path(config.style_dataset_dir),
        "style_transfer_dataset_dir": project_path(config.style_transfer_dataset_dir),
        "vgg_pretrained_path": project_path(config.vgg_pretrained_path),
        "encoder_runtime_split": (
            f"style/train in-memory monitored-only n={config.encoder_sample_scale}, "
            f"{config.encoder_train_ratio:.2f}:{1 - config.encoder_train_ratio:.2f}"
        ),
        "encoder_path": project_path(config.encoder_path),
        "save_dir_style_transfer": project_path(config.save_dir_style_transfer) if config.save_dir_style_transfer else None,
        "save_dir_classifier": project_path(config.save_dir_classifier),
        "seed": config.seed,
        "num_classes": config.num_classes,
        "style_transfer_iters": config.max_iter_style_transfer,
        "classifier_iters": config.max_iter_classifier,
        "loss_weights": (
            f"content={config.content_weight}, style={config.style_weight}, "
            f"perceptual={config.perceptual_weight}, tv={config.tv_weight}, "
            f"energy={config.energy_weight}"
        ),
    }
    for key, value in rows.items():
        _log(f" - {key}: {value}")


def _run_named_stage(name: str, func: Callable[[ExperimentConfig], None], config: ExperimentConfig) -> None:
    _stage_banner(f"START stage={name}")
    started = time.time()
    try:
        func(config)
    except Exception as exc:
        elapsed = time.time() - started
        _log(f"[PCASTNet][FAILED] stage={name} elapsed={elapsed:.1f}s error={type(exc).__name__}: {exc}")
        raise
    elapsed = time.time() - started
    _stage_banner(f"DONE stage={name} elapsed={elapsed:.1f}s")


def _legacy_classifier_output_dir(save_dir: Path, model_name: str) -> Path:
    return Path(str(save_dir) + f"_{model_name}_c")


def _select_encoder_checkpoint(save_dir: Path, model_name: str) -> Path:
    output_dirs = [save_dir, _legacy_classifier_output_dir(save_dir, model_name + "_encoder")]
    last: list[Path] = []
    best: list[Path] = []
    for output_dir in output_dirs:
        if output_dir.exists():
            last.extend(output_dir.glob("*_last_encoder.pth.tar"))
            best.extend(output_dir.glob("*_best_encoder.pth.tar"))
    last = sorted(last)
    best = sorted(best)
    selected = last[-1] if last else (best[-1] if best else None)
    if selected is None:
        searched = ", ".join(str(path) for path in output_dirs)
        raise FileNotFoundError(
            "No *_last_encoder.pth.tar or *_best_encoder.pth.tar found after encoder training. "
            f"Searched: {searched}"
        )
    return selected.resolve()


def _write_encoder_manifest(config: ExperimentConfig, selected_encoder: Path) -> None:
    import json

    save_dir = project_path(config.encoder_save_dir)
    manifest_dir = save_dir.parent if save_dir is not None else selected_encoder.parent
    manifest_path = manifest_dir / "encoder_manifest.json"
    payload = {
        "selected_encoder_path": str(selected_encoder),
        "selection_policy": "prefer *_last_encoder.pth.tar, fallback to *_best_encoder.pth.tar",
        "copy_policy": "manifest-only; no canonical encoder copy is created",
        "encoder_output_dir": str(selected_encoder.parent),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Encoder manifest saved at {manifest_path}", flush=True)
    print(f"[+] Downstream encoder path set to {selected_encoder}", flush=True)


def _build_stc(config: ExperimentConfig, network_type: str):
    STC, build_ADAILNet_module, build_CNN, build_VGG_module = import_legacy()
    assert_no_parent_repo_imports()
    stc = STC(**config.to_legacy_stc_kwargs())
    if network_type == "adailn":
        encoder_path = project_path(config.encoder_path)
        if encoder_path is None or not encoder_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
        stc.net = build_ADAILNet_module(
            num_classes=config.num_classes,
            rho_c=config.rho_c,
            rho_s=config.rho_s,
            encoder_pth=str(encoder_path),
        )
    elif network_type == "cnn":
        stc.net = build_CNN(num_classes=config.num_classes)
    elif network_type == "vgg":
        stc.net = build_VGG_module(num_classes=config.num_classes, encoder_pth=str(project_path(config.encoder_path)))
    else:
        raise ValueError(f"Unsupported network type: {network_type}")
    return stc


def train_encoder(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    STC, _build_ADAILNet_module, _build_CNN, build_VGG_module = import_legacy()
    assert_no_parent_repo_imports()
    stc = STC(**config.to_encoder_stc_kwargs())
    stc.encoder_runtime_split = True
    stc.encoder_sample_scale = config.encoder_sample_scale
    stc.encoder_train_ratio = config.encoder_train_ratio
    stc.encoder_runtime_seed = config.seed
    vgg_pretrained_path = project_path(config.vgg_pretrained_path)
    if vgg_pretrained_path is None or not vgg_pretrained_path.exists():
        raise FileNotFoundError(f"VGG pretrained encoder not found: {vgg_pretrained_path}")
    print(f"[+] Initialize encoder from pretrained VGG: {vgg_pretrained_path}", flush=True)
    stc.net = build_VGG_module(num_classes=config.num_classes, encoder_pth=str(vgg_pretrained_path))
    stc.train_classifier(
        max_iter=config.encoder_max_iter,
        save_iter=1e25,
        save_dir=str(project_path(config.encoder_save_dir)),
        lr=config.lr,
        lr_decay=config.lr_decay,
        batch_size=config.encoder_batch_size,
        use_train_datasets=["style"],
        use_val_datasets=["style"],
        use_test_datasets=["style"],
        preheat=config.encoder_preheat,
        realy_stop=config.encoder_realy_stop,
        freeze_encoder=False,
        freeze_classifier=False,
    )
    save_dir = project_path(config.encoder_save_dir)
    if save_dir is None:
        raise ValueError("encoder_save_dir may not be None")
    selected = _select_encoder_checkpoint(save_dir, config.model_name)
    config.encoder_path = str(selected)
    _write_encoder_manifest(config, selected)


def train_style_transfer(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    stc = _build_stc(config, "adailn")
    kwargs = {
        "max_iter": config.max_iter_style_transfer,
        "lr": config.lr,
        "lr_decay": config.lr_decay,
        "batch_size": config.batch_size_style_transfer,
        "content_weight": config.content_weight,
        "style_weight": config.style_weight,
        "perceptual_weight": config.perceptual_weight,
        "totalvariation_weight": config.tv_weight,
        "energe_weight": config.energy_weight,
        "preheat": config.preheat_style_transfer,
        "realy_stop": config.realy_stop_style_transfer,
    }
    if config.save_dir_style_transfer:
        kwargs["save_dir"] = str(project_path(config.save_dir_style_transfer))
    stc.train_style_transfer(**kwargs)


def generate_dataset(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    stc = _build_stc(config, "adailn")
    stc.make_adailn_dataset(config.folder_label)


def train_classifier(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    stc = _build_stc(config, "cnn")
    stc.train_classifier(
        max_iter=config.max_iter_classifier,
        save_iter=1e25,
        save_dir=str(project_path(config.save_dir_classifier)),
        lr=config.lr,
        lr_decay=config.lr_decay,
        batch_size=config.batch_size_classifier,
        use_train_datasets=config.use_train_datasets,
        use_val_datasets=config.use_val_datasets,
        use_test_datasets=config.use_test_datasets,
        preheat=config.preheat_classifier,
        realy_stop=config.realy_stop_classifier,
        freeze_encoder=False,
        freeze_classifier=False,
    )


def run_stage(config: ExperimentConfig, stage: str, dry_run: bool = False) -> None:
    if dry_run:
        print_environment_report(config)
        print(f"[dry-run] Stage requested: {stage}. No training or generation was started.", flush=True)
        return

    stages = {
        "encoder": (("encoder", train_encoder),),
        "style-transfer": (("style-transfer", train_style_transfer),),
        "generate": (("generate", generate_dataset),),
        "classifier": (("classifier", train_classifier),),
        "all": (
            ("encoder", train_encoder),
            ("style-transfer", train_style_transfer),
            ("generate", generate_dataset),
            ("classifier", train_classifier),
        ),
    }
    if stage not in stages:
        raise ValueError(f"Unsupported stage: {stage}")
    _summarize_config(config)
    _stage_banner("Preflight asset report")
    print_environment_report(config)
    started = time.time()
    _log(f"[PCASTNet] Running stage selector={stage}")
    for name, func in stages[stage]:
        _run_named_stage(name, func, config)
    _stage_banner(f"FINISHED stage selector={stage} total_elapsed={time.time() - started:.1f}s")
