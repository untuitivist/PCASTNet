from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch


def clear_cuda_cache(repeats: int = 50000) -> None:
    """Keep legacy aggressive CUDA cache clearing in one visible place."""
    for _ in range(repeats):
        torch.cuda.empty_cache()


def suffixed_run_dir(base_dir: str | Path, model_name: str, stage_suffix: str) -> Path:
    """Match the legacy STC save directory convention exactly.

    Legacy stage code expects paths like:
        <base>_<model_name>_c
        <base>_<model_name>_st
    """
    return Path(f"{base_dir}_{model_name}_{stage_suffix}")


def write_json(path: str | Path, payload: dict[str, Any], indent: int | None = None) -> None:
    path = Path(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)


def read_json_or_empty(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def update_json(path: str | Path, values: dict[str, Any], indent: int | None = 4) -> None:
    payload = read_json_or_empty(path)
    payload.update(values)
    write_json(path, payload, indent=indent)


def concat_datasets(datasets: Iterable[Any]) -> Any | None:
    merged = None
    for dataset in datasets:
        merged = dataset if merged is None else merged + dataset
    return merged
