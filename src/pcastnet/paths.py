"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


def project_path(value: str | Path | None) -> Path | None:
    """Resolve a config path relative to the reset project root."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path: str | Path | None) -> str:
    if path is None:
        return "<none>"
    return str(project_path(path))

