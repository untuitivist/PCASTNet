"""Bridge to legacy modules while keeping imports local to reset/."""

from __future__ import annotations

import sys
from pathlib import Path

from .paths import PROJECT_ROOT, SRC_DIR


def configure_legacy_imports() -> None:
    """Make legacy absolute imports resolve inside reset only."""
    wanted = [str(SRC_DIR)]
    resolved_wanted = {str(Path(item).resolve()) for item in wanted}
    sys.path[:] = [
        item
        for item in sys.path
        if not item or str(Path(item).resolve()) not in resolved_wanted
    ]
    sys.path.insert(0, str(SRC_DIR))


def import_legacy():
    """Import legacy training objects after local path setup."""
    configure_legacy_imports()
    from STC import STC  # type: ignore
    from models.adailn import build_ADAILNet_module  # type: ignore
    from models.cnn import build_CNN  # type: ignore
    from models.vgg import build_VGG_module  # type: ignore

    return STC, build_ADAILNet_module, build_CNN, build_VGG_module


def assert_no_parent_repo_imports() -> None:
    """Fail fast if a legacy module was imported from outside reset."""
    reset_root = PROJECT_ROOT.resolve()
    offenders: list[tuple[str, Path]] = []
    for name, module in list(sys.modules.items()):
        file_name = getattr(module, "__file__", None)
        if not file_name:
            continue
        path = Path(file_name).resolve()
        if name in {"STC", "function", "data_loader", "sampler"} or name.startswith("models"):
            try:
                path.relative_to(reset_root)
            except ValueError:
                offenders.append((name, path))
    if offenders:
        detail = ", ".join(f"{name}={path}" for name, path in offenders)
        raise RuntimeError(f"Legacy import escaped reset project root: {detail}")
