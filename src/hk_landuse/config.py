from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["project_root"] = str(path.parent)
    for key in cfg["paths"]:
        value = Path(cfg["paths"][key])
        if not value.is_absolute():
            value = path.parent / value
        cfg["paths"][key] = str(value.resolve())
    return cfg
