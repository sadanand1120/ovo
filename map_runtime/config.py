from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, default_path: str | Path | None = None, inherit: bool = True) -> dict[str, Any]:
    with open(path, "r") as handle:
        cfg_special = yaml.full_load(handle)
    inherit_from = cfg_special.get("inherit_from")
    cfg: dict[str, Any] = {}
    if inherit_from is not None and inherit:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as handle:
            cfg = yaml.full_load(handle)
    update_recursive(cfg, cfg_special)
    return cfg


def update_recursive(dict1: dict[str, Any], dict2: dict[str, Any]) -> None:
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = {}
        if isinstance(value, dict):
            update_recursive(dict1[key], value)
        else:
            dict1[key] = value
