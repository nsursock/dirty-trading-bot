"""Centralized YAML configuration loader (attribute-style access)."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load(path):
    with open(path) as fh:
        return _wrap(yaml.safe_load(fh))


def load_smoke():
    return load(CONFIG_DIR / "smoke.yaml")
