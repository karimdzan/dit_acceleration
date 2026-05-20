"""Shared helpers for the Hydra CLI entrypoints."""
from pathlib import Path

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")
