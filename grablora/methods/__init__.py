"""Importing this package registers GRAB-LoRA in the METHODS registry."""
from __future__ import annotations

from .base import METHODS, Method, build, register   # noqa: F401
from . import bg_lora         # noqa: F401  registers "bg_lora"

#: display order for figures/tables
ORDER = ["bg_lora"]

__all__ = ["METHODS", "Method", "build", "register", "ORDER"]