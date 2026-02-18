from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager


@contextmanager
def suppress_deepspeed_discovery():
    """
    Hide deepspeed from importlib discovery while importing transformers internals.
    This prevents optional deepspeed side effects on platforms without full support.
    """
    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name, package=None):
        if name == "deepspeed":
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = patched_find_spec
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec


def disable_diffusers_peft_version_gate():
    """
    Diffusers checks peft version at import-time; disable that strict gate so
    inference can run without upgrading unrelated LoRA tooling.
    """
    os.environ.setdefault("_CHECK_PEFT", "0")
