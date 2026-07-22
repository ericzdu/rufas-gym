"""Locating and importing RuFaS without installing it.

RuFaS is a sibling checkout, not a dependency we install, so every entry point that
touches it goes through here. Two things matter:

* `RUFAS` must be importable — we put the repo root on `sys.path`.
* RuFaS resolves its input paths **relative to its own repo root**, so anything that
  loads a scenario or runs a simulation has to do it with the cwd set there. That is
  what `rufas_cwd()` is for; `runner.py` in rufas-web does the same thing.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_THIS = Path(__file__).resolve()


def rufas_root() -> Path:
    """Repo root of the RuFaS checkout. Override with the RUFAS_ROOT env var."""
    env = os.environ.get("RUFAS_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = (_THIS.parent.parent.parent / "RuFaS").resolve()
    if not (root / "RUFAS").is_dir():
        raise RuntimeError(
            f"No RuFaS checkout at {root} (expected a RUFAS/ package inside). "
            "Set RUFAS_ROOT to point at it."
        )
    return root


def ensure_importable() -> Path:
    """Put the RuFaS root on `sys.path` and return it. Idempotent."""
    root = rufas_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@contextmanager
def rufas_cwd() -> Iterator[Path]:
    """Run a block with the cwd at the RuFaS root, restoring it afterwards."""
    root = ensure_importable()
    prev = Path.cwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)


def resolve(path: str | Path) -> Path:
    """Resolve a RuFaS-relative input path against the RuFaS root."""
    p = Path(path)
    return p if p.is_absolute() else (rufas_root() / p)
