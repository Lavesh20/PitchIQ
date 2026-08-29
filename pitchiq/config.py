"""Secrets and paths.

Every credential this project uses lives in ``.env`` at the repo root
and is read through :func:`secret`. Nothing else reads the file, no
value is ever logged, and ``.env`` is gitignored.

Containment rule for this project: credentials and any GitHub operation
performed with them stay scoped to this repository. The token here is
not for other repos, other accounts, or other machines, and no external
repository content is pulled into this tree.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"

load_dotenv(ENV_FILE)


class MissingSecret(RuntimeError):
    """A required credential is not present in .env."""


def secret(name: str, required: bool = True) -> str | None:
    """Read a credential by name.

    Returns the value, or raises :class:`MissingSecret` when it is
    absent and ``required``. The value is never printed, and the error
    message names only the key.
    """
    value = os.getenv(name)

    if value:
        return value

    if required:
        raise MissingSecret(
            f"{name} is not set. Add it to {ENV_FILE.name} at the repo root."
        )

    return None


def masked(name: str) -> str:
    """A safe-to-print description of a credential, for logs and reports."""
    value = os.getenv(name)

    if not value:
        return f"{name}=<unset>"

    return f"{name}=<set, {len(value)} chars, {value[:4]}…>"
