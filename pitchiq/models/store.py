"""Save and load trained models.

Until this existed every script refitted from scratch, which was cheap
enough to hide three real problems. A forecast could not be traced back
to the model that produced it; nothing outside a Python session could
use a model; and tuning paid the fit cost on every iteration.

A trained Dixon-Coles model is 2,862 numbers, so it stores as readable
JSON rather than a pickle. That matters more than the few kilobytes it
costs: a pickle is opaque, is tied to the class definition that wrote
it, and will happily execute code on load.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import DATA
from .dixon_coles import DixonColesConfig, DixonColesResult

MODELS = DATA / "models"

FORMAT_VERSION = 1


def save(
    model: DixonColesResult,
    path: Path | None = None,
    matches: pd.DataFrame | None = None,
    note: str = "",
) -> Path:
    """Write a fitted model to JSON, with the provenance to reproduce it."""
    path = path or MODELS / "dixon_coles.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": FORMAT_VERSION,
        "model": "dixon_coles",
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "config": asdict(model.config),
        "converged": model.converged,
        "log_likelihood": model.log_likelihood,
        "home_advantage": model.home_advantage,
        "home_advantages": model.home_advantages,
        "rho": model.rho,
        "attack": model.attack,
        "defence": model.defence,
    }

    if matches is not None and len(matches):
        payload["trained_on"] = {
            "matches": int(len(matches)),
            "clubs": int(len(set(matches.home_key) | set(matches.away_key))),
            "first": str(matches.date.min().date()),
            "last": str(matches.date.max().date()),
        }

    path.write_text(json.dumps(payload, indent=2, sort_keys=False))

    return path


def load(path: Path | None = None) -> DixonColesResult:
    """Read a model back. Raises if the file was written by a newer format."""
    path = path or MODELS / "dixon_coles.json"

    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run scripts/train.py first."
        )

    payload = json.loads(path.read_text())

    version = payload.get("format_version", 0)
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{path} was written in format {version}; this code reads {FORMAT_VERSION}."
        )

    return DixonColesResult(
        attack=payload["attack"],
        defence=payload["defence"],
        home_advantage=payload["home_advantage"],
        home_advantages=payload.get("home_advantages", {}),
        rho=payload["rho"],
        config=DixonColesConfig(**payload["config"]),
        converged=payload["converged"],
        log_likelihood=payload["log_likelihood"],
    )


def describe(path: Path | None = None) -> dict:
    """Provenance of a saved model, without loading the ratings."""
    path = path or MODELS / "dixon_coles.json"
    payload = json.loads(path.read_text())

    return {
        k: payload[k]
        for k in ("saved_at", "note", "config", "converged", "trained_on")
        if k in payload
    }
