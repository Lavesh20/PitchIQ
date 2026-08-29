"""Credentials must stay in .env and nowhere else.

A token committed to the tree is the one mistake here that cannot be
undone by editing a file, so this runs as a test rather than as advice.
"""

import re
import subprocess
from pathlib import Path

import pytest

from pitchiq import config

ROOT = config.ROOT

# GitHub's token formats, plus the generic fine-grained prefix.
TOKEN_PATTERNS = [
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
]

SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".pytest_cache"}


def _tracked_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == ".env":
            continue
        yield path


def test_env_file_is_gitignored():
    ignored = (ROOT / ".gitignore").read_text().splitlines()

    assert ".env" in [line.strip() for line in ignored]


def test_env_file_is_not_world_readable():
    if not config.ENV_FILE.exists():
        pytest.skip("no .env in this checkout")

    mode = config.ENV_FILE.stat().st_mode & 0o077

    assert mode == 0, f".env is group/other readable (mode {oct(mode)})"


def test_no_token_outside_env():
    """No GitHub token may appear in any file except .env."""
    offenders = []

    for path in _tracked_files():
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        for pattern in TOKEN_PATTERNS:
            if pattern.search(text):
                offenders.append(str(path.relative_to(ROOT)))
                break

    assert not offenders, f"GitHub token found outside .env: {offenders}"


def test_env_not_staged_if_repo_exists():
    """If this becomes a git repo, .env must never be tracked."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    # Non-zero means either "not a repo" or "not tracked"; both are fine.
    assert result.returncode != 0, ".env is tracked by git"
