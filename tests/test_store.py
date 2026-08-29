"""Saving and loading trained models."""

import json

import numpy as np
import pandas as pd
import pytest

from pitchiq.models import dixon_coles as dc
from pitchiq.models import store


@pytest.fixture
def model():
    rng = np.random.default_rng(0)
    clubs = ["a", "b", "c", "d"]

    rows = [
        {
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=k),
            "home_key": clubs[i], "away_key": clubs[j],
            "fthg": int(rng.poisson(1.5)), "ftag": int(rng.poisson(1.1)),
        }
        for k, (i, j) in enumerate(
            (int(x), int(y))
            for x, y in rng.choice(4, size=(200, 2))
            if x != y
        )
    ]

    return dc.fit(pd.DataFrame(rows), dc.DixonColesConfig(xi=0.002))


def test_round_trip_preserves_every_parameter(model, tmp_path):
    path = store.save(model, tmp_path / "m.json")
    loaded = store.load(path)

    assert loaded.attack == pytest.approx(model.attack)
    assert loaded.defence == pytest.approx(model.defence)
    assert loaded.rho == pytest.approx(model.rho)
    assert loaded.home_advantage == pytest.approx(model.home_advantage)
    assert loaded.config == model.config


def test_round_trip_preserves_predictions(model, tmp_path):
    """The point of saving is that the loaded model forecasts identically."""
    loaded = store.load(store.save(model, tmp_path / "m.json"))

    before = model.predict("a", "b")
    after = loaded.predict("a", "b")

    for outcome in ("H", "D", "A"):
        assert after[outcome] == pytest.approx(before[outcome])


def test_saved_file_is_readable_json(model, tmp_path):
    """Not a pickle: a pickle is opaque and executes code on load."""
    path = store.save(model, tmp_path / "m.json")
    payload = json.loads(path.read_text())

    assert payload["model"] == "dixon_coles"
    assert "attack" in payload and "defence" in payload


def test_provenance_is_recorded(model, tmp_path):
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2024-06-01"]),
        "home_key": ["a", "b"], "away_key": ["b", "a"],
    })
    path = store.save(model, tmp_path / "m.json", matches=frame, note="hello")

    described = store.describe(path)

    assert described["note"] == "hello"
    assert described["trained_on"]["matches"] == 2
    assert described["trained_on"]["first"] == "2020-01-01"
    assert described["trained_on"]["last"] == "2024-06-01"


def test_missing_file_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        store.load(tmp_path / "absent.json")


def test_newer_format_is_refused(model, tmp_path):
    path = store.save(model, tmp_path / "m.json")

    payload = json.loads(path.read_text())
    payload["format_version"] = store.FORMAT_VERSION + 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="format"):
        store.load(path)
