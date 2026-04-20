"""
Tests for ``src/clean/utils.py`` — shared helpers used across every
data-cleaning module.

These are pure-function / file-IO tests; no Excel fixtures needed.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from clean.constants import TARGET_ISO3_32
from clean.utils import (
    filter_to_target_countries,
    filter_to_year_range,
    load_config,
    map_country_to_iso3,
    save_dataframe,
    setup_logging,
)

# ---------------------------------------------------------------------------
# map_country_to_iso3
# ---------------------------------------------------------------------------


def test_map_country_to_iso3_maps_known_countries():
    df = pd.DataFrame({"country": ["Denmark", "Germany", "United States"]})
    result = map_country_to_iso3(df)
    assert "iso3" in result.columns
    assert list(result["iso3"]) == ["DNK", "DEU", "USA"]


def test_map_country_to_iso3_returns_nan_for_unknown():
    df = pd.DataFrame({"country": ["Atlantis"]})
    result = map_country_to_iso3(df)
    assert pd.isna(result["iso3"].iloc[0])


def test_map_country_to_iso3_preserves_original_columns():
    df = pd.DataFrame({"country": ["Denmark"], "value": [42]})
    result = map_country_to_iso3(df)
    assert "value" in result.columns
    assert result["value"].iloc[0] == 42


# ---------------------------------------------------------------------------
# save_dataframe
# ---------------------------------------------------------------------------


def test_save_dataframe_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = save_dataframe(df, tmp_path / "out.parquet")
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 2


def test_save_dataframe_csv(tmp_path):
    df = pd.DataFrame({"x": [10]})
    out = save_dataframe(df, tmp_path / "out.csv")
    assert out.exists()
    loaded = pd.read_csv(out)
    assert loaded["x"].iloc[0] == 10


def test_save_dataframe_creates_parent_dirs(tmp_path):
    df = pd.DataFrame({"a": [1]})
    out = save_dataframe(df, tmp_path / "nested" / "dir" / "out.parquet")
    assert out.exists()


def test_save_dataframe_rejects_unknown_extension(tmp_path):
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Output must end with"):
        save_dataframe(df, tmp_path / "out.xlsx")


# ---------------------------------------------------------------------------
# filter_to_target_countries
# ---------------------------------------------------------------------------


def test_filter_to_target_countries_keeps_only_targets():
    df = pd.DataFrame({"iso3": ["DNK", "ZZZ", "USA"], "v": [1, 2, 3]})
    result = filter_to_target_countries(df)
    assert set(result["iso3"]) <= TARGET_ISO3_32
    assert "ZZZ" not in result["iso3"].values


def test_filter_to_target_countries_drops_nan_iso3():
    df = pd.DataFrame({"iso3": [None, "DNK"], "v": [1, 2]})
    result = filter_to_target_countries(df)
    assert len(result) == 1
    assert result["iso3"].iloc[0] == "DNK"


def test_filter_to_target_countries_normalises_case():
    df = pd.DataFrame({"iso3": ["dnk", " usa "], "v": [1, 2]})
    result = filter_to_target_countries(df)
    assert set(result["iso3"]) == {"DNK", "USA"}


# ---------------------------------------------------------------------------
# filter_to_year_range
# ---------------------------------------------------------------------------


def test_filter_to_year_range_both_bounds():
    df = pd.DataFrame({"year": [1990, 2000, 2010, 2020]})
    result = filter_to_year_range(df, year_min=2000, year_max=2010)
    assert list(result["year"]) == [2000, 2010]


def test_filter_to_year_range_min_only():
    df = pd.DataFrame({"year": [1990, 2000, 2010]})
    result = filter_to_year_range(df, year_min=2000)
    assert list(result["year"]) == [2000, 2010]


def test_filter_to_year_range_max_only():
    df = pd.DataFrame({"year": [1990, 2000, 2010]})
    result = filter_to_year_range(df, year_max=2000)
    assert list(result["year"]) == [1990, 2000]


def test_filter_to_year_range_no_bounds_returns_all():
    df = pd.DataFrame({"year": [1990, 2000]})
    result = filter_to_year_range(df)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_from_default_path():
    cfg = load_config()
    # config.yaml must have at least 'controls' or 'indices'
    assert isinstance(cfg, dict)
    assert len(cfg) > 0


def test_load_config_from_explicit_path(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("key: value\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg == {"key": "value"}


def test_load_config_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


def test_setup_logging_creates_log_file(tmp_path):
    log_file = tmp_path / "test.log"
    setup_logging(log_file)
    logger = logging.getLogger("test_utils_logger")
    logger.info("hello")
    # Flush handlers
    for h in logging.getLogger().handlers:
        h.flush()
    assert log_file.exists()
    assert log_file.stat().st_size > 0
