"""
Unit tests for constants module.
Run with: pytest tests/
"""

from clean.constants import COUNTRY_TO_ISO3, DEFAULT_YEAR_MAX, DEFAULT_YEAR_MIN, TARGET_ISO3_32


def test_target_iso3_count():
    """Test that we have exactly 32 countries."""
    assert len(TARGET_ISO3_32) == 32, f"Expected 32 countries, got {len(TARGET_ISO3_32)}"


def test_target_iso3_format():
    """Test that all ISO3 codes are properly formatted."""
    for code in TARGET_ISO3_32:
        assert len(code) == 3, f"ISO3 code {code} should be 3 characters"
        assert code.isupper(), f"ISO3 code {code} should be uppercase"
        assert code.isalpha(), f"ISO3 code {code} should only contain letters"


def test_country_mapping_coverage():
    """Test that country mapping covers all target countries."""
    mapped_iso3 = set(COUNTRY_TO_ISO3.values())
    assert TARGET_ISO3_32.issubset(
        mapped_iso3
    ), "Not all TARGET_ISO3_32 countries are in COUNTRY_TO_ISO3"


def test_no_duplicate_mappings():
    """Test that there are no duplicate country mappings."""
    # Each country name should map to only one ISO3
    country_names = list(COUNTRY_TO_ISO3.keys())
    assert len(country_names) == len(
        set(country_names)
    ), "Duplicate country names found in COUNTRY_TO_ISO3"


def test_year_range_valid():
    """Test that default year range is valid."""
    assert (
        DEFAULT_YEAR_MIN < DEFAULT_YEAR_MAX
    ), "DEFAULT_YEAR_MIN should be less than DEFAULT_YEAR_MAX"
    assert 1900 < DEFAULT_YEAR_MIN < 2100, "DEFAULT_YEAR_MIN should be reasonable"
    assert 1900 < DEFAULT_YEAR_MAX < 2100, "DEFAULT_YEAR_MAX should be reasonable"
