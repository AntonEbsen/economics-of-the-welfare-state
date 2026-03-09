"""
Centralized constants for data cleaning.
Contains the canonical 32-country sample and country name mappings.
"""

# --- Target sample (the 32 OECD countries) ---
TARGET_ISO3_32 = {
    "AUS", "AUT", "BEL", "BGR", "CAN", "CZE", "DNK", "EST", "FIN", "FRA", 
    "DEU", "GRC", "HUN", "ISL", "IRL", "ITA", "JPN", "LVA", "LTU", "LUX", 
    "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE", "CHE", 
    "GBR", "USA"
}

# Country name to ISO3 mapping
# Used for datasets that provide country names instead of ISO codes
COUNTRY_TO_ISO3 = {
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bulgaria": "BGR",
    "Canada": "CAN",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Japan": "JPN",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "PRT",
    "Slovak Republic": "SVK",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "United Kingdom": "GBR",
    "UK": "GBR",
    "U.K.": "GBR",
    "United States": "USA",
    "United States of America": "USA",
    "USA": "USA",
}

# Default year range for processing
DEFAULT_YEAR_MIN = 1980
DEFAULT_YEAR_MAX = 2023

# --- Welfare Regime Mappings ---
# Based on Esping-Andersen (1990) and subsequent classifications
# Note: Some countries belong to multiple categories (e.g., Mediterranean is a subset of Conservative/other)
WELFARE_REGIME_MAP = {
    "Liberal": ["AUS", "CAN", "JPN", "IRL", "NZL", "GBR", "USA"],
    "Conservative": ["AUT", "BEL", "FRA", "DEU", "GRC", "ITA", "LUX", "NLD", "PRT", "ESP", "CHE"],
    "Social Democrat": ["DNK", "FIN", "NOR", "SWE"],
    "Mediterranean": ["GRC", "ITA", "PRT", "ESP"],
    "Post-Communist": ["BGR", "HRV", "CZE", "EST", "HUN", "LVA", "LTU", "POL", "ROU", "SVK", "SVN", "UKR"]
}
