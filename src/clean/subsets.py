"""
Country grouping and subsetting utilities.
"""

from .constants import TARGET_ISO3_32

# Regional groupings
COUNTRY_GROUPS = {
    'western_europe': {
        'AUT', 'BEL', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 'IRL', 
        'ITA', 'LUX', 'NLD', 'NOR', 'PRT', 'ESP', 'SWE', 'CHE', 'GBR'
    },
    'eastern_europe': {
        'BGR', 'CZE', 'EST', 'HUN', 'LVA', 'LTU', 'POL', 'SVK', 'SVN'
    },
    'anglo': {
        'USA', 'GBR', 'CAN', 'AUS', 'NZL'
    },
    'nordic': {
        'DNK', 'FIN', 'ISL', 'NOR', 'SWE'
    },
    'eu_founders': {
        'BEL', 'FRA', 'DEU', 'ITA', 'LUX', 'NLD'
    },
    'post_communist': {
        'BGR', 'CZE', 'EST', 'HUN', 'LVA', 'LTU', 'POL', 'SVK', 'SVN'
    },
}


def filter_by_region(df, region: str):
    """
    Filter dataset to specific country group.
    
    Args:
        df: DataFrame with 'iso3' column
        region: Region name from COUNTRY_GROUPS
        
    Returns:
        Filtered DataFrame
        
    Example:
        >>> nordic_data = filter_by_region(master, 'nordic')
        >>> # Returns data for DNK, FIN, ISL, NOR, SWE
    """
    if region not in COUNTRY_GROUPS:
        raise ValueError(
            f"Unknown region: {region}. "
            f"Available: {list(COUNTRY_GROUPS.keys())}"
        )
    
    countries = COUNTRY_GROUPS[region]
    filtered = df[df['iso3'].isin(countries)].copy()
    
    print(f"✅ Filtered to {region}: {len(filtered)} rows, {filtered['iso3'].nunique()} countries")
    return filtered


def get_region_for_country(iso3: str) -> list[str]:
    """
    Find which region(s) a country belongs to.
    
    Args:
        iso3: ISO3 country code
        
    Returns:
        List of region names
        
    Example:
        >>> get_region_for_country('NOR')
        ['western_europe', 'nordic']
    """
    regions = []
    for region_name, countries in COUNTRY_GROUPS.items():
        if iso3 in countries:
            regions.append(region_name)
    return regions


def list_regions():
    """Print all available regions and their countries."""
    print("=" * 60)
    print("COUNTRY GROUPS")
    print("=" * 60)
    
    for region, countries in COUNTRY_GROUPS.items():
        print(f"\n{region.upper().replace('_', ' ')} ({len(countries)} countries):")
        print(f"  {', '.join(sorted(countries))}")
    
    print("\n" + "=" * 60)
