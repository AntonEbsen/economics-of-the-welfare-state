"""
Variable metadata and data dictionary.
Provides documentation for all variables in the dataset.
"""

VARIABLE_METADATA = {
    # Identifiers
    "iso3": {
        "label": "ISO 3166-1 alpha-3 country code",
        "source": "ISO Standard",
        "unit": "Text",
        "description": "Three-letter country code",
    },
    "year": {
        "label": "Year",
        "source": "All datasets",
        "unit": "Year",
        "description": "Calendar year of observation",
    },
    # CPDS variables
    "sstran": {
        "label": "Social security transfers",
        "source": "CPDS",
        "unit": "% of GDP",
        "description": "Public social security transfers as percentage of GDP",
    },
    "deficit": {
        "label": "Government deficit",
        "source": "CPDS",
        "unit": "% of GDP",
        "description": "General government deficit as percentage of GDP. Negative values indicate surplus.",
    },
    "debt": {
        "label": "Government debt",
        "source": "CPDS",
        "unit": "% of GDP",
        "description": "Gross general government debt as percentage of GDP",
    },
    # Population
    "ln_population": {
        "label": "Log population",
        "source": "World Bank",
        "unit": "Natural log",
        "description": "Natural logarithm of total population",
    },
    # GDP
    "ln_gdppc": {
        "label": "Log GDP per capita",
        "source": "World Bank / OECD",
        "unit": "Natural log (constant USD)",
        "description": "Natural logarithm of GDP per capita in constant 2015 USD",
    },
    # Inflation
    "inflation_cpi": {
        "label": "Inflation rate",
        "source": "World Bank",
        "unit": "% annual change",
        "description": "Annual percentage change in consumer price index",
    },
    # Dependency ratio
    "dependency_ratio": {
        "label": "Age dependency ratio",
        "source": "World Bank",
        "unit": "% of working-age population",
        "description": "Ratio of dependents (people younger than 15 or older than 64) to working-age population (ages 15-64)",
    },
}


def get_variable_info(var_name: str) -> dict:
    """
    Get metadata for a specific variable.

    Args:
        var_name: Variable name

    Returns:
        Dictionary with metadata or error message

    Example:
        >>> info = get_variable_info('sstran')
        >>> print(info['description'])
        'Public social security transfers as percentage of GDP'
    """
    return VARIABLE_METADATA.get(
        var_name, {"error": f"No metadata available for variable: {var_name}"}
    )


def print_codebook(variables=None):
    """
    Print a formatted codebook for specified variables.

    Args:
        variables: List of variable names. If None, print all.

    Example:
        >>> print_codebook(['sstran', 'deficit', 'debt'])
    """
    if variables is None:
        variables = VARIABLE_METADATA.keys()

    print("=" * 80)
    print("DATA CODEBOOK")
    print("=" * 80)

    for var in variables:
        info = get_variable_info(var)
        if "error" not in info:
            print(f"\n{var.upper()}")
            print("-" * 40)
            print(f"Label:       {info['label']}")
            print(f"Source:      {info['source']}")
            print(f"Unit:        {info['unit']}")
            print(f"Description: {info['description']}")

    print("\n" + "=" * 80)


def export_codebook_to_csv(output_path: str, variables=None):
    """
    Export codebook to CSV file.

    Args:
        output_path: Path to output CSV file
        variables: List of variable names. If None, export all.
    """
    from pathlib import Path

    import pandas as pd

    if variables is None:
        variables = list(VARIABLE_METADATA.keys())

    # Create rows
    rows = []
    for var in variables:
        info = get_variable_info(var)
        if "error" not in info:
            rows.append(
                {
                    "Variable": var,
                    "Label": info["label"],
                    "Source": info["source"],
                    "Unit": info["unit"],
                    "Description": info["description"],
                }
            )

    df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Codebook exported to: {output_path}")
    return df
