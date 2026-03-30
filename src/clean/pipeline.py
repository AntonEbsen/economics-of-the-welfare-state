# Master data processing pipeline.
# v0.3.1 - Robust Validation Update
"""
Master data processing pipeline.
Processes all datasets with a single function call.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .constants import DEFAULT_YEAR_MAX, DEFAULT_YEAR_MIN
from .validation import validate_output

logger = logging.getLogger(__name__)


def process_all_datasets(
    repo_root: Path,
    year_min: int = DEFAULT_YEAR_MIN,
    year_max: int = DEFAULT_YEAR_MAX,
    save_outputs: bool = True,
    validate: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Process all datasets (CPDS, Population, GDP, Inflation, Dependency, KOF) in one call.

    Args:
        repo_root: Path to repository root
        year_min: Minimum year to include
        year_max: Maximum year to include
        save_outputs: If True, save processed data to parquet and CSV
        validate: If True, validate outputs before returning

    Returns:
        Dictionary with keys: 'cpds', 'population', 'gdppc', 'inflation', 'dependency', 'kof'
        Each value is a processed DataFrame
    """
    results = {}
    repo_root = Path(repo_root)
    raw_path = repo_root / "data" / "raw"
    processed_path = repo_root / "data" / "processed"

    logger.info("=" * 60)
    logger.info("🔄 Processing all datasets...")
    logger.info("=" * 60)

    # 1. CPDS
    logger.info("\n📊 Processing CPDS...")
    try:
        from .cpds import filter_cpds_32countries, read_cpds_excel, save_cpds, standardize_cpds

        df_cpds_raw = read_cpds_excel(raw_path / "cpds_raw.xlsx")
        df_cpds_std = standardize_cpds(df_cpds_raw)
        df_cpds = filter_cpds_32countries(df_cpds_std, year_min=year_min, year_max=year_max)

        if validate:
            validate_output(
                df_cpds,
                required_cols=["iso3", "year", "sstran", "deficit", "debt"],
                dataset_name="CPDS",
                year_min=year_min,
                year_max=year_max,
                expect_32_countries=False,  # Not all countries have all years
            )

        if save_outputs:
            save_cpds(df_cpds, processed_path / f"cpds_32countries_{year_min}_{year_max}.parquet")
            save_cpds(df_cpds, processed_path / f"cpds_32countries_{year_min}_{year_max}.csv")

        results["cpds"] = df_cpds
        logger.info(f"✅ CPDS: {len(df_cpds)} rows")
    except Exception as e:
        logger.error(f"❌ CPDS failed: {e}")
        results["cpds"] = None

    # 2. Population
    logger.info("\n👥 Processing Population...")
    try:
        from .population import filter_32_and_log as filter_pop_32
        from .population import read_population_excel, standardize_worldbank_population_to_long
        from .population import save_processed as save_population

        df_pop_raw = read_population_excel(raw_path / "Population_raw.xlsx")
        df_pop_long = standardize_worldbank_population_to_long(df_pop_raw)
        df_pop = filter_pop_32(df_pop_long)  # Uses default year range

        if validate:
            validate_output(
                df_pop,
                required_cols=["iso3", "year", "ln_population"],
                dataset_name="Population",
                year_min=year_min,
                year_max=year_max,
                expect_32_countries=False,
            )

        if save_outputs:
            save_population(
                df_pop, processed_path / f"population_32countries_{year_min}_{year_max}.parquet"
            )
            save_population(
                df_pop, processed_path / f"population_32countries_{year_min}_{year_max}.csv"
            )

        results["population"] = df_pop
        logger.info(f"✅ Population: {len(df_pop)} rows")
    except Exception as e:
        logger.error(f"❌ Population failed: {e}")
        results["population"] = None

    # 3. GDP per capita
    logger.info("\n💰 Processing GDP per capita...")
    try:
        from .gdppc import GDPPCConfig, get_final_gdppc, read_gdppc_excel, standardize_gdppc_to_long
        from .gdppc import map_country_to_iso3 as map_gdp_to_iso3
        from .gdppc import save_processed as save_gdppc

        cfg = GDPPCConfig(year_min=year_min, year_max=year_max, country_col="Reference area")
        df_gdp_raw = read_gdppc_excel(raw_path / "GDP_per_capita.xlsx")
        df_gdp_long = standardize_gdppc_to_long(df_gdp_raw, cfg=cfg)
        df_gdp_mapped = map_gdp_to_iso3(df_gdp_long)
        df_gdp = get_final_gdppc(df_gdp_mapped, cfg=cfg)

        if validate:
            validate_output(
                df_gdp,
                required_cols=["iso3", "year", "ln_gdppc"],
                dataset_name="GDP per capita",
                year_min=year_min,
                year_max=year_max,
                expect_32_countries=False,
            )

        if save_outputs:
            save_gdppc(df_gdp, processed_path / f"gdppc_32countries_{year_min}_{year_max}.parquet")
            save_gdppc(df_gdp, processed_path / f"gdppc_32countries_{year_min}_{year_max}.csv")

        results["gdppc"] = df_gdp
        logger.info(f"✅ GDP per capita: {len(df_gdp)} rows")
    except Exception as e:
        logger.error(f"❌ GDP per capita failed: {e}")
        results["gdppc"] = None

    # 4. Inflation CPI
    logger.info("\n📈 Processing Inflation CPI...")
    try:
        from .inflation import filter_32_countries as filter_inflation_32
        from .inflation import read_inflation_excel, save_inflation, standardize_inflation_to_long
        from .utils import map_country_to_iso3 as map_inflation_to_iso3

        df_inf_raw = read_inflation_excel(raw_path / "Inflation_cpi.xlsx")
        df_inf_long = standardize_inflation_to_long(df_inf_raw)
        df_inf_mapped = map_inflation_to_iso3(df_inf_long)
        df_inf = filter_inflation_32(df_inf_mapped, year_min=year_min, year_max=year_max)

        if validate:
            validate_output(
                df_inf,
                required_cols=["iso3", "year", "inflation_cpi"],
                dataset_name="Inflation CPI",
                year_min=year_min,
                year_max=year_max,
                expect_32_countries=False,
            )

        if save_outputs:
            save_inflation(
                df_inf, processed_path / f"inflation_32countries_{year_min}_{year_max}.parquet"
            )
            save_inflation(
                df_inf, processed_path / f"inflation_32countries_{year_min}_{year_max}.csv"
            )

        results["inflation"] = df_inf
        logger.info(f"✅ Inflation CPI: {len(df_inf)} rows")
    except Exception as e:
        logger.error(f"❌ Inflation CPI failed: {e}")
        results["inflation"] = None

    # 5. Dependency Ratio
    logger.info("\n👶👴 Processing Dependency Ratio...")
    try:
        from .dependency_ratio import filter_32_countries as filter_dep_32
        from .dependency_ratio import (
            read_dependency_excel,
            save_dependency,
            standardize_dependency_to_long,
        )
        from .utils import map_country_to_iso3 as map_dep_to_iso3

        df_dep_raw = read_dependency_excel(raw_path / "Dependency_ratio.xlsx")
        df_dep_long = standardize_dependency_to_long(df_dep_raw)
        df_dep_mapped = map_dep_to_iso3(df_dep_long)
        df_dep = filter_dep_32(df_dep_mapped, year_min=year_min, year_max=year_max)

        if validate:
            validate_output(
                df_dep,
                required_cols=["iso3", "year", "dependency_ratio"],
                dataset_name="Dependency Ratio",
                year_min=year_min,
                year_max=year_max,
                expect_32_countries=False,
            )

        if save_outputs:
            save_dependency(
                df_dep,
                processed_path / f"dependency_ratio_32countries_{year_min}_{year_max}.parquet",
            )
            save_dependency(
                df_dep, processed_path / f"dependency_ratio_32countries_{year_min}_{year_max}.csv"
            )

        results["dependency"] = df_dep
        logger.info(f"✅ Dependency Ratio: {len(df_dep)} rows")
    except Exception as e:
        logger.error(f"❌ Dependency Ratio failed: {e}")
        results["dependency"] = None

    # 6. KOF Globalization Index
    logger.info("\n🌍 Processing KOF Globalization Index...")
    try:
        from .kofgi import KOFConfig, filter_kof_32countries, read_kof_excel, standardize_kof
        from .kofgi import save_processed as save_kof

        kof_cfg = KOFConfig(year_min=year_min, year_max=year_max)
        df_kof_raw = read_kof_excel(raw_path / "KOF_index_raw.xlsx")
        df_kof_std = standardize_kof(df_kof_raw)
        df_kof = filter_kof_32countries(df_kof_std, cfg=kof_cfg)

        if validate:
            validate_output(
                df_kof,
                required_cols=["iso3", "year", "KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"],
                dataset_name="KOF Globalization Index",
                year_min=year_min,
                year_max=year_max,
                expect_32_countries=False,
            )

        if save_outputs:
            save_kof(df_kof, processed_path / f"kofgi_32countries_{year_min}_{year_max}.parquet")
            save_kof(df_kof, processed_path / f"kofgi_32countries_{year_min}_{year_max}.csv")

        results["kof"] = df_kof
        logger.info(f"✅ KOF Index: {len(df_kof)} rows")
    except Exception as e:
        logger.error(f"❌ KOF failed: {e}")
        results["kof"] = None

    # 7. Diagnostics
    logger.info("\n🔍 Running research diagnostics...")
    try:
        from .utils import load_config

        config = load_config()
        ["sstran"] + config.get("controls", [])

        # We need a master dataframe to run diagnostics on.
        # Since this function processes individual datasets, we return the dict.
        # But for convenience, let's suggest running diagnostics after merging.
        logger.info("Tip: Run generate_diagnostic_report(master_df, variables) after merging.")
    except Exception as e:
        logger.warning(f"Diagnostics skipped: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    successful = sum(1 for v in results.values() if v is not None)
    logger.info(f"✅ Completed: {successful}/6 datasets processed successfully")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    from .utils import setup_logging

    parser = argparse.ArgumentParser(description="Run the welfare state data cleaning pipeline.")
    parser.add_argument(
        "--year-min", type=int, default=DEFAULT_YEAR_MIN, help="Minimum year to include."
    )
    parser.add_argument(
        "--year-max", type=int, default=DEFAULT_YEAR_MAX, help="Maximum year to include."
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save outputs to disk.")
    parser.add_argument("--repo-root", type=str, default=".", help="Path to repository root.")

    args = parser.parse_args()

    # Initialize robust logging to pipeline.log
    setup_logging("pipeline.log")
    logger.info(f"Starting pipeline with year range {args.year_min}-{args.year_max}")

    process_all_datasets(
        repo_root=Path(args.repo_root),
        year_min=args.year_min,
        year_max=args.year_max,
        save_outputs=not args.no_save,
        validate=True,
    )
