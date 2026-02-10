"""
Data merging utilities for combining processed datasets.
"""
from __future__ import annotations

import pandas as pd
from typing import Literal


def merge_all_datasets(
    results_dict: dict[str, pd.DataFrame],
    how: Literal['inner', 'outer', 'left'] = 'outer',
    validate: str = 'one_to_one'
) -> pd.DataFrame:
    """
    Merge all processed datasets on (iso3, year) into one master DataFrame.
    
    Args:
        results_dict: Dictionary from process_all_datasets() with keys:
                     'cpds', 'population', 'gdppc', 'inflation', 'dependency'
        how: Type of merge ('inner', 'outer', 'left')
            - 'inner': Only keep rows with data in ALL datasets
            - 'outer': Keep all rows from all datasets (may have NaNs)
            - 'left': Keep all rows from first dataset
        validate: Pandas merge validation ('one_to_one', 'one_to_many', etc.)
        
    Returns:
        Master DataFrame with all variables merged on (iso3, year)
        
    Example:
        >>> results = process_all_datasets(REPO_ROOT)
        >>> master = merge_all_datasets(results, how='outer')
        >>> master.columns
        ['iso3', 'year', 'sstran', 'deficit', 'debt', 
         'ln_population', 'ln_gdppc', 'inflation_cpi', 'dependency_ratio']
    """
    # Filter out None values
    available = {k: v for k, v in results_dict.items() if v is not None}
    
    if not available:
        raise ValueError("No datasets available to merge (all are None)")
    
    # Start with the first available dataset
    dataset_order = ['cpds', 'population', 'gdppc', 'inflation', 'dependency']
    first_key = next(k for k in dataset_order if k in available)
    master = available[first_key].copy()
    
    print(f"Starting merge with {first_key}: {len(master)} rows")
    
    # Merge each subsequent dataset
    for name in dataset_order:
        if name in available and name != first_key:
            before_rows = len(master)
            master = master.merge(
                available[name],
                on=['iso3', 'year'],
                how=how,
                validate=validate
            )
            print(f"Merged {name}: {before_rows} → {len(master)} rows")
    
    # Sort by iso3 and year
    master = master.sort_values(['iso3', 'year']).reset_index(drop=True)
    
    print(f"\n✅ Final merged dataset: {len(master)} rows, {len(master.columns)} columns")
    print(f"   Countries: {master['iso3'].nunique()}")
    print(f"   Years: {master['year'].min()}-{master['year'].max()}")
    
    return master


def get_merge_summary(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of data coverage in the merged dataset.
    
    Args:
        master_df: Merged dataset from merge_all_datasets()
        
    Returns:
        DataFrame showing coverage statistics by country
    """
    summary = master_df.groupby('iso3').agg({
        'year': ['min', 'max', 'count'],
    })
    
    # Flatten column names
    summary.columns = ['year_min', 'year_max', 'n_years']
    
    # Add missing value counts for key variables
    key_vars = [c for c in master_df.columns if c not in ['iso3', 'year']]
    
    for var in key_vars:
        if var in master_df.columns:
            summary[f'{var}_missing'] = master_df.groupby('iso3')[var].apply(
                lambda x: x.isnull().sum()
            )
    
    return summary.sort_values('n_years', ascending=False)


def save_master_dataset(
    master_df: pd.DataFrame,
    output_path: str,
    formats: list[str] = ['parquet', 'csv']
) -> dict[str, str]:
    """
    Save the master merged dataset in multiple formats.
    
    Args:
        master_df: Merged dataset
        output_path: Base path without extension (e.g., 'data/final/master_dataset')
        formats: List of formats to save ('parquet', 'csv', 'stata')
        
    Returns:
        Dictionary mapping format to file path
    """
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    if 'parquet' in formats:
        path = output_path.with_suffix('.parquet')
        master_df.to_parquet(path, index=False)
        saved_files['parquet'] = str(path)
        print(f"✅ Saved: {path}")
    
    if 'csv' in formats:
        path = output_path.with_suffix('.csv')
        master_df.to_csv(path, index=False)
        saved_files['csv'] = str(path)
        print(f"✅ Saved: {path}")
    
    if 'stata' in formats:
        path = output_path.with_suffix('.dta')
        master_df.to_stata(path, write_index=False)
        saved_files['stata'] = str(path)
        print(f"✅ Saved: {path}")
    
    return saved_files
