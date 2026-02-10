"""
Visualization utilities for panel data.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_series(
    df: pd.DataFrame,
    variable: str,
    countries: list[str] = None,
    figsize: tuple = (12, 6),
    save_path: str = None
):
    """
    Plot variable over time for selected countries.
    
    Args:
        df: Panel DataFrame
        variable: Variable to plot
        countries: List of ISO3 codes. If None, plot all.
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Example:
        >>> plot_time_series(
        ...     master, 
        ...     'ln_gdppc',
        ...     countries=['USA', 'DEU', 'FRA', 'GBR']
        ... )
    """
    if countries is None:
        countries = df['iso3'].unique()[:10]  # Limit to 10 for readability
    
    plt.figure(figsize=figsize)
    
    for country in countries:
        data = df[df['iso3'] == country].sort_values('year')
        plt.plot(data['year'], data[variable], label=country, marker='o', markersize=3)
    
    plt.xlabel('Year')
    plt.ylabel(variable)
    plt.title(f'{variable} over time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    variables: list[str] = None,
    figsize: tuple = (10, 8),
    save_path: str = None
):
    """
    Plot correlation heatmap.
    
    Args:
        df: DataFrame
        variables: Variables to include
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    import numpy as np
    
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
        variables = [v for v in variables if v not in ['iso3', 'year']]
    
    corr = df[variables].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()


def plot_country_coverage(
    df: pd.DataFrame,
    variable: str = None,
    figsize: tuple = (14, 10),
    save_path: str = None
):
    """
    Visual heatmap of data availability by country and year.
    
    Args:
        df: Panel DataFrame
        variable: Specific variable to check. If None, checks all data.
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    # Create pivot table
    if variable:
        pivot = df.pivot_table(
            index='iso3',
            columns='year',
            values=variable,
            aggfunc='count'
        ).fillna(0)
    else:
        # Count non-null values across all columns
        pivot = df.groupby(['iso3', 'year']).size().unstack(fill_value=0)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap='YlGnBu',
        cbar_kws={'label': 'Data available'},
        linewidths=0.5
    )
    plt.title(f"Data Coverage: {variable if variable else 'All variables'}")
    plt.xlabel('Year')
    plt.ylabel('Country (ISO3)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()


def plot_distribution(
    df: pd.DataFrame,
    variable: str,
    by_group: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None
):
    """
    Plot distribution of a variable, optionally by group.
    
    Args:
        df: DataFrame
        variable: Variable to plot
        by_group: Group variable (e.g., 'iso3' for country comparison)
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=figsize)
    
    if by_group:
        for group in df[by_group].unique():
            data = df[df[by_group] == group][variable].dropna()
            plt.hist(data, alpha=0.5, label=group, bins=30)
        plt.legend()
    else:
        plt.hist(df[variable].dropna(), bins=50, edgecolor='black')
    
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {variable}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")
    
    plt.show()
