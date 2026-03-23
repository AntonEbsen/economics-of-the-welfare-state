
# --- Cell 4 ---
from pathlib import Path
import sys

# Setup paths
REPO_ROOT = Path.cwd().resolve().parent
SRC_PATH = REPO_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

print(f"Repository root: {REPO_ROOT}")

# --- Cell 8 ---
from clean import process_all_datasets, merge_all_datasets

# Process ALL datasets with one command!
results = process_all_datasets(
    repo_root=REPO_ROOT,
    year_min=1980,
    year_max=2023,
    validate=True,
    save_outputs=True
)

print(f"\n Processed {len(results)} datasets!")
for name, df in results.items():
    if df is not None:
        print(f"   {name}: {len(df):,} observations")

# --- Cell 9 ---
# Merge into master dataset
master = merge_all_datasets(results, how='outer')

print(f"\n Master Dataset:")
print(f"   {len(master):,} observations")
print(f"   {master['iso3'].nunique()} countries")
print(f"   {master['year'].min()}-{master['year'].max()}")
print(f"   {len(master.columns)} variables")
master.head()

# --- Cell 11 ---
from clean import save_master_dataset

# Save the master dataset
saved_paths = save_master_dataset(
    master, 
    output_path=REPO_ROOT / "data" / "final" / "master_dataset",
    formats=['parquet', 'csv', 'stata']
)

# --- Cell 14 ---
from clean import generate_quality_report

# Generate comprehensive quality report
quality_report = generate_quality_report(
    master,
    output_path=REPO_ROOT / "reports" / "quality_report.html"
)

# --- Cell 17 ---
from clean import generate_summary_stats

# Generate summary statistics
stats = generate_summary_stats(master)
stats

# --- Cell 18 ---
import os

# Define the root directory (assuming the notebook is in a subfolder like 'notebooks')
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))  # Moves up one level from the notebooks folder
output_dir = os.path.join(root_dir, 'outputs', 'tables')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate the LaTeX table (replace this with your actual function)
latex_stats = generate_summary_stats(master, output_format='latex')

# Define the output file path
output_file = os.path.join(output_dir, 'summary_stats.tex')

# Write the LaTeX table to the file
with open(output_file, 'w') as f:
    f.write(latex_stats)

print(f"LaTeX table saved to: {output_file}")
print("\nLaTeX Table (copy to paper):")
print(latex_stats)


# --- Cell 21 ---
from clean import test_stationarity

# Test which variables are stationary
stationarity_results = test_stationarity(
    master,
    variables=['ln_gdppc', 'sstran', 'deficit', 'debt', 'inflation_cpi', 'ln_population', 'dependency_ratio', 'KOFGI', 'KOFEcGI', 'KOFSoGI', 'KOFPoGI'],
    test='adf'  # Augmented Dickey-Fuller test
)

stationarity_results

# --- Cell 23 ---
from clean import create_lags, check_panel_balance

# Check if panel is balanced
balance = check_panel_balance(master)
print(f"Panel balanced: {balance['balanced']}")

# Create lags for regression
master_with_lags = create_lags(
    master,
    variables=['ln_gdppc', 'deficit'],
    lags=[1, 2, 3]
)

print(f"\n✅ Added lag variables:")
lag_cols = [c for c in master_with_lags.columns if '_lag' in c]
print(lag_cols)

# --- Cell 26 ---
from clean.panel_utils import add_welfare_regimes

# ── Apply categorization ─────────────────────────────────────────────
master_regimes = add_welfare_regimes(master, id_var="iso3")

# ── Data Check ──────────────────────────────────────────────────────
print("\nRegime Counts (Observations):")
print(master_regimes["welfare_regime"].value_counts())

print("\nSample mapping (First few rows):")
display(master_regimes[["iso3", "welfare_regime"]].drop_duplicates().head(10))

# Verify dummy indicators
dummy_cols = [c for c in master_regimes.columns if c.startswith("regime_")]
print("\nIndicator column counts:")
print(master_regimes[dummy_cols].sum())


# --- Cell 28 ---
from clean import plot_time_series, plot_correlation_matrix
import matplotlib.pyplot as plt

# Plot GDP over time for selected countries
plot_time_series(
    master,
    'ln_gdppc',
    countries=['USA', 'DEU', 'GBR', 'FRA', 'SWE', 'DNK']
)

# --- Cell 29 ---
# Correlation heatmap
plot_correlation_matrix(
    master,
    variables=['ln_gdppc', 'sstran', 'deficit', 'debt', 'inflation_cpi']
)

# --- Cell 31 ---
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Compute cross-country mean per year ─────────────────────────────────────────
agg = master.dropna(subset=["sstran"]).copy()

# Ensure numeric types
agg["year"] = pd.to_numeric(agg["year"], errors="coerce")
agg["sstran"] = pd.to_numeric(agg["sstran"], errors="coerce")
agg = agg.dropna(subset=["year", "sstran"])

# Group and calculate mean
agg = (
    agg.groupby("year")["sstran"]
    .agg(mean="mean", n="count")
    .reset_index()
)

# Extract as raw float arrays for matplotlib
x_year = np.array(agg["year"], dtype=float)
y_mean = np.array(agg["mean"], dtype=float)

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

# Mean line
ax.plot(
    x_year, y_mean,
    color="#2563EB", linewidth=2.5,
    marker="o", markersize=3.5, markeredgewidth=0,
    label="Cross-country mean"
)

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Social Security Transfers (% GDP)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.legend(framealpha=0.9, loc="upper left", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.45)
ax.set_xlim(x_year.min(), x_year.max())

# Annotate country count
n_min, n_max = int(agg["n"].min()), int(agg["n"].max())
ax.annotate(
    f"N = {n_min}–{n_max} countries per year",
    xy=(0.02, 0.04), xycoords="axes fraction",
    fontsize=9, color="grey"
)

plt.tight_layout()

# ── Save export ──────────────────────────────────────────────────────
from pathlib import Path

# Create outputs/figures director if it doesn't exist
out_dir = Path(REPO_ROOT) / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# Save high-res for publications
out_path_png = out_dir / "sstran_average.png"
out_path_pdf = out_dir / "sstran_average.pdf"

fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
fig.savefig(out_path_pdf, bbox_inches="tight")

print(f"\n✅ Saved figure to: {out_path_png}")

plt.show()

# Quick numeric summary
print(f"\nPeriod average: {agg['mean'].mean():.2f}% of GDP")
print(f"Peak year:      {int(agg.loc[agg['mean'].idxmax(), 'year'])} ({agg['mean'].max():.2f}%)")
print(f"Trough year:    {int(agg.loc[agg['mean'].idxmin(), 'year'])} ({agg['mean'].min():.2f}%)")

# --- Cell 33 ---
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from pathlib import Path

# ── Compute cross-country mean for KOF indices ───────────────────────
indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
labels = {
    "KOFGI": "Total Globalization",
    "KOFEcGI": "Economic Globalization",
    "KOFSoGI": "Social Globalization",
    "KOFPoGI": "Political Globalization"
}
colors = {
    "KOFGI": "#E11D48",    # Red
    "KOFEcGI": "#2563EB",  # Blue
    "KOFSoGI": "#10B981",  # Green
    "KOFPoGI": "#F59E0B"   # Orange
}

# Ensure numeric types and drop NaNs
kof_data = master[["year", "iso3"] + indices].copy()
for col in ["year"] + indices:
    kof_data[col] = pd.to_numeric(kof_data[col], errors="coerce")

# Calculate yearly mean
agg_kof = (
    kof_data.dropna(subset=indices, how="all")
    .groupby("year")[indices]
    .mean()
    .reset_index()
)

# Extract x_year as raw float array
x_year = np.array(agg_kof["year"], dtype=float)

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

for idx in indices:
    y_values = np.array(agg_kof[idx], dtype=float)
    ax.plot(
        x_year, y_values,
        label=labels[idx], color=colors[idx],
        linewidth=2.5, marker="o", markersize=3, markeredgewidth=0
    )

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("KOF Indices of Globalization", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
ax.legend(framealpha=0.9, loc="upper left", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.45)
ax.set_xlim(x_year.min(), x_year.max())
ax.set_ylim(50, 90) # KOF indices are 0-100

plt.tight_layout()

# ── Save export ──────────────────────────────────────────────────────
out_dir = Path(REPO_ROOT) / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

out_path_png = out_dir / "kof_indices_average.png"
out_path_pdf = out_dir / "kof_indices_average.pdf"

fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
fig.savefig(out_path_pdf, bbox_inches="tight")

print(f"\n✅ Saved figure to: {out_path_png}")

plt.show()

# Quick summary
print(f"\nOverall Globalization (KOFGI) trend: {agg_kof['KOFGI'].iloc[0]:.1f} (1980) \u2192 {agg_kof['KOFGI'].iloc[-1]:.1f} ({int(x_year[-1])})")

# --- Cell 35 ---
from clean import filter_by_region, list_regions

# See available regions
list_regions()

# --- Cell 38 ---
from clean.robustness import run_robustness_checks, compare_robustness_results
from statsmodels.formula.api import ols

# Run model with automated robustness checks
robust_results = run_robustness_checks(
    master_with_lags,
    'sstran ~ ln_gdppc + deficit + ln_gdppc_lag1',
    ols,
    checks=['drop_outliers', 'winsorize', 'pre_2008', 'post_2008']
)

# --- Cell 39 ---
# Compare coefficients across specifications
comparison = compare_robustness_results(robust_results, variable='ln_gdppc')

# --- Cell 41 ---
from clean import create_publication_table

# Create publication-ready table
pub_table = create_publication_table(
    list(robust_results.values()),
    model_names=['Baseline', 'No Outliers', 'Winsorized', 'Pre-2008', 'Post-2008'],
    output_format='text'
)

print(pub_table)

# --- Cell 43 ---
from clean import generate_methods_section

# Auto-generate data section for paper
methods = generate_methods_section(
    master,
    output_path=REPO_ROOT / "paper" / "methods.md"
)

print("Methods section generated! Preview:")
print(methods[:500] + "...")

# --- Cell 46 ---
from clean.export import export_to_r, export_to_stata_script

# Generate R script with pre-configured models
export_to_r(
    master_with_lags,
    REPO_ROOT / 'analysis' / 'analysis.R',
    include_packages=['fixest', 'plm', 'did']
)

# Generate Stata script
export_to_stata_script(
    master_with_lags,
    REPO_ROOT / 'analysis' / 'analysis.do'
)

# --- Cell 48 ---
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ── Prepare variables with lags ──────────────────────────────────────
# Sort by country and year to properly compute lags
df_corr = master.sort_values(["iso3", "year"]).copy()

# Variables to lag (all except the main dependent variable)
lag_vars = {
    "KOFGI": "OG (t-1)",
    "KOFEcGI": "EG (t-1)",
    "KOFPoGI": "PG (t-1)",
    "KOFSoGI": "SG (t-1)",
    "ln_gdppc": "GDPpc (t-1)",
    "inflation_cpi": "Inf. (t-1)",
    "deficit": "Deficit (t-1)",
    "debt": "Gov. debt (t-1)",
    "ln_population": "Log pop. (t-1)",
    "dependency_ratio": "Dep. (t-1)"
}

# Apply 1-year lag within each country group
for col, new_name in lag_vars.items():
    if col in df_corr.columns:
        df_corr[new_name] = df_corr.groupby("iso3")[col].shift(1)

# Main dependent variable
df_corr["WS"] = df_corr["sstran"]

# Order of columns for the table per request
col_order = ["WS"] + list(lag_vars.values())

# Keep only necessary columns
corr_data = df_corr[col_order]

# ── Calculate Correlation and P-values ───────────────────────────────
def calculate_pvalues(df):
    df_cols = pd.DataFrame(columns=df.columns)
    pvalues = df_cols.transpose().join(df_cols, how="outer")
    for r in df.columns:
        for c in df.columns:
            if r == c:
                pvalues.loc[r, c] = 0.0
            else:
                # Drop pairwise NaNs to compute valid scipy pearsonr
                mask = df[r].notna() & df[c].notna()
                if mask.sum() > 2: # need at least 3 points
                    pvalues.loc[r, c] = stats.pearsonr(df[r][mask], df[c][mask])[1]
                else:
                    pvalues.loc[r, c] = np.nan
    return pvalues

# Compute pearson correlation matrix
corr_matrix = corr_data.corr(method="pearson")
# Compute significance matrix
pval_matrix = calculate_pvalues(corr_data)

# Format cells with significance stars (*** 1%, ** 5%, * 10%)
def format_with_stars(val, pval):
    if pd.isna(val) or pd.isna(pval):
        return ""
    
    val_str = f"{val:.2f}"
    if pval < 0.01:
        return val_str + "***"
    elif pval < 0.05:
        return val_str + "**"
    elif pval < 0.10:
        return val_str + "*"
    else:
        return val_str

# Create formatted lower-triangle table
formatted_table = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
for r in corr_matrix.index:
    for c in corr_matrix.columns:
        idx_r = list(corr_matrix.index).index(r)
        idx_c = list(corr_matrix.columns).index(c)
        
        if idx_r == idx_c:
            formatted_table.loc[r, c] = "1.00"
        elif idx_c > idx_r:
            formatted_table.loc[r, c] = "" # Upper triangle
        else:
            formatted_table.loc[r, c] = format_with_stars(corr_matrix.loc[r, c], pval_matrix.loc[r, c])

# ── Save and Display ──────────────────────────────────────────────────
out_dir = Path(REPO_ROOT) / "outputs" / "tables"
out_dir.mkdir(parents=True, exist_ok=True)

# Save to CSV
out_path_csv = out_dir / "correlation_matrix.csv"
formatted_table.to_csv(out_path_csv)
print(f"✅ Correlation matrix saved to CSV: {out_path_csv}")

# Save to LaTeX
out_path_tex = out_dir / "correlation_matrix.tex"
with open(out_path_tex, 'w', encoding='utf-8') as f:
    f.write(formatted_table.to_latex(
        caption="Correlation Matrix",
        label="tab:correlation_matrix",
        column_format="l" + "c" * len(formatted_table.columns),
        position="htbp"
    ))
print(f"✅ Correlation matrix saved to LaTeX: {out_path_tex}\n")

print("Table 8: Correlation matrix")
display(formatted_table)

print("\nNotes: OG: Overall Globalization, EG: Economic, PG: Political, SG: Social")
print("***, **, * denote statistical significance at 1, 5 and 10%, respectively")


# --- Cell 51 ---
# ── Baseline Regressions (No Interactions) ───────────────────────────
from linearmodels.panel import compare
from analysis.regression_utils import prepare_regression_data, run_panel_ols, LATEX_LABEL_MAP

indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
baseline_results = {}

for idx_name in indices:
    current_ctrl_vars = [idx_name, "ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"]
    reg_data = create_lags(master_regimes, current_ctrl_vars, lags=[1])
    
    g_var = f"{idx_name}_lag1"
    lagged_ctrls = [f"{v}_lag1" for v in current_ctrl_vars if v != idx_name]
    
    # Prepare data WITHOUT interactions
    ols_data, exog_vars = prepare_regression_data(reg_data, "sstran", g_var, lagged_ctrls, interactions=False)
    baseline_results[idx_name] = run_panel_ols(ols_data, "sstran", exog_vars)

# Compare models
baseline_comparison = compare(baseline_results, stars=True)
print("✅ Baseline models estimated.")

# Export to LaTeX
output_file = "../outputs/tables/baseline_regression_table.tex"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

latex_str = baseline_comparison.summary.as_latex()
for old, new in LATEX_LABEL_MAP.items():
    latex_str = latex_str.replace(old, new)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(latex_str)

print(f"✅ Baseline table saved to: {output_file}")
display(baseline_comparison)

# --- Cell 55 ---
from linearmodels.panel import compare
from analysis.regression_utils import prepare_regression_data, run_panel_ols, generate_marginal_effects, LATEX_LABEL_MAP

indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
interaction_results = {}

for idx_name in indices:
    # Prepare variables with lags
    current_ctrl_vars = [idx_name, "ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"]
    reg_data = create_lags(master_regimes, current_ctrl_vars, lags=[1])
    
    g_var = f"{idx_name}_lag1"
    lagged_ctrls = [f"{v}_lag1" for v in current_ctrl_vars if v != idx_name]
    
    # Use regression utils for unified prep and estimation
    ols_data, exog_vars = prepare_regression_data(reg_data, "sstran", g_var, lagged_ctrls, interactions=True)
    interaction_results[idx_name] = run_panel_ols(ols_data, "sstran", exog_vars)
    
    # Marginal effects summary
    print(f"\n{'='*20} {idx_name} {'='*20}")
    me_table = generate_marginal_effects(interaction_results[idx_name], g_var)
    display(me_table.round(4))

# Summary Comparison
interaction_comparison = compare(interaction_results, stars=True)
display(interaction_comparison)

# Export Summary Table to LaTeX
output_file = "../outputs/tables/interaction_regression_table.tex"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

latex_str = interaction_comparison.summary.as_latex()
for old, new in LATEX_LABEL_MAP.items():
    latex_str = latex_str.replace(old, new)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(latex_str)
print(f"\n✅ Interaction table saved to: {output_file}")

# --- Cell 56 ---
from linearmodels.panel import compare
from analysis.regression_utils import prepare_regression_data, run_panel_ols, generate_marginal_effects, LATEX_LABEL_MAP

indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
interaction_results = {}

for idx_name in indices:
    # Prepare variables with lags
    current_ctrl_vars = [idx_name, "ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"]
    reg_data = create_lags(master_regimes, current_ctrl_vars, lags=[1])
    
    g_var = f"{idx_name}_lag1"
    lagged_ctrls = [f"{v}_lag1" for v in current_ctrl_vars if v != idx_name]
    
    # Use regression utils for unified prep and estimation
    ols_data, exog_vars = prepare_regression_data(reg_data, "sstran", g_var, lagged_ctrls, interactions=True)
    interaction_results[idx_name] = run_panel_ols(ols_data, "sstran", exog_vars)
    
    # Marginal effects summary
    print(f"\n{'='*20} {idx_name} {'='*20}")
    me_table = generate_marginal_effects(interaction_results[idx_name], g_var)
    display(me_table.round(4))

# Summary Comparison
interaction_comparison = compare(interaction_results, stars=True)
display(interaction_comparison)

# Export Summary Table to LaTeX
output_file = "../outputs/tables/interaction_regression_table.tex"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

latex_str = interaction_comparison.summary.as_latex()
for old, new in LATEX_LABEL_MAP.items():
    latex_str = latex_str.replace(old, new)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(latex_str)
print(f"\n✅ Interaction table saved to: {output_file}")

# --- Cell 57 ---
from linearmodels.panel import compare
from analysis.regression_utils import prepare_regression_data, run_panel_ols, generate_marginal_effects, LATEX_LABEL_MAP

indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
interaction_results = {}

for idx_name in indices:
    # Prepare variables with lags
    current_ctrl_vars = [idx_name, "ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"]
    reg_data = create_lags(master_regimes, current_ctrl_vars, lags=[1])
    
    g_var = f"{idx_name}_lag1"
    lagged_ctrls = [f"{v}_lag1" for v in current_ctrl_vars if v != idx_name]
    
    # Use regression utils for unified prep and estimation
    ols_data, exog_vars = prepare_regression_data(reg_data, "sstran", g_var, lagged_ctrls, interactions=True)
    interaction_results[idx_name] = run_panel_ols(ols_data, "sstran", exog_vars)
    
    # Marginal effects summary
    print(f"\n{'='*20} {idx_name} {'='*20}")
    me_table = generate_marginal_effects(interaction_results[idx_name], g_var)
    display(me_table.round(4))

# Summary Comparison
interaction_comparison = compare(interaction_results, stars=True)
display(interaction_comparison)

# Export Summary Table to LaTeX
output_file = "../outputs/tables/interaction_regression_table.tex"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

latex_str = interaction_comparison.summary.as_latex()
for old, new in LATEX_LABEL_MAP.items():
    latex_str = latex_str.replace(old, new)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(latex_str)
print(f"\n✅ Interaction table saved to: {output_file}")

# --- Cell 58 ---
from linearmodels.panel import compare
from analysis.regression_utils import prepare_regression_data, run_panel_ols, generate_marginal_effects, LATEX_LABEL_MAP

indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
interaction_results = {}

for idx_name in indices:
    # Prepare variables with lags
    current_ctrl_vars = [idx_name, "ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"]
    reg_data = create_lags(master_regimes, current_ctrl_vars, lags=[1])
    
    g_var = f"{idx_name}_lag1"
    lagged_ctrls = [f"{v}_lag1" for v in current_ctrl_vars if v != idx_name]
    
    # Use regression utils for unified prep and estimation
    ols_data, exog_vars = prepare_regression_data(reg_data, "sstran", g_var, lagged_ctrls, interactions=True)
    interaction_results[idx_name] = run_panel_ols(ols_data, "sstran", exog_vars)
    
    # Marginal effects summary
    print(f"\n{'='*20} {idx_name} {'='*20}")
    me_table = generate_marginal_effects(interaction_results[idx_name], g_var)
    display(me_table.round(4))

# Summary Comparison
interaction_comparison = compare(interaction_results, stars=True)
display(interaction_comparison)

# Export Summary Table to LaTeX
output_file = "../outputs/tables/interaction_regression_table.tex"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

latex_str = interaction_comparison.summary.as_latex()
for old, new in LATEX_LABEL_MAP.items():
    latex_str = latex_str.replace(old, new)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(latex_str)
print(f"\n✅ Interaction table saved to: {output_file}")

# --- Cell 60 ---
from linearmodels.panel import compare
from analysis.regression_utils import prepare_regression_data, run_panel_ols, generate_marginal_effects, LATEX_LABEL_MAP

indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
interaction_results = {}

for idx_name in indices:
    # Prepare variables with lags
    current_ctrl_vars = [idx_name, "ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"]
    reg_data = create_lags(master_regimes, current_ctrl_vars, lags=[1])
    
    g_var = f"{idx_name}_lag1"
    lagged_ctrls = [f"{v}_lag1" for v in current_ctrl_vars if v != idx_name]
    
    # Use regression utils for unified prep and estimation
    ols_data, exog_vars = prepare_regression_data(reg_data, "sstran", g_var, lagged_ctrls, interactions=True)
    interaction_results[idx_name] = run_panel_ols(ols_data, "sstran", exog_vars)
    
    # Marginal effects summary
    print(f"\n{'='*20} {idx_name} {'='*20}")
    me_table = generate_marginal_effects(interaction_results[idx_name], g_var)
    display(me_table.round(4))

# Summary Comparison
interaction_comparison = compare(interaction_results, stars=True)
display(interaction_comparison)

# Export Summary Table to LaTeX
output_file = "../outputs/tables/interaction_regression_table.tex"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

latex_str = interaction_comparison.summary.as_latex()
for old, new in LATEX_LABEL_MAP.items():
    latex_str = latex_str.replace(old, new)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(latex_str)
print(f"\n✅ Interaction table saved to: {output_file}")

# --- Cell 63 ---
# ── Feedback Regressions: Globalization = f(sstran_{t-1}, controls_{t-1}) ──
from linearmodels.panel import compare
import os
from analysis.regression_utils import prepare_regression_data, run_panel_ols, LATEX_LABEL_MAP

# 1. Define variables
dv_indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
ctrl_vars = ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"]
iv_var = "sstran"
feedback_models = {}

for dv_name in dv_indices:
    # Create lags for sstran and all controls
    all_needed_vars = [dv_name, iv_var] + ctrl_vars
    reg_data = create_lags(master_regimes, all_needed_vars, lags=[1])
    
    dv_var = dv_name # Dependent variable is the current level
    iv_lagged = f"{iv_var}_lag1"
    ctrls_lagged = [f"{v}_lag1" for v in ctrl_vars]
    
    # Use regression utils (no interactions for feedback)
    ols_data, exog_vars = prepare_regression_data(reg_data, dv_var, iv_lagged, ctrls_lagged, interactions=False)
    feedback_models[dv_name] = run_panel_ols(ols_data, dv_var, exog_vars)

# 2. Compare models with stars
feedback_comparison = compare(feedback_models, stars=True)
print("✅ Feedback regressions estimated with significance stars.")

# 3. Export to LaTeX
output_file = "../outputs/tables/feedback_regression_table.tex"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

latex_str = feedback_comparison.summary.as_latex()
for old, new in LATEX_LABEL_MAP.items():
    latex_str = latex_str.replace(old, new)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(latex_str)

print(f"✅ Feedback regression table saved to: {output_file}")
display(feedback_comparison)


# --- Cell 64 ---

