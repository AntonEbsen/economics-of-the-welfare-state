import nbformat as nbf
import os

nb_path = r'c:\Users\Anton\economics-of-the-welfare-state\notebooks\02_modern_pipeline.ipynb'

# Read the notebook using nbformat
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Correlation Matrix Cell (Include LaTeX)
new_corr_code = r'''import pandas as pd
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
'''

# Replace old correlation cell
for cell in nb.cells:
    if cell.cell_type == 'code' and ('Table 8: Correlation Matrix' in cell.source or 'calculate_pvalues' in cell.source):
        cell.source = new_corr_code
        break

# 2. Add Welfare Regime Categorization Section
# Before the end, after merging or cleaning
insert_idx = len(nb.cells)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and 'Part 6: Visualization' in cell.source:
        insert_idx = i
        break

md_regime = nbf.v4.new_markdown_cell('### 🏛️ Welfare Regime Categorization\nDefining Liberal, Conservative, Social Democrat, Mediterranean, and Post-Communist regimes for fixed effects analysis.')

code_regime = nbf.v4.new_code_cell(r'''from clean.panel_utils import add_welfare_regimes

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
''')

# Ensure we don't duplicate
if not any('🏛️ Welfare Regime Categorization' in c.source for c in nb.cells):
    nb.cells.insert(insert_idx, code_regime)
    nb.cells.insert(insert_idx, md_regime)

# Write out
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Success! Updated {nb_path} with LaTeX export and Welfare Regime categorization.')
