"""
Data Bridge: Export econometric results to JSON for the Research Portal.
"""

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from clean.panel_utils import create_lags
from clean.utils import load_config
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from analysis.regression_utils import LATEX_LABEL_MAP, prepare_regression_data, run_panel_ols

logger = logging.getLogger(__name__)


def export_all_web_data(master: pd.DataFrame, config: dict, out_dir: str | Path):
    """
    Main export engine.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = config.get("indices", ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    controls = config.get("controls", ["ln_gdppc", "inflation_cpi", "deficit", "debt"])
    dep_var = config.get("dependent_var", "sstran")

    # 1. Export Specification Curves
    logger.info("📡 Exporting Specification Curves...")
    spec_data = {}
    final_models = {}
    final_model_data = {}

    for idx in indices:
        idx_models = []

        # All subsets models (Vibration of Effects)
        for r in range(len(controls) + 1):
            for current_ctrls_tuple in itertools.combinations(controls, r):
                current_ctrls = list(current_ctrls_tuple)

                all_vars = [idx] + current_ctrls
                reg_data = create_lags(master, all_vars, lags=[1])

                g_var = f"{idx}_lag1"
                lagged_ctrls = [f"{v}_lag1" for v in current_ctrls]

                ols_data, exog_vars = prepare_regression_data(
                    reg_data, dep_var, g_var, lagged_ctrls, interactions=False
                )

                res = run_panel_ols(ols_data, dep_var, exog_vars)

                # Extract point estimate and CI for the main index
                coef = float(res.params[g_var])
                se = float(res.std_errors[g_var])

                model_entry = {
                    "name": f"Subset ({len(current_ctrls)} controls)",
                    "coefficient": round(coef, 4),
                    "lower_ci": round(coef - 1.96 * se, 4),
                    "upper_ci": round(coef + 1.96 * se, 4),
                    "significant": bool(res.pvalues[g_var] < 0.05),
                    "controls": current_ctrls,  # Keep bare variable names for easy matching
                    "included_controls": [
                        LATEX_LABEL_MAP.get(f"{c}_lag1", c).replace("$_{t-1}$", "")
                        for c in current_ctrls
                    ],
                }
                idx_models.append(model_entry)

                if len(current_ctrls) == len(controls):
                    final_models[idx] = res
                    final_model_data[idx] = (ols_data, exog_vars)

        # Sort the specification curve by coefficient size
        idx_models.sort(key=lambda x: x["coefficient"])

        # Re-assign rank as the name for the tooltip
        for rank, m in enumerate(idx_models):
            m["rank"] = rank + 1

        spec_data[idx] = idx_models

    with open(out_dir / "spec_curves.json", "w") as f:
        json.dump(spec_data, f, indent=2)

    # 2. Export Diagnostics (Hausman, RESET, etc.)
    logger.info("🩺 Exporting Model Diagnostics...")
    diag_results = []

    for idx, (ols_data, exog_vars) in final_model_data.items():
        res = final_models[idx]

        # Hausman
        import statsmodels.api as sm

        exog = sm.add_constant(ols_data[exog_vars])
        fe = PanelOLS(ols_data[dep_var], exog, entity_effects=True).fit(cov_type="unadjusted")
        re = RandomEffects(ols_data[dep_var], exog).fit(cov_type="unadjusted")

        shared = [v for v in fe.params.index if v in re.params.index and v != "const"]
        b_diff = fe.params[shared].values - re.params[shared].values
        v_diff = fe.cov.loc[shared, shared].values - re.cov.loc[shared, shared].values
        h_stat = float(b_diff @ np.linalg.pinv(v_diff) @ b_diff)
        h_p = 1 - stats.chi2.cdf(h_stat, len(shared))

        # RESET
        resids = res.resids
        # Simplified RESET for JSON
        lb_res = acorr_ljungbox(resids.dropna(), lags=[1], return_df=True)
        lb_p = float(lb_res["lb_pvalue"].iloc[0])

        diag_results.append(
            {
                "model": idx,
                "hausman_p": round(h_p, 3),
                "hausman_preferred": "FE" if h_p < 0.05 else "RE",
                "serial_corr_p": round(lb_p, 3),
                "serial_corr_status": "Clean" if lb_p > 0.05 else "Detected",
                "nobs": int(res.nobs),
                "rsquared": round(float(res.rsquared), 3),
            }
        )

    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diag_results, f, indent=2)

    # 3. Export Summary Stats
    logger.info("📉 Exporting Summary Stats...")
    summary = master[[dep_var] + indices + controls].describe().T
    summary_json = summary[["count", "mean", "std", "min", "max"]].to_dict(orient="index")

    # Format labels
    formatted_summary = {}
    for k, v in summary_json.items():
        label = LATEX_LABEL_MAP.get(f"{k}_lag1", k).replace("$_{t-1}$", "")
        formatted_summary[label] = {mk: round(mv, 2) for mk, mv in v.items()}

    with open(out_dir / "summary_stats.json", "w") as f:
        json.dump(formatted_summary, f, indent=2)

    # 4. Export Map Data (1980-2023)
    logger.info("🗺️ Exporting Map Data (1980-2023)...")
    # We only need a subset of variables to keep the file size manageable
    map_vars = ["iso3", "year", dep_var] + indices
    map_df = master[(master["year"] >= 1980) & (master["year"] <= 2023)][map_vars].dropna()

    # Convert to list of dicts for JSON
    map_data = map_df.to_dict(orient="records")

    with open(out_dir / "map_data.json", "w") as f:
        json.dump(map_data, f, indent=2)

    # 5. Export Simulator Parameters
    logger.info("🎛️ Exporting Simulator Parameters...")
    # Get 2023 baselines for each country
    latest_data = master[master["year"] == 2023][["iso3", dep_var] + indices].dropna()
    baselines = latest_data.set_index("iso3").to_dict(orient="index")

    # Get coefficients from the final models (the ones with all controls)
    sim_coefs = {}
    for idx, res in final_models.items():
        g_var = f"{idx}_lag1"
        sim_coefs[idx] = {
            "coef": round(float(res.params[g_var]), 5),
            "se": round(float(res.std_errors[g_var]), 5),
        }

    simulator_data = {"baselines": baselines, "coefficients": sim_coefs, "indices": indices}

    with open(out_dir / "simulator.json", "w") as f:
        json.dump(simulator_data, f, indent=2)

    # 6. Export Variance Decomposition (Partial R-squared)
    logger.info("📊 Exporting Variance Decomposition...")
    importance_data = {}

    for idx, (ols_data, exog_vars) in final_model_data.items():
        base_res = final_models[idx]
        base_r2 = base_res.rsquared

        # Calculate partial R-squared for each var (excluding constant)
        var_importance = []
        for var in exog_vars:
            if var == "const":
                continue

            # Reduced model without the target variable
            reduced_exog = [v for v in exog_vars if v != var]
            from statsmodels.api import add_constant

            reduced_X = add_constant(ols_data[reduced_exog])
            reduced_X = reduced_X.loc[:, ~reduced_X.columns.duplicated()]

            try:
                mod_red = PanelOLS(ols_data[dep_var], reduced_X, entity_effects=True)
                res_red = mod_red.fit(cov_type="unadjusted")
                red_r2 = res_red.rsquared

                # Partial R-squared formula
                if (1 - red_r2) > 0:
                    partial_r2 = (base_r2 - red_r2) / (1 - red_r2)
                    partial_r2 = max(0, partial_r2)  # prevent tiny negative numerical drift
                else:
                    partial_r2 = 0
            except Exception:
                partial_r2 = 0

            label = LATEX_LABEL_MAP.get(var, var).replace("$_{t-1}$", "").replace("log ", "")
            var_importance.append(
                {
                    "variable": var,
                    "label": label,
                    "partial_r2": round(float(partial_r2) * 100, 2),  # As percentage
                }
            )

        # Sort by importance
        var_importance.sort(key=lambda x: x["partial_r2"], reverse=True)
        importance_data[idx] = var_importance

    with open(out_dir / "importance.json", "w") as f:
        json.dump(importance_data, f, indent=2)

    # 7. Export Interactive Regression Table Data (Clustered vs Driscoll-Kraay)
    logger.info("📑 Exporting Interactive Regression Table...")
    master_table = {}

    for idx, (ols_data, exog_vars) in final_model_data.items():
        # Re-run models to get both SE types
        from statsmodels.api import add_constant

        X = add_constant(ols_data[exog_vars])
        X = X.loc[:, ~X.columns.duplicated()]

        mod = PanelOLS(ols_data[dep_var], X, entity_effects=True)
        res_clust = mod.fit(cov_type="clustered", cluster_entity=True)
        res_dk = mod.fit(cov_type="kernel")

        # Merge results for each variable
        idx_results = []
        # Sort variables to put index first, then remaining controls
        g_var = f"{idx}_lag1"
        sorted_vars = [g_var] + [v for v in exog_vars if v != g_var and v != "const"]

        for var in sorted_vars:
            if var not in res_clust.params:
                continue

            coef = float(res_clust.params[var])
            label = LATEX_LABEL_MAP.get(var, var).replace("$_{t-1}$", "")

            idx_results.append(
                {
                    "variable": var,
                    "label": label,
                    "coef": round(coef, 4),
                    "clustered": {
                        "se": round(float(res_clust.std_errors[var]), 4),
                        "pval": round(float(res_clust.pvalues[var]), 4),
                        "stars": _get_stars(res_clust.pvalues[var]),
                    },
                    "dk": {
                        "se": round(float(res_dk.std_errors[var]), 4),
                        "pval": round(float(res_dk.pvalues[var]), 4),
                        "stars": _get_stars(res_dk.pvalues[var]),
                    },
                }
            )

        master_table[idx] = {
            "nobs": int(res_clust.nobs),
            "rsquared": round(float(res_clust.rsquared), 3),
            "variables": idx_results,
        }

    with open(out_dir / "master_table.json", "w") as f:
        json.dump(master_table, f, indent=2)

    logger.info(f"✅ Web data sync complete! Files saved to {out_dir}")


def _get_stars(pval):
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


if __name__ == "__main__":
    from clean.utils import setup_logging

    setup_logging()

    config = load_config()
    master = pd.read_parquet("data/final/master_dataset.parquet")

    # Save directly into Astro's public folder for easy access
    web_data_dir = Path("web/public/data")
    export_all_web_data(master, config, web_data_dir)
