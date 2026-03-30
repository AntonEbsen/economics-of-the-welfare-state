import os
import sys

import streamlit as st
import yaml

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from analysis.regression_utils import LATEX_LABEL_MAP

st.set_page_config(page_title="Welfare State & Globalization Dashboard", layout="wide")

st.title("🌐 Globalization & The Welfare State")
st.markdown(
    """
This dashboard provides an interactive look at the relationship between different dimensions of globalization
and social security transfers across OECD countries.
"""
)

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Sidebar
st.sidebar.header("Settings")
selected_index = st.sidebar.selectbox("Select Globalization Index", config["indices"])

# Main Content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Configuration")
    st.write(f"**Dependent Variable:** `{config['dependent_var']}`")
    st.write(f"**Controls:** {', '.join(config['controls'])}")
    st.write(f"**Lags:** {config['lags']}")

with col2:
    st.subheader("LaTeX Mapping")
    with st.expander("View Variable Labels"):
        st.json(LATEX_LABEL_MAP)

st.divider()

# Placeholder for results (In a real scenario, we might load saved .parquet results)
st.info("Tip: Use the sidebar to explore different globalization dimensions.")

# Visualization Placeholder
st.subheader("Project Statistics")
stats_col1, stats_col2, stats_col3 = st.columns(3)
stats_col1.metric("Countries", "30+")
stats_col2.metric("Time span", "1970-2022")
stats_col3.metric("Regression Models", "12")

st.markdown("---")
st.caption("Developed for Economics of the Welfare State (Modern Pipeline)")
