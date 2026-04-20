"""
One-command analysis preparation script.
Processes all data, generates quality reports, and exports for analysis.

Assumes the ``econ-clean`` package has been installed in editable mode
(``pip install -e .``) so that ``clean`` and ``analysis`` resolve via the
regular import machinery.
"""

from pathlib import Path

from clean import (
    merge_all_datasets,
    process_all_datasets,
    save_master_dataset,
)
from clean.metadata import export_codebook_to_csv
from clean.quality import generate_quality_report
from clean.stats import export_stata_labels, generate_summary_stats

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = (SCRIPT_DIR / "..").resolve()


def prepare_analysis():
    """
    Complete data preparation pipeline:
    1. Process all raw datasets
    2. Merge into master dataset
    3. Generate quality report
    4. Create summary statistics
    5. Export codebook
    6. Export Stata labels
    7. Save in multiple formats
    """
    print("\n" + "=" * 70)
    print("🚀 AUTOMATED ANALYSIS PREPARATION")
    print("=" * 70)

    # Step 1: Process all datasets
    print("\n📊 Step 1/7: Processing all datasets...")
    results = process_all_datasets(
        repo_root=REPO_ROOT, year_min=1980, year_max=2023, validate=True, save_outputs=True
    )

    # Step 2: Merge datasets
    print("\n🔗 Step 2/7: Merging all datasets...")
    master = merge_all_datasets(results, how="outer")

    # Step 3: Quality report (writes HTML to reports/data_quality_report.html)
    print("\n🔍 Step 3/7: Generating quality report...")
    generate_quality_report(master, output_path=REPO_ROOT / "reports" / "data_quality_report.html")

    # Step 4: Summary statistics
    print("\n📈 Step 4/7: Generating summary statistics...")
    summary_stats = generate_summary_stats(master, output_format="pandas")
    summary_stats.to_csv(REPO_ROOT / "reports" / "summary_statistics.csv")

    # Also save as LaTeX
    latex_stats = generate_summary_stats(master, output_format="latex")
    with open(REPO_ROOT / "reports" / "summary_statistics.tex", "w") as f:
        f.write(latex_stats)
    print("✅ Summary statistics saved (CSV & LaTeX)")

    # Step 5: Export codebook
    print("\n📖 Step 5/7: Exporting codebook...")
    export_codebook_to_csv(REPO_ROOT / "reports" / "codebook.csv")

    # Step 6: Export Stata labels
    print("\n📝 Step 6/7: Exporting Stata labels...")
    export_stata_labels(master, REPO_ROOT / "analysis" / "label_variables.do")

    # Step 7: Save master dataset
    print("\n💾 Step 7/7: Saving master dataset...")
    save_master_dataset(
        master, REPO_ROOT / "data" / "final" / "master_dataset", formats=["parquet", "csv", "stata"]
    )

    # Final summary
    print("\n" + "=" * 70)
    print("✅ ANALYSIS PREPARATION COMPLETE!")
    print("=" * 70)
    print("\n📂 Output files created:")
    print("   Master dataset: data/final/master_dataset.[parquet|csv|dta]")
    print("   Quality report: reports/data_quality_report.html")
    print("   Summary stats:  reports/summary_statistics.[csv|tex]")
    print("   Codebook:       reports/codebook.csv")
    print("   Stata labels:   analysis/label_variables.do")
    print(f"\n🎯 Dataset ready for analysis: {len(master):,} observations")
    print(f"   Countries: {master['iso3'].nunique()}")
    print(f"   Years: {master['year'].min()}-{master['year'].max()}")
    print(f"   Variables: {len(master.columns)}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    prepare_analysis()
