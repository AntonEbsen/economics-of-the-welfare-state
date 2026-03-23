import os


def inject_latex_results(tex_path: str, results_dict: dict, output_path: str = None):
    """
    Replace placeholders in a LaTeX file with actual results.
    Placeholders should be in the format: {{KEY}}

    Args:
        tex_path: Path to the .tex template.
        results_dict: Dictionary mapping keys to values (strings).
        output_path: Path to save the injected .tex. Defaults to overwriting tex_path.
    """
    if not os.path.exists(tex_path):
        print(f"Template not found: {tex_path}")
        return

    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    for key, value in results_dict.items():
        placeholder = f"{{{{{key}}}}}"
        content = content.replace(placeholder, str(value))

    out = output_path if output_path else tex_path
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ LaTeX injection complete: {out}")


# Example usage function for the pipeline
def update_paper_stats(main_effect: float, p_value: float):
    stats = {
        "MAIN_BETA": f"{main_effect:.3f}",
        "MAIN_PVAL": f"{p_value:.3f}",
        "SIGNIFICANCE": "significant" if p_value < 0.05 else "not significant",
    }
    # This would target a file like: paper/results_section.tex
    # inject_latex_results('paper/manuscript.tex', stats)
    pass
