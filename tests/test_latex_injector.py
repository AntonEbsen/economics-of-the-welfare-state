"""
Tests for ``src/analysis/latex_injector.py`` — placeholder substitution
in LaTeX templates.
"""

from __future__ import annotations

from analysis.latex_injector import inject_latex_results


def test_inject_replaces_placeholders(tmp_path):
    tpl = tmp_path / "template.tex"
    tpl.write_text(r"The coefficient is {{COEF}} (p={{PVAL}}).", encoding="utf-8")

    inject_latex_results(str(tpl), {"COEF": "-0.143", "PVAL": "0.002"})
    assert tpl.read_text(encoding="utf-8") == r"The coefficient is -0.143 (p=0.002)."


def test_inject_writes_to_separate_output(tmp_path):
    tpl = tmp_path / "template.tex"
    out = tmp_path / "injected.tex"
    tpl.write_text("Result: {{X}}", encoding="utf-8")

    inject_latex_results(str(tpl), {"X": "42"}, output_path=str(out))
    # Template untouched
    assert "{{X}}" in tpl.read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8") == "Result: 42"


def test_inject_missing_template_does_not_raise(tmp_path, capsys):
    inject_latex_results(str(tmp_path / "no_such.tex"), {"A": "1"})
    captured = capsys.readouterr()
    assert "Template not found" in captured.out


def test_inject_leaves_unmatched_placeholders_intact(tmp_path):
    tpl = tmp_path / "template.tex"
    tpl.write_text("{{FOUND}} and {{MISSING}}", encoding="utf-8")

    inject_latex_results(str(tpl), {"FOUND": "yes"})
    assert tpl.read_text(encoding="utf-8") == "yes and {{MISSING}}"
