# Paper draft

This folder contains the LaTeX working draft and rendered PDF for the TD3
portfolio allocation paper.

Current LaTeX source:

```text
main.tex
```

Preferred build command from the repository root:

```bash
latexmk -pdf paper/main.tex
```

If `latexmk` is unavailable, compile with Tectonic:

```bash
tectonic paper/main.tex
```

The generated PDF is written to:

```text
main.pdf
```

The Markdown file remains as a content-review draft:

```text
Robust Evaluation of TD3 Portfolio Allocation under Realistic Cross-Asset Frictions - revision David.md
```

The LaTeX draft references figures from:

```text
figures/
```

and final ranking/statistical validation tables from:

```text
../outputs/tables/final_constrained_td3_report_with_v3_clean_no_dxy_v7_clean_garch_v4_v7_v8_60ep_10seeds/
../outputs/tables/statistical_validation_final_v3_clean_no_dxy_v7_clean_garch/
```
