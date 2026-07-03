# Reproducing the analysis

The raw Formspree export contains personal data and is intentionally not stored
in this repository. With the CSV saved locally, regenerate all anonymous tables,
macros, and calibration-range figures from the repository root:

```sh
python analysis/analyze_sej.py /path/to/formspree_submissions.csv
```

The analysis uses the ten realizations documented in Appendix A, a 10% intrinsic-
range overshoot, uniform backgrounds for proportions, and log-uniform backgrounds
for positive scale variables. Generated report inputs are committed so the LaTeX
document remains buildable without access to respondents' names or email addresses.
