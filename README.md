# Forecasting the Dutch Housing Crisis

Final report for TU Delft course **WI4138: Decision Theory and Expert Judgment**.
The study applies Cooke's Classical Model to seven experts' assessments of the
Dutch housing market in 2027.

## Build the report

The project is compatible with Overleaf. Locally, it can be built with Tectonic:

```sh
tectonic -X compile report.tex --outdir output/pdf
```

The report uses BibLaTeX with the BibTeX backend for portable builds.

## Reproduce the analysis

The Formspree export contains names and email addresses and is deliberately not
committed. To regenerate the anonymous results tables and figures:

```sh
python analysis/analyze_sej.py /path/to/formspree_submissions.csv
```

See [`analysis/README.md`](analysis/README.md) for the modelling assumptions.
