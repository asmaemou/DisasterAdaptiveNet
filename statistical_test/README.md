# RS-BDNet statistical tests

This folder keeps the statistical analysis separate from the training and
ablation code.

The active Python environment must provide the packages listed in
`requirements.txt`. The supplied batch job activates
`disasteradaptivenet_cu128` before running the analysis.

## Analyses

The script uses damage Macro-F1 from the same six datasets for every model
configuration and calculates:

1. Friedman omnibus test across all six configurations.
2. Kendall's W as the Friedman effect size.
3. Two-sided paired Wilcoxon signed-rank tests comparing full RS-BDNet with
   each of the five ablations.
4. Holm-adjusted p-values for the five post-hoc comparisons.
5. Paired percentile-bootstrap 95% confidence intervals for the mean
   dataset-level Macro-F1 difference.

The six datasets are the paired observations. Pixels are not treated as
independent samples.

## Run on the cluster

From the repository root:

```bash
sbatch statistical_test/run_rsbdnet_statistical_tests.sbatch
```

Monitor the job using the job number returned by Slurm:

```bash
tail -f transformer/sbatch/logs/rsbd_stats_<JOB_ID>.out
```

## Output files

Results are written to `output/RSBDNet_statistical_tests/`:

- `friedman_kendall_results.csv`
- `wilcoxon_holm_bootstrap_results.csv`
- `average_ranks.csv`
- `statistical_test_report.json`

## Interpretation limits

- Only six paired dataset-level observations are available, so the tests have
  limited power.
- The bootstrap confidence intervals quantify variation across the evaluated
  datasets, not variation across random training seeds.
- Multiple-seed training is still needed to measure optimization variability.
- A future image-level analysis should use paired images or appropriately
  defined spatial units, never individual pixels as independent observations.
