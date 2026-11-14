# Machine-Learning and Structured Neural Pricing Models under Heavy-Tailed Losses

Code for the paper *Machine-Learning and Structured Neural Pricing Models under Heavy-Tailed Losses: Predictive
Performance, Calibration, and Stability* (Danlei Zhu and Donglin Wang).

Eight pure-premium models are tuned with one protocol and compared on 50 train/validation/test splits (60/20/20,
seeds 41–90) of the French motor portfolio freMTPL2:

GLM (Poisson frequency × Gamma severity), Tweedie GLM, EBM, GBM (LightGBM), LocalGLMnet, CANN, Neural GA2M and
Neural GA2M+.

All models use the Tweedie deviance with power p = 1.6, and exposure enters as a weight.

The pipeline also covers:

- the screening of interaction pairs on a capped target (Neural GA2M+);
- global rescaling and a credibility-cell calibrator;
- paired comparisons over the splits.

## Repository layout

```
paper2_pipeline.ipynb            full pipeline, run from the repository root
py_files/
  paper2_adjust_clean_plus.py    data preparation, Tweedie deviance, stratified split, seeding
  paper2_models.py               model definitions and fitting (GLMs, EBM, GBM, LocalGLMnet, CANN, Neural GA2M)
  paper2_tuning.py               tuning stages and smoke test
  paper2_predictions.py          storage and loading of raw predictions
  paper2_datasets.py             data-set settings, split, stressed-seed labels, Tweedie power
  paper2_run_full.py             50-split run of the seven main models (also a command-line script)
  paper2_analysis_full.py        tables and figures R1–R11 of a run directory
  ga2m_plus.py                   Neural GA2M+: pair screening on the capped target, fit, structure stability
  paper2_calibration_study.py    credibility cells for 20 groupings and a grid of k (unclipped), heterogeneity tests
  paper2_age_calibration.py      proposed calibrator (clipped cells after global rescaling) by grouping and k
  paper2_paper_figures.py        figures and tables of the Neural GA2M+ and calibration sections
  paper2_appendix.py             appendix tables and figures (signed balance, stress label, partial dependence)
  paper2_clipping.py             clipped vs unclipped cells, sensitivity to the clipping bounds
  paper2_calibrator_selection.py grouping of the calibrator chosen on the validation set by cross-fitting
  paper2_extra_tables.py         paired comparisons with bootstrap confidence intervals (command-line script)
data/                            freMTPL2freq.csv, freMTPL2sev.csv (not tracked)
tuning/                          selected_configs.json and tuning results (written by the tuning stages)
```

## Data

freMTPL2 is available in the R package [CASdatasets](https://github.com/dutangc/CASdatasets) and on
[Kaggle](https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims). Place
`freMTPL2freq.csv` and `freMTPL2sev.csv` in `data/`.

## Installation

```
pip install -r requirements.txt
```

## Running

Open `paper2_pipeline.ipynb` from the repository root and run the sections in order. Each long section is resumable:

- finished seeds are read from disk and skipped;
- a run directory started with other settings or another Tweedie power is refused.

The main run can also be started from the command line:

```
python py_files/paper2_run_full.py --freq data/freMTPL2freq.csv --sev data/freMTPL2sev.csv \
    --configs tuning/selected_configs.json --out v2_full
```

| Step | Function | Output directory |
|---|---|---|
| Tuning (stages E, G, 1, 1b, 2, 2c, 3) | `paper2_tuning.run_tuning` | `tuning/` |
| Main run, 7 models × 50 splits | `paper2_run_full.run_full` | `v2_full/` |
| Neural GA2M+ screening, cap 0.995 | `ga2m_plus.screen_fixed_cap` | `v2_ga2mplus_screen995/` |
| Neural GA2M+ fit | `ga2m_plus.run_fixed_cap` | `v2_ga2mplus_c995_meanrank/` |
| Merge of the two runs | `ga2m_plus.merge_results` | `v2_full_plus995/` |
| Raw predictions of the main models | `paper2_predictions.archive_predictions` | `v2_predictions/` |
| Calibration study | `paper2_calibration_study.run_study` | `v2_calibration/` |
| Proposed calibrator | `paper2_age_calibration.compute_variants` | `v2_age_calibration/` |
| Paper figures and tables | `paper2_paper_figures`, `paper2_appendix` | `v2_paper/` |
| Clipping | `paper2_clipping.run` | `v2_clipping/` |
| Grouping chosen on validation | `paper2_calibrator_selection.run` | `v2_calibrator_selection/` |
| Paired tables | `py_files/paper2_extra_tables.py` | `v2_paper/tables/paired/` |

Every step after the main run and the Neural GA2M+ fit reads stored outputs only; no model is refitted.

The sets are used as follows:

- **Training set:** model fitting and the global rescaling factor.
- **Validation set:** tuning, early stopping and the totals of the credibility cells.
- **Test set:** every reported number.

## Where each result of the paper comes from

| Paper | Produced by | File |
|---|---|---|
| Table 4, Figure 2 (interaction pairs) | `paper2_paper_figures.ga2m_report` | `v2_paper/tables/paper_tables_ga2m.xlsx`, `v2_paper/figures/G1_interaction_stability` |
| Table 5 (main results) | `paper2_analysis_full.run_all(v2_full_plus995)` | `v2_full_plus995/analysis/tables/R1_main_table` |
| Table 6 (paired differences from GBM) | `paper2_extra_tables.py` | `v2_paper/tables/paired/table_vsgbm.tex` |
| Table 7 (Neural GA2M+ − Neural GA2M by claim location) | `paper2_extra_tables.py` | `v2_paper/tables/paired/table_groups.tex` |
| Figures 3, 4, 5, A2 | `paper2_paper_figures.ga2m_report` | `v2_paper/figures/G4_*`, `G2_*`, `G3_*`, `G5_*` |
| Figure 6 (calibrator by grouping) | `paper2_age_calibration.report` | `v2_age_calibration/fig1_heatmaps_k2000` |
| Figure A1 (sensitivity to k) | `paper2_age_calibration.report` | `v2_age_calibration/fig2_k_curves_by_model` |
| Table 8, Figures 7, A3 (calibration methods) | `paper2_paper_figures.calibration_report` | `v2_paper/tables/paper_tables_calibration.xlsx`, `v2_paper/figures/C4_*`, `C2_*` |
| Figure 8, Table A1 (partial dependence, shape stability) | `paper2_appendix.pd_figure` | `v2_paper/figures/F10_partial_dependence`, `v2_paper/tables/A2_shape_stability` |
| Figure 9 (categorical relativities) | `paper2_appendix.relativity_figure` | `v2_paper/figures/A1_categorical_relativities` |
| Table A2 (signed balance) | `paper2_appendix.signed_balance_table` | `v2_paper/tables/A3_signed_balance` |
| Table A3 (regular vs stressed splits) | `paper2_appendix.model_free_stress` | `v2_paper/tables/S1_model_free_stress` |
| Table A4 (WMAE, WRMSE) | `perf.csv` columns `weighted_mae`, `weighted_rmse` | `v2_full_plus995/perf.csv` |
| Figure A4 (clipped vs unclipped) | `paper2_clipping.tradeoff_figure` | `v2_clipping/clipping_tradeoff` |
| Figure A5 (grouping chosen on validation) | `paper2_calibrator_selection.figure` | `v2_calibrator_selection/calibrator_selection` |

The noise floor of the test balance comes from two places:

- the `noise floor` sheet of `paper_tables_calibration.xlsx`;
- R3 of `paper2_analysis_full`.

The cell exposures of the calibrator come from `paper2_age_calibration.cell_diagnostics`.

## Statistical comparisons

All models are evaluated on the same splits, so comparisons use the paired differences
Δ_j = M_j(model) − M_j(reference), j = 1, …, 50. The following are reported:

- the share of splits in which a model is better;
- the Wilcoxon signed-rank test, Holm-adjusted within each table and metric;
- a 95% percentile bootstrap interval of the mean difference: 10,000 resamples of the splits, seed 2026, with
  resampling within the group for claim-location groups;
- the Nadeau–Bengio corrected t-test for reference, with variance factor 1/J + n_test/n_train.
