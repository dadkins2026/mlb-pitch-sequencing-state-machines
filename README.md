# MLB Pitch Sequencing State Machines

This project models Major League Baseball pitch selection as a pitcher-specific sequential decision problem. Given pitcher identity, batter context, count, previous pitch sequence, lineup handedness, and recent usage history, the project asks:

> How accurately can we predict the next pitch type, and what do the model outputs reveal about pitcher-specific pitch-selection strategy?

The final workflow uses public Statcast data for the top 100 pitchers by 2025 pitch volume, builds state-machine transition probabilities, estimates OLS beta surfaces, compares boosted tree models, evaluates out-of-time 2026 generalization, and produces paper-ready visualizations for a PNAS-style final report.

## Main Findings

- A dummy majority-pitch baseline reaches about 31.9% exact accuracy on the 2025 holdout.
- A smoothed pitcher/count/previous-pitch/batter-side state machine reaches about 42.3% exact accuracy and 84.2% top-3 accuracy on the 2025 holdout.
- A recency-weighted stacked model combining histogram gradient boosting, XGBoost, and state-machine probabilities reaches about 44.1% exact accuracy on the 2025 holdout and 40.4% exact accuracy on 2026 validation.
- OLS beta heatmaps and grouped SHAP outputs show that pitcher arsenal, previous pitch type, count leverage, batter handedness, and recent usage history are the most important explanatory signals.

These models predict observed pitch choice. They do not prove that a pitch was optimal or causal, because pitch selection is affected by scouting reports, catcher signs, fatigue, pitcher health, hitter adjustment, and other unobserved factors.

## Directory Layout

```text
.
├── 00_pull.ipynb
├── 01_merge.ipynb
├── 02_state_machine_ols.ipynb
├── 03_temporal_model_sweep.ipynb
├── 04_paper_outputs.ipynb
├── 05_pitch_sequence_pipeline.ipynb
├── ...
├── 16_make_final_paper_visuals.ipynb
├── code/
├── data/
├── output/
├── paper_overleaf/
└── requirements.txt
```

- `00_` through `04_`: main numbered notebooks for the final paper workflow.
- `05_` through `16_`: sequential audit notebooks converted from the Python scripts in `code/`.
- `code/`: reusable Python pipeline scripts.
- `data/`: raw and processed Statcast data. If large data files are omitted from a clone, they can be regenerated from public Statcast data by running the notebooks in order.
- `output/`: top-100 model metrics, transition tables, OLS beta tables, prediction audits, SHAP outputs, stacked-model results, and paper figures.
- `paper_overleaf/`: PNAS-style manuscript draft and selected figure files.

## Colab Quick Start

1. Open the public GitHub repository in Google Colab.
2. Start with `00_pull.ipynb`.
3. Run `00_pull.ipynb`, `01_merge.ipynb`, `02_state_machine_ols.ipynb`, `03_temporal_model_sweep.ipynb`, and `04_paper_outputs.ipynb` in order.
4. Keep `TOP_N = 100` for the final project run. Use a smaller value only for debugging.
5. Run `10_tune_boosted_and_shap.ipynb`, `11_compare_recency_boosters.ipynb`, `12_stacked_exact_model.ipynb`, and `16_make_final_paper_visuals.ipynb` for the final boosted, stacked, SHAP, and visualization extensions.

Local equivalent:

```bash
python3 -m pip install -r requirements.txt
python3 code/pitch_sequence_pipeline.py --top-n 100
python3 code/pitch_sequence_extension.py --top-n 100
python3 code/state_machine_and_ols.py --top-n 100
python3 code/pitch_choice_diagnostics.py --top-n 100
python3 code/paper_ready_analysis.py --top-n 100
python3 code/tune_boosted_and_shap.py --top-n 100
python3 code/compare_recency_boosters.py --top-n 100 --skip-catboost
python3 code/stacked_exact_model.py --top-n 100
python3 code/make_final_paper_visuals.py
```

`compare_recency_boosters.py` can run CatBoost if CatBoost is installed. The final submitted comparison focuses on HGB, XGBoost, and the stacked model.

## Notebook Map

| Notebook | Inputs | Function | Outputs |
| --- | --- | --- | --- |
| `00_pull.ipynb` | Public Statcast data; 2025 pitcher volume ranking | Selects the top 100 pitchers and caches pitcher-level Statcast files | `output/top_100_pitchers_2025.csv`; `data/raw/statcast_2025_pitcher_*.parquet` |
| `01_merge.ipynb` | Raw Statcast parquet files; top-100 pitcher list | Merges player/context data, engineers count/base/lag features, constructs lineup handedness features, and prints diagnostics before and after merges | `data/processed/sequence_features_2025_top100.parquet`; batter lookup files |
| `02_state_machine_ols.ipynb` | 2025 processed feature table | Builds pitcher-specific state machines, OLS linear probability beta tables, arsenal masks, and per-pitcher/pooled boosted comparisons | `output/state_machine_transitions.csv`; `output/ols_beta_pitcher_pitch_intercepts.csv`; `output/state_ols_model_comparison_metrics.csv` |
| `03_temporal_model_sweep.ipynb` | 2025 features; same-pitcher 2026 features | Evaluates out-of-time 2026 validation and runs model/context diagnostic sweeps | `output/temporal_validation_2025_to_2026.csv`; `output/next_pitch_model_sweep_metrics.csv`; prediction audits |
| `04_paper_outputs.ipynb` | Model outputs; processed features | Produces paper-ready figures, tables, SHAP summaries, and result outline | `output/paper_ready/figures/`; `output/paper_ready/tables/`; `output/paper_ready/paper_ready_summary.json` |
| `05_pitch_sequence_pipeline.ipynb` | Same as `code/pitch_sequence_pipeline.py` | Source notebook for pulling data and engineering the first processed sequence table | Same as `00_pull.ipynb` and `01_merge.ipynb` pipeline outputs |
| `06_pitch_sequence_extension.ipynb` | Processed features and model outputs | Source notebook for extended validation figures and supporting pitch-sequence outputs | `outputs/figures/`; additional model comparison CSVs |
| `07_state_machine_and_ols.ipynb` | Processed 2025 and 2026 feature tables | Source notebook for state-machine, OLS, arsenal masking, and model comparison code | `output/state_ols_*`; `output/ols_*`; `output/state_machine_*` |
| `08_pitch_choice_diagnostics.ipynb` | Processed features; model predictions | Source notebook for next-pitch sweep, context uncertainty, calibration, and count-level diagnostics | `output/pitch_choice_context_entropy.csv`; `output/prediction_confidence_calibration.csv`; `output/figures/` |
| `09_paper_ready_analysis.ipynb` | Model comparison tables; SHAP-ready models | Source notebook for paper-ready model comparisons, beta heatmaps, and SHAP panels | `output/paper_ready/` |
| `10_tune_boosted_and_shap.ipynb` | Processed features; prior paper-ready outputs | Tunes HGB hyperparameters and rebuilds SHAP summaries without rerunning the full pipeline | `output/paper_ready/boosted_tuning_shap_summary.json`; SHAP figures/tables |
| `11_compare_recency_boosters.ipynb` | Processed 2025 and 2026 features | Compares recency-weighted HGB, XGBoost, and optional CatBoost on the same holdout/validation splits | `output/paper_ready/tables/recency_booster_model_comparison.csv` |
| `12_stacked_exact_model.ipynb` | Processed features; current-game and recent-start feature builders | Trains the stacked HGB/XGBoost/state-machine exact-pitch model | `output/stacked_exact/tables/stacked_current_game_model_comparison.csv`; summary JSON |
| `13_export_site_data.ipynb` | Output tables and figures | Exports selected results into the website/app data format | `public/data/site-data.json` |
| `14_make_colab_notebooks.ipynb` | Python scripts in `code/` | Converts selected scripts into notebook form | Generated Colab notebooks |
| `15_make_review_notebooks.ipynb` | Python scripts and root notebooks | Creates review copies of notebooks for Colab editing | `colab_review_notebooks/` |
| `16_make_final_paper_visuals.ipynb` | Existing model outputs and paper-ready tables | Builds the final narrative figure pack for the paper | `output/final_paper_visuals/figures/`; `output/final_paper_visuals/tables/` |

## Merge Diagnostics

The pipeline uses a reusable `merge_with_diagnostics()` helper in `code/pitch_sequence_pipeline.py`. It logs:

- rows in the left table before the merge,
- rows in the right table before the merge,
- rows after the merge,
- unmatched left rows after the merge,
- merge key columns and validation mode.

This is used for the lineup-handedness merge in the feature-engineering pipeline and is visible in `01_merge.ipynb`.

## Reproducibility And Repo Hygiene

- Notebook filenames are numbered sequentially and contain no spaces.
- Project paths are built from the repository root with `pathlib`; no local absolute paths are required.
- Functions are defined near the top of notebooks/scripts before execution blocks.
- The README documents inputs, function, and outputs for each notebook.
- The repo uses `code/`, `data/`, and `output/` as the core project layout.
- `.env`, virtual environments, local build folders, and cache folders are ignored.
- Data are public Statcast-derived baseball data; large files may be regenerated if they are not included in a clone.

## Course Topic Coverage

- Regression: OLS linear probability models and beta coefficient tables.
- Linear Algebra and NumPy: design matrices, one-hot encodings, transition matrices, and probability vectors.
- Gradient Descent: multinomial logistic regression stacker and baselines.
- Decision Trees and Random Forests: tree-based comparison and SHAP-friendly model interpretation.
- Gradient-Boosting Trees: HGB, XGBoost, and optional CatBoost comparison.
- SHAP Explainers: grouped and pitch-specific SHAP outputs.
- Reinforcement Learning: discussed as future work for policy/reward modeling of pitch calls.
- Social Network Analysis: pitcher-batter repeated-interaction and matchup features.
- Agentic AI in Practice: AI-assisted coding, debugging, model comparison, visualization, and critical reflection.

## Paper Outputs

The PNAS-style manuscript draft is in `paper_overleaf/main.tex`. The strongest figure set for the final narrative is in `output/final_paper_visuals/figures/`, especially:

- `01_model_comparison_exact_top3.png`
- `04_state_context_entropy_mode_share.png`
- `08_ols_pitcher_pitch_beta_heatmap.png`
- `09_ols_context_beta_heatmap.png`
- `11_shap_grouped_global_importance.png`
- `12_shap_pitch_specific_panels.png`
- `14_pitch_transition_matrix.png`

## Data Availability

Raw pitch-level observations come from public Baseball Savant Statcast data. The project stores processed data under `data/processed/` and output artifacts under `output/`. If the final public repository omits large data files, rerun `00_pull.ipynb` and `01_merge.ipynb` to rebuild them.
