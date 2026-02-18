# EPA-NFL

Build an Expected Points (EP) model and Expected Points Added (EPA) from NFL play-by-play data.

This repo follows the roadmap in `docs/epa_project_roadmap.md` and implements each stage as a standalone script under `src/`.

## Project Structure

- `data/raw/` — nflfastR play-by-play parquet files (`play_by_play_YYYY.parquet`)
- `data/clean/` — optional cleaned outputs
- `src/` — pipeline stages (load, clean, possessions, targets, models, EPA, visuals)
- `outputs/figures/` — heatmaps and charts (ignored by git)
- `outputs/tables/` — summary tables (ignored by git)

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow numpy scikit-learn matplotlib seaborn plotly xgboost tqdm jupyterlab
```

On macOS, XGBoost requires OpenMP:

```bash
brew install libomp
```

## Quick Start

Run the full pipeline for stable 2025 analysis (recommended training on 2018–2024):

```bash
source .venv/bin/activate

python src/train_ep_model.py --years 2018 2019 2020 2021 2022 2023 2024
python src/generate_ep.py --train-years 2018 2019 2020 2021 2022 2023 2024 --predict-years 2025
python src/compute_epa.py --train-years 2018 2019 2020 2021 2022 2023 2024 --predict-years 2025
python src/sanity_checks.py --train-years 2018 2019 2020 2021 2022 2023 2024 --predict-years 2025
python src/visualization.py --train-years 2018 2019 2020 2021 2022 2023 2024 --predict-years 2025
python src/team_analysis.py --train-years 2018 2019 2020 2021 2022 2023 2024 --predict-years 2025
python src/publish_findings.py --train-years 2018 2019 2020 2021 2022 2023 2024 --predict-years 2025
```

If you want a faster smoke test, you can train and predict on 2025 only, but results will be noisier.

## Pipeline Stages (Scripts)

1. `src/load_data.py` — load raw parquet data
2. `src/clean_data.py` — clean and filter plays
3. `src/possession.py` — add possession tracking
4. `src/target_construction.py` — build `future_points` target
5. `src/train_baseline.py` — baseline linear regression EP model
6. `src/train_ep_model.py` — main XGBoost EP model
7. `src/generate_ep.py` — generate EP values for play states
8. `src/compute_epa.py` — compute EPA
9. `src/sanity_checks.py` — validate EP/EPA behavior
10. `src/visualization.py` — EP heatmaps by down
11. `src/team_analysis.py` — 2025 EPA summaries
12. `src/publish_findings.py` — markdown summary report

## Notes

- The model only uses **pre-play state features** to avoid leakage.
- `outputs/` and `.venv/` are gitignored to keep the repo lightweight.
- You may see harmless `pyarrow` CPU info warnings during parquet reads.
- Matplotlib writes its cache to `outputs/.mplconfig` to avoid permissions issues.

## Roadmap

See `docs/epa_project_roadmap.md` for the original step-by-step plan.
