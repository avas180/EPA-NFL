# Rebuilding an Expected Points (EP) / EPA Model from NFL Play-by-Play Data

This document is a step-by-step roadmap for constructing an Expected Points (EP) model and deriving Expected Points Added (EPA) using nflfastR play-by-play data (Parquet files). The intent is for this file to be executable guidance for code generation tools (e.g., Codex) and for a human to follow sequentially.

---

## 1. Project Goals

Primary goals:
1. Construct an Expected Points (EP) model that predicts the future points scored by the offense from a given game state.
2. Compute Expected Points Added (EPA) for every play in the 2025 NFL season.
3. Visualize the EP surface (value of field position and down/distance states).
4. Analyze strategic implications and trends during the 2025 season.

Secondary goals:
- Understand how football behaves as a state-transition system.
- Identify teams and players that consistently generate positive EPA.
- Compare run vs pass decision efficiency.

---

## 2. Programming Environment

### Language
Python 3.11+

### Required Packages
Install using:

```bash
pip install pandas pyarrow numpy scikit-learn matplotlib seaborn plotly xgboost tqdm jupyterlab
```

Package roles:

| Package | Purpose |
|--------|------|
| pandas | data manipulation |
| pyarrow | reading parquet efficiently |
| numpy | numerical operations |
| scikit-learn | baseline regression models |
| xgboost | main EP prediction model |
| matplotlib | static plots |
| seaborn | statistical visualizations |
| plotly | interactive visualizations |
| tqdm | progress bars |
| jupyterlab | experimentation notebooks |

---

## 3. Repository Structure

```
EPA-NFL/
│
├── data/
│   └── raw/                # original parquet files
│
├── notebooks/              # exploratory analysis
│   ├── 01_explore_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ep_model.ipynb
│   └── 04_2025_analysis.ipynb
│
├── src/
│   ├── load_data.py
│   ├── feature_engineering.py
│   ├── target_construction.py
│   ├── train_ep_model.py
│   ├── compute_epa.py
│   └── visualization.py
│
├── models/
│   └── ep_model.pkl
│
├── outputs/
│   ├── figures/
│   └── tables/
│
└── README.md
```

---

## 4. Data Requirements

### Should you use only 2025?
You **can train** an EP model using only 2025, but it will be noisy.

Recommendation:
- Train EP model on **2018–2024 data** (stable football environment)
- Apply EPA analysis specifically to **2025 only**

Reason:
Expected Points is a property of the sport, not the season. More seasons produce a stable estimate of state value, while still allowing 2025 to be studied as a unique strategic environment.

---

## 5. Understanding the Target Variable (Critical Step)

We are NOT predicting yards gained or points on the play.

We predict:

**Future points scored by the offense on the current drive before possession changes.**

### How to compute future points
For each play:

1. Identify the current possession team.
2. Track forward in the dataset until:
   - touchdown
   - field goal
   - turnover
   - end of half/game
3. Sum points scored by the offense during that possession.

This becomes the regression target: `future_points`.

This step is the most important in the entire project.

---

## 6. Define the Game State (Model Inputs)

Use the following features only (initial model):

| Feature | Description |
|---|---|
| down | current down (1–4) |
| ydstogo | yards needed for first down |
| yardline_100 | distance to opponent end zone |
| half_seconds_remaining | clock importance |
| score_differential | offense score - defense score |
| goal_to_go | whether first down marker is the endzone |

Do NOT include play result variables (yards gained, pass/run outcome, etc.).

We are valuing the state BEFORE the play occurs.

---

## 7. Data Cleaning Steps

1. Remove non-plays (timeouts, penalties without plays, etc.)
2. Remove special teams plays initially (punts/kickoffs) for the first model
3. Remove plays near halftime with < 30 seconds remaining
4. Ensure `yardline_100` is offense-relative

---

## 8. Model Training

### Baseline Model (for understanding)
Linear regression:

```python
from sklearn.linear_model import LinearRegression
```

### Main Model (recommended)
XGBoost regression:

```python
from xgboost import XGBRegressor
```

Train:

```
future_points = f(state_features)
```

Split:
- train: 80%
- test: 20%

Evaluation metrics:
- RMSE
- MAE

---

## 9. Construct Expected Points (EP)

After training:

For each play state:

```
EP_before = model.predict(state_features_before_play)
```

Store this value in the dataset.

---

## 10. Compute EPA

For each play:

```
EPA = EP_after - EP_before
```

Where `EP_after` is the model prediction using the next play's state.

Special cases:

| Event | EP_after value |
|---|---|
| touchdown | 6.0 - opponent_start_EP |
| field goal | 3.0 - opponent_start_EP |
| turnover | opponent_EP * (-1) |

Opponent starting EP can be approximated using a kickoff state at own 25.

---

## 11. Required Visualizations (Mandatory)

### Field Value Maps
- Heatmap: yardline vs expected points
- Separate by down (1st, 2nd, 3rd, 4th)

### State Effects
- EP vs yards-to-go
- EP vs score differential

### Play Value
- Histogram of EPA distribution
- Top 20 highest EPA plays (2025)
- Lowest 20 EPA plays (turnovers)

### Strategy Analysis (Important)
- Run EPA vs Pass EPA
- Early-down pass vs run success
- 4th-down go-for-it success vs punt

### Team Analysis (2025)
- Offensive EPA/play by team
- Defensive EPA/play allowed
- Quarterback EPA/play

---

## 12. 2025 Season Insights to Investigate

Add these analyses after computing EPA:

1. Early-down pass rate vs offensive efficiency
2. Which teams benefited most from 4th-down aggression
3. Red-zone interception impact
4. Which offenses rely on explosive plays vs sustained drives
5. EPA consistency (variance of EPA/play)

Additional interesting metric to compute:

### Success Rate
A play is successful if:
- 1st down: gain ≥ 40% yards-to-go
- 2nd down: gain ≥ 60% yards-to-go
- 3rd/4th down: gain ≥ 100% yards-to-go

Compare Success Rate vs EPA/play.

---

## 13. Stretch Goals

- Win Probability model
- Drive survival probability
- 4th-down decision calculator
- Player decision quality analysis

---

## 14. Order of Implementation (Follow Exactly)

1. Load parquet data
2. Clean dataset
3. Build possession tracking
4. Compute future_points target
5. Train baseline regression
6. Train XGBoost model
7. Generate EP values
8. Compute EPA
9. Validate with sanity checks
10. Produce field value heatmaps
11. Analyze 2025 teams
12. Publish findings

---

## 15. Sanity Checks (Very Important)

Your model is probably correct if:

- EP near own goal line ≈ negative
- EP near opponent goal line ≈ ~6
- 3rd & long significantly worse than 2nd & short
- Interceptions produce large negative EPA
- 4th down conversions produce large positive EPA

If these are not true, the target construction step is wrong.

---

End of roadmap.

