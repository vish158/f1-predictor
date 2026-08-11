# 🏎️ F1 Grand Prix Winner Predictor

> A machine learning pipeline that predicts Formula 1 race winners using 14+ years of historical race data, built with XGBoost and deployed as an interactive Streamlit web app.

[![CI](https://github.com/YOUR_USERNAME/f1-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/f1-predictor/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/deployed-Streamlit%20Cloud-ff4b4b)](https://streamlit.io/cloud)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📸 Demo

> _Configure any race grid → model outputs per-driver win probabilities in seconds._

![App Screenshot](/F1 Race Predictor · Streamlit.pdf)

**[🔴 Live Demo → your-app.streamlit.app](https://your-app.streamlit.app)**

---

## 🧠 How It Works

The model learns from 14+ years of F1 results (2010–2024) fetched from the [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) — the actively maintained successor to the now-defunct Ergast API.

### Features used

| Feature | Description |
|---|---|
| `grid` | Starting grid position |
| `grid_advantage` | 20 − grid (flipped so higher = better) |
| `driver_avg_pos_5` | Driver's average finishing position over last 5 races |
| `driver_wins_5` | Driver's win count over last 5 races |
| `constructor_avg_pos_5` | Constructor's average finishing position over last 5 races |
| `circuit_win_rate` | Driver's historical win rate at this specific circuit |
| `cumulative_points` | Driver's running championship points total |
| `round` | Race round number in the season |

> **Key insight:** Grid position alone explains ~40% of race outcomes — pole sitters win roughly 40% of F1 races. Rolling driver form and circuit-specific history are the next strongest signals.

### Algorithm

**XGBoost** (gradient boosting) — builds 300 sequential decision trees where each tree corrects the mistakes of the previous one. Class imbalance (only ~5% of race entries are winners) is handled via `scale_pos_weight`.

### Evaluation

| Metric | Value |
|---|---|
| ROC-AUC | ~0.87–0.92 |
| Train/test split | Time-based (train ≤ 2022, test ≥ 2023) |

ROC-AUC measures ranking quality: does the model assign higher probability to actual winners? A score of 0.5 = random guessing; 1.0 = perfect.

---

## 📁 Repository Structure

```
f1-predictor/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   └── pipeline.py                 # Core ML pipeline (fetch → features → train → predict)
│
├── notebooks/
│   └── f1_model_walkthrough.ipynb  # Step-by-step Jupyter notebook
│
├── tests/
│   └── test_pipeline.py            # Unit tests (pytest)
│
└── .github/
    └── workflows/
        └── ci.yml                  # GitHub Actions CI (runs tests on every push)
```

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/f1-predictor.git
cd f1-predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

- Click **"Fetch Data & Train Model"** in the sidebar (takes 2–4 minutes)
- Configure a race grid and click **"Run Prediction"**

### 5. (Optional) Run the Jupyter notebook

```bash
cd notebooks
jupyter notebook f1_model_walkthrough.ipynb
```

---

## 📓 Use the Pipeline Directly (no UI)

```python
from src.pipeline import (
    fetch_race_results,
    build_features,
    encode_features,
    split_data,
    train_model,
    evaluate_model,
    predict_next_race,
)

# 1. Fetch data
df = fetch_race_results(2010, 2024)

# 2. Build features
df = build_features(df)

# 3. Encode
df, encoders = encode_features(df)

# 4. Split
X_train, y_train, X_test, y_test = split_data(df, test_start_year=2023)

# 5. Train
model = train_model(X_train, y_train, X_test, y_test)

# 6. Evaluate
evaluate_model(model, X_test, y_test)

# 7. Predict
grid = [
    {"driver": "max_verstappen", "grid": 1, "constructor": "red_bull"},
    {"driver": "leclerc",        "grid": 2, "constructor": "ferrari"},
    {"driver": "hamilton",       "grid": 3, "constructor": "mercedes"},
]
results = predict_next_race(model, encoders, "monaco", grid, df)
print(results)
```

---

## 🌐 Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"**
5. Select your repo, set **Main file path** to `app.py`
6. Click **Deploy** — done in ~2 minutes

Your app gets a permanent public URL like `https://your-app.streamlit.app`.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover feature engineering correctness, absence of data leakage, and that all required feature columns are present and NaN-free.

Tests also run automatically on every push via **GitHub Actions** (see `.github/workflows/ci.yml`).

---

## 📡 Data Source

Race data is fetched from the **[Jolpica F1 API](https://github.com/jolpica/jolpica-f1)** — a free, open-source API maintained by volunteers.

- Base URL: `https://api.jolpi.ca/ergast/f1`
- Rate limits: 4 requests/second, 500 requests/hour
- The pipeline adds a 0.3s delay between requests to be a responsible consumer

> Jolpica is run by volunteers at ~$45/month in hosting costs. Consider [donating](https://ko-fi.com/jolpica) if this project is useful to you.

---

## 🔮 Ideas for Improvement

- [ ] Add qualifying lap time features (Q1/Q2/Q3) via `fetch_qualifying()`
- [ ] Add weather data (rain dramatically changes race outcomes)
- [ ] Add tire strategy features (soft/medium/hard compound selection)
- [ ] Build an Elo rating system for dynamic driver skill scoring
- [ ] Add SHAP values for per-prediction explainability
- [ ] Add safety car probability features per circuit
- [ ] Hyperparameter tuning with Optuna

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas / numpy | Data manipulation |
| XGBoost | Gradient boosting classifier |
| scikit-learn | Preprocessing and evaluation |
| Streamlit | Web app framework |
| matplotlib | Visualisations |
| Jolpica F1 API | Race data source |
| GitHub Actions | Continuous integration |
| Streamlit Cloud | Free deployment |

---

## 👤 Author

**Vishnu** — CS Graduate, AI Trainer & Data Analyst  
Portfolio: [vish158.github.io](https://vish158.github.io)

---

## 📄 License

[MIT License](LICENSE) — free to use, modify, and distribute.
