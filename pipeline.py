"""
F1 Grand Prix Winner Prediction — Core Pipeline
================================================
Handles data fetching, feature engineering, model training, and prediction.
All functions are self-contained and can be called independently from a
Jupyter notebook or the Streamlit app.
"""

import time
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

BASE_URL = "https://api.jolpi.ca/ergast/f1"

FEATURES = [
    "grid", "driver_enc", "constructor_enc", "circuit_enc",
    "driver_avg_pos_5", "driver_wins_5", "constructor_avg_pos_5",
    "circuit_win_rate", "grid_advantage", "cumulative_points", "round",
]


# ─────────────────────────────────────────────
# STEP 1: Fetch data
# ─────────────────────────────────────────────

def fetch_race_results(season_start: int = 2010, season_end: int = 2024, delay: float = 0.3) -> pd.DataFrame:
    """
    Fetches race results for every season in the given range from the Jolpica F1 API.
    Includes retry logic and rate-limit-safe delays between requests.

    Args:
        season_start: First season to fetch (inclusive).
        season_end:   Last season to fetch (inclusive).
        delay:        Seconds to wait between API calls (default 0.3 stays well
                      under Jolpica's 4 req/sec limit).

    Returns:
        pd.DataFrame with one row per driver per race.
    """
    all_results = []

    for year in range(season_start, season_end + 1):
        url = f"{BASE_URL}/{year}/results.json?limit=1000"

        for attempt in range(3):  # retry up to 3 times
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    print(f"⚠️  Skipping {year} after 3 failures: {e}")
                    continue
                time.sleep(2 ** attempt)  # exponential back-off

        data = response.json()
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

        for race in races:
            circuit  = race["Circuit"]["circuitId"]
            round_no = int(race["round"])
            race_date = race["date"]

            for result in race["Results"]:
                raw_pos = result.get("position", "99")
                all_results.append({
                    "year":        year,
                    "round":       round_no,
                    "date":        race_date,
                    "circuit":     circuit,
                    "driver":      result["Driver"]["driverId"],
                    "constructor": result["Constructor"]["constructorId"],
                    "grid":        int(result.get("grid", 20)),
                    "position":    int(raw_pos) if raw_pos.isdigit() else 99,
                    "points":      float(result.get("points", 0)),
                    "status":      result.get("status", "Unknown"),
                    "laps":        int(result.get("laps", 0)),
                })

        print(f"  ✓ {year} — {len(races)} races")
        time.sleep(delay)

    df = pd.DataFrame(all_results)
    print(f"\n✅ Fetched {len(df):,} rows across {season_end - season_start + 1} seasons.")
    return df


def fetch_qualifying(season_start: int = 2010, season_end: int = 2024, delay: float = 0.3) -> pd.DataFrame:
    """
    Fetches qualifying results (Q1/Q2/Q3 lap times) for bonus features.

    Returns:
        pd.DataFrame with qualifying times per driver per round.
    """
    all_qual = []

    for year in range(season_start, season_end + 1):
        url = f"{BASE_URL}/{year}/qualifying.json?limit=1000"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Qualifying fetch failed for {year}: {e}")
            continue

        races = response.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for race in races:
            for q in race.get("QualifyingResults", []):
                all_qual.append({
                    "year":    year,
                    "round":   int(race["round"]),
                    "circuit": race["Circuit"]["circuitId"],
                    "driver":  q["Driver"]["driverId"],
                    "q1":      q.get("Q1", None),
                    "q2":      q.get("Q2", None),
                    "q3":      q.get("Q3", None),
                })
        time.sleep(delay)

    return pd.DataFrame(all_qual)


# ─────────────────────────────────────────────
# STEP 2: Feature engineering
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs all predictive features from raw race results.
    Uses shift(1) on every rolling calculation to prevent data leakage —
    the model never sees the outcome of the race it is trying to predict.

    Features created:
        won                  — target: 1 if driver finished P1, else 0
        driver_avg_pos_5     — driver's mean finishing position, last 5 races
        driver_wins_5        — driver's win count, last 5 races
        constructor_avg_pos_5— constructor's mean finishing position, last 5 races
        circuit_win_rate     — driver's historical win rate at this specific circuit
        grid_advantage       — 20 - grid_position (higher = better start)
        cumulative_points    — driver's running championship points total
    """
    df = df.sort_values(["driver", "year", "round"]).reset_index(drop=True)

    # Target
    df["won"] = (df["position"] == 1).astype(int)

    # Driver rolling form
    df["driver_avg_pos_5"] = (
        df.groupby("driver")["position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["driver_wins_5"] = (
        df.groupby("driver")["won"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )

    # Constructor rolling form
    df["constructor_avg_pos_5"] = (
        df.groupby("constructor")["position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Circuit-specific win rate
    df["circuit_win_rate"] = (
        df.groupby(["circuit", "driver"])["won"]
        .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(0)

    # Grid advantage (flipped so higher = better)
    df["grid_advantage"] = 20 - df["grid"]

    # Cumulative championship points
    df["cumulative_points"] = (
        df.groupby(["year", "driver"])["points"]
        .transform(lambda x: x.shift(1).cumsum())
    ).fillna(0)

    # Fill NaNs for a driver's very first appearance
    for col in ["driver_avg_pos_5", "driver_wins_5", "constructor_avg_pos_5"]:
        df[col] = df[col].fillna(df[col].median())

    return df


# ─────────────────────────────────────────────
# STEP 3: Encode categoricals
# ─────────────────────────────────────────────

def encode_features(df: pd.DataFrame):
    """
    Label-encodes driver, circuit, and constructor IDs into integers.

    Returns:
        (df_encoded, encoders_dict) — the enriched DataFrame and the fitted
        encoders (needed later when predicting a future race).
    """
    encoders = {
        "driver":      LabelEncoder(),
        "circuit":     LabelEncoder(),
        "constructor": LabelEncoder(),
    }
    df["driver_enc"]      = encoders["driver"].fit_transform(df["driver"])
    df["circuit_enc"]     = encoders["circuit"].fit_transform(df["circuit"])
    df["constructor_enc"] = encoders["constructor"].fit_transform(df["constructor"])
    return df, encoders


# ─────────────────────────────────────────────
# STEP 4: Train / test split
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_start_year: int = 2023):
    """
    Splits data chronologically — never randomly — to prevent leakage.
    Training uses all seasons before test_start_year; testing uses the rest.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    train = df[df["year"] <  test_start_year]
    test  = df[df["year"] >= test_start_year]

    X_train, y_train = train[FEATURES], train["won"]
    X_test,  y_test  = test[FEATURES],  test["won"]

    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Win rate — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
# STEP 5: Train model
# ─────────────────────────────────────────────

def train_model(X_train, y_train, X_test, y_test) -> XGBClassifier:
    """
    Trains an XGBoost classifier with class-imbalance correction.
    scale_pos_weight compensates for the fact that only ~5% of entries
    are race winners.

    Returns:
        Trained XGBClassifier model.
    """
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio → scale_pos_weight = {scale:.1f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    return model


# ─────────────────────────────────────────────
# STEP 6: Evaluate
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, threshold: float = 0.3) -> dict:
    """
    Evaluates model performance on held-out test data.

    Args:
        threshold: Decision boundary for converting probabilities to labels.
                   Lower than 0.5 because wins are rare — we'd rather catch
                   potential winners than miss them.

    Returns:
        dict with roc_auc, y_prob, and y_pred arrays.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    print(classification_report(y_test, y_pred, target_names=["No Win", "Win"]))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")
    return {"roc_auc": auc, "y_prob": y_prob, "y_pred": y_pred}


# ─────────────────────────────────────────────
# STEP 7: Predict next race
# ─────────────────────────────────────────────

def predict_next_race(
    model,
    encoders: dict,
    circuit_id: str,
    drivers_grid: list[dict],
    df_history: pd.DataFrame,
    round_num: int = 1,
) -> pd.DataFrame:
    """
    Generates win probabilities for every driver in an upcoming race.

    Args:
        model:        Trained XGBClassifier.
        encoders:     Dict of fitted LabelEncoders from encode_features().
        circuit_id:   Jolpica circuit ID string, e.g. 'monaco', 'silverstone'.
        drivers_grid: List of dicts: [{"driver": "...", "grid": 1, "constructor": "..."}, ...]
        df_history:   Full historical DataFrame (used to compute rolling features).
        round_num:    Race round number in the current season.

    Returns:
        DataFrame sorted by win_pct descending.
    """
    def safe_encode(encoder, value):
        return int(encoder.transform([value])[0]) if value in encoder.classes_ else -1

    rows = []
    for entry in drivers_grid:
        driver      = entry["driver"]
        grid        = int(entry["grid"])
        constructor = entry["constructor"]

        recent          = df_history[df_history["driver"] == driver].tail(5)
        circuit_history = df_history[
            (df_history["driver"] == driver) & (df_history["circuit"] == circuit_id)
        ]

        rows.append({
            "grid":                  grid,
            "driver_enc":            safe_encode(encoders["driver"], driver),
            "constructor_enc":       safe_encode(encoders["constructor"], constructor),
            "circuit_enc":           safe_encode(encoders["circuit"], circuit_id),
            "driver_avg_pos_5":      recent["position"].mean()  if len(recent) else 10.0,
            "driver_wins_5":         recent["won"].sum()        if len(recent) else 0.0,
            "constructor_avg_pos_5": recent["position"].mean()  if len(recent) else 10.0,
            "circuit_win_rate":      circuit_history["won"].mean() if len(circuit_history) else 0.0,
            "grid_advantage":        20 - grid,
            "cumulative_points":     recent["points"].sum()     if len(recent) else 0.0,
            "round":                 round_num,
        })

    pred_df = pd.DataFrame(rows)
    probs   = model.predict_proba(pred_df[FEATURES])[:, 1]

    results = pd.DataFrame({
        "driver":          [e["driver"] for e in drivers_grid],
        "constructor":     [e["constructor"] for e in drivers_grid],
        "grid":            [e["grid"] for e in drivers_grid],
        "win_probability": probs,
    }).sort_values("win_probability", ascending=False)

    total = results["win_probability"].sum()
    results["win_pct"] = (results["win_probability"] / total * 100).round(1)
    return results.reset_index(drop=True)
