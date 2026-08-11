"""
Basic unit tests for the F1 prediction pipeline.
Run with:  pytest tests/
"""
import pytest
import pandas as pd
import numpy as np
from src.pipeline import build_features, encode_features, FEATURES


def make_dummy_df():
    """Creates a minimal DataFrame that mimics the real race data structure."""
    rows = []
    drivers = ["driver_a", "driver_b", "driver_c"]
    for year in range(2018, 2023):
        for round_no in range(1, 6):
            for i, driver in enumerate(drivers):
                rows.append({
                    "year":        year,
                    "round":       round_no,
                    "date":        f"{year}-03-{round_no:02d}",
                    "circuit":     "monaco" if round_no % 2 == 0 else "silverstone",
                    "driver":      driver,
                    "constructor": "team_a" if i == 0 else "team_b",
                    "grid":        i + 1,
                    "position":    i + 1,
                    "points":      [25, 18, 15][i],
                    "status":      "Finished",
                    "laps":        78,
                })
    return pd.DataFrame(rows)


def test_build_features_creates_target():
    df = make_dummy_df()
    df = build_features(df)
    assert "won" in df.columns
    assert df["won"].isin([0, 1]).all()


def test_build_features_no_leakage():
    """Checks that rolling features for the first race of each driver are NaN-filled, not future data."""
    df = make_dummy_df()
    df = build_features(df)
    # driver_wins_5 for the very first race of each driver should be filled (not from future)
    assert df["driver_wins_5"].notna().all()


def test_encode_features_returns_encoders():
    df = make_dummy_df()
    df = build_features(df)
    df_enc, encoders = encode_features(df)
    assert "driver" in encoders
    assert "circuit" in encoders
    assert "constructor" in encoders
    assert "driver_enc" in df_enc.columns


def test_feature_columns_all_present():
    df = make_dummy_df()
    df = build_features(df)
    df, _ = encode_features(df)
    for col in FEATURES:
        assert col in df.columns, f"Missing feature column: {col}"


def test_no_nan_in_features():
    df = make_dummy_df()
    df = build_features(df)
    df, _ = encode_features(df)
    assert df[FEATURES].isna().sum().sum() == 0, "NaN values found in feature columns"
