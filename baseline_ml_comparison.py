"""
baseline_ml_comparison.py
---------------------------
Baseline machine-learning demonstration for the HAPS dataset, designed to
answer a concrete question the dataset paper needs to address: does
sourcing platform geometry (altitude, drift) from real flight telemetry,
rather than independent synthetic distributions, measurably change
downstream model behavior compared to a purely synthetic dataset built
with the same propagation physics and the same random seed for every
other variable?

Experiment design
------------------
Two datasets are generated from the same code, same seed (42), and same
propagation physics. The ONLY difference is the source of HAPS altitude,
drift speed, and drift heading:
  - haps_dataset_real_2M.csv:        resampled from real Loon telemetry
  - haps_dataset_synthetic_2M.csv:   independent uniform/Gaussian draws

For each dataset, we train a Random Forest baseline to predict two
representative targets:
  - snr_db (regression)
  - outage (binary classification)

from a feature set that excludes any field directly derived from the
labels themselves (i.e. we predict from "input" features: geometry,
environment, weather, system config, mobility -- not from rx_power_dbm,
fspl_db, etc., which would make the prediction task trivial).

We report three things:
  1. In-distribution performance: train and test on the SAME dataset
     (the usual ML benchmark number).
  2. Cross-distribution performance: train on one dataset, test on the
     OTHER. If the real-data and synthetic-data platform geometry
     distributions are genuinely different, a model trained on one
     should show a measurable performance drop when evaluated on the
     other -- this is the direct test of whether the real-data layer
     "matters" for a downstream model, rather than an assumption.
  3. Feature importance for haps_altitude_km and haps_drift_mps
     specifically, to see how much the model relies on exactly the
     fields that differ between the two datasets.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
)


FEATURE_COLS = [
    "haps_altitude_km", "ground_distance_km", "slant_range_km", "elevation_deg",
    "azimuth_deg", "environment", "weather_state", "rain_rate_mm_per_hr",
    "cloud_lwp_kg_per_m2", "temperature_c", "humidity_pct", "carrier_freq_ghz",
    "bandwidth_mhz", "tx_power_dbm", "tx_antenna_gain_dbi", "rx_antenna_gain_dbi",
    "noise_figure_db", "is_mobile", "user_speed_mps", "haps_drift_mps",
    "haps_drift_heading_deg", "is_los",
]
CATEGORICAL_COLS = ["environment", "weather_state"]


def load_and_prepare(csv_path, n_rows=None):
    df = pd.read_csv(csv_path, nrows=n_rows)
    df = df[FEATURE_COLS + ["snr_db", "outage"]].copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category").cat.codes
    df["is_mobile"] = df["is_mobile"].astype(int)
    df["is_los"] = df["is_los"].astype(int)
    return df


def evaluate_regression(model, X, y, label):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    print(f"  [{label}] SNR regression: MAE={mae:.3f} dB, RMSE={rmse:.3f} dB, R2={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_classification(model, X, y, label):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    auc = roc_auc_score(y, proba)
    print(f"  [{label}] Outage classification: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    return {"accuracy": acc, "f1": f1, "auc": auc}


def run_experiment(real_csv, synthetic_csv, n_rows=500_000, random_state=0):
    print("Loading datasets...")
    real_df = load_and_prepare(real_csv, n_rows=n_rows)
    synth_df = load_and_prepare(synthetic_csv, n_rows=n_rows)

    feature_cols_encoded = [c for c in real_df.columns if c not in ("snr_db", "outage")]

    real_train, real_test = train_test_split(real_df, test_size=0.2, random_state=random_state)
    synth_train, synth_test = train_test_split(synth_df, test_size=0.2, random_state=random_state)

    results = {}

    print("\n=== Training on REAL-platform-geometry dataset ===")
    reg_real = RandomForestRegressor(n_estimators=60, max_depth=14, n_jobs=-1, random_state=random_state)
    reg_real.fit(real_train[feature_cols_encoded], real_train["snr_db"])
    clf_real = RandomForestClassifier(n_estimators=60, max_depth=14, n_jobs=-1, random_state=random_state)
    clf_real.fit(real_train[feature_cols_encoded], real_train["outage"])

    print("\n=== Training on SYNTHETIC-platform-geometry dataset ===")
    reg_synth = RandomForestRegressor(n_estimators=60, max_depth=14, n_jobs=-1, random_state=random_state)
    reg_synth.fit(synth_train[feature_cols_encoded], synth_train["snr_db"])
    clf_synth = RandomForestClassifier(n_estimators=60, max_depth=14, n_jobs=-1, random_state=random_state)
    clf_synth.fit(synth_train[feature_cols_encoded], synth_train["outage"])

    print("\n=== In-distribution evaluation (train and test on same dataset) ===")
    results["real_on_real"] = evaluate_regression(reg_real, real_test[feature_cols_encoded], real_test["snr_db"], "real->real")
    results["real_on_real_clf"] = evaluate_classification(clf_real, real_test[feature_cols_encoded], real_test["outage"], "real->real")
    results["synth_on_synth"] = evaluate_regression(reg_synth, synth_test[feature_cols_encoded], synth_test["snr_db"], "synth->synth")
    results["synth_on_synth_clf"] = evaluate_classification(clf_synth, synth_test[feature_cols_encoded], synth_test["outage"], "synth->synth")

    print("\n=== Cross-distribution evaluation (train on one, test on the other) ===")
    results["real_on_synth"] = evaluate_regression(reg_real, synth_test[feature_cols_encoded], synth_test["snr_db"], "real->synth")
    results["real_on_synth_clf"] = evaluate_classification(clf_real, synth_test[feature_cols_encoded], synth_test["outage"], "real->synth")
    results["synth_on_real"] = evaluate_regression(reg_synth, real_test[feature_cols_encoded], real_test["snr_db"], "synth->real")
    results["synth_on_real_clf"] = evaluate_classification(clf_synth, real_test[feature_cols_encoded], real_test["outage"], "synth->real")

    print("\n=== Feature importance for platform-geometry fields ===")
    imp_real = pd.Series(reg_real.feature_importances_, index=feature_cols_encoded).sort_values(ascending=False)
    imp_synth = pd.Series(reg_synth.feature_importances_, index=feature_cols_encoded).sort_values(ascending=False)
    print("Real-data model, top 8 features (SNR regression):")
    print(imp_real.head(8))
    print("\nSynthetic-data model, top 8 features (SNR regression):")
    print(imp_synth.head(8))

    print(f"\nhaps_altitude_km importance: real-model={imp_real.get('haps_altitude_km', 0):.4f}, "
          f"synth-model={imp_synth.get('haps_altitude_km', 0):.4f}")
    print(f"haps_drift_mps importance:   real-model={imp_real.get('haps_drift_mps', 0):.4f}, "
          f"synth-model={imp_synth.get('haps_drift_mps', 0):.4f}")

    return results, imp_real, imp_synth


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", default="haps_dataset_real_2M.csv")
    parser.add_argument("--synthetic-csv", default="haps_dataset_synthetic_2M.csv")
    parser.add_argument("--n-rows", type=int, default=500_000)
    args = parser.parse_args()

    run_experiment(args.real_csv, args.synthetic_csv, n_rows=args.n_rows)
