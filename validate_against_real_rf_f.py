"""
Compare the rain attenuation model with measured CML data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import itur.models.itu838 as itu838


def _gamma_R(rain_rate, freq_ghz, tau_deg=45.0):
    """Return rain specific attenuation in dB/km."""
    value = itu838.rain_specific_attenuation(
        R=rain_rate,
        f=freq_ghz,
        el=45.0,
        tau=tau_deg,
    )

    if hasattr(value, "value"):
        return float(value.value)

    return float(value)


def load_real_cml_path_loss(cml_dat_path):
    """Load and resample the measured CML path-loss data."""
    df = pd.read_csv(
        cml_dat_path,
        sep=r"\s+",
        comment="#",
    )

    df.columns = [str(c).strip() for c in df.columns]

    df["Time"] = pd.to_datetime(
        df["Time"],
        errors="coerce",
    )

    df = df.dropna(subset=["Time"])

    df["path_loss_h_db"] = (
        df["Tx H [dBm]"] - df["Rx H [dBm]"]
    )
    df["path_loss_v_db"] = (
        df["Tx V [dBm]"] - df["Rx V [dBm]"]
    )

    path_loss = df.set_index("Time")[
        ["path_loss_h_db", "path_loss_v_db"]
    ].mean(axis=1)

    return path_loss.resample("1min").mean()


def load_real_rain_gauge(rain_gauge_dat_path):
    """Load the measured rain-rate series."""
    df = pd.read_csv(
        rain_gauge_dat_path,
        sep=r"\s+",
        comment="#",
    )

    df.columns = [str(c).strip() for c in df.columns]

    time_col = next(
        c for c in df.columns
        if c.lower() == "time"
    )

    rain_col = next(
        c for c in df.columns
        if "rain.rate" in c.lower()
    )

    df[time_col] = pd.to_datetime(
        df[time_col],
        errors="coerce",
    )

    df[rain_col] = pd.to_numeric(
        df[rain_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[time_col, rain_col]
    )

    return (
        df.set_index(time_col)[rain_col]
        .resample("1min")
        .mean()
    )


def run_validation(
    cml_path_loss_db,
    rain_rate_series,
    freq_ghz=38.0,
    path_length_km=1.85,
):
    """Compare measured and P.838-predicted rain attenuation."""
    baseline = cml_path_loss_db.median()

    measured_excess = (
        cml_path_loss_db - baseline
    ).clip(lower=0.0)

    data = pd.concat(
        [
            measured_excess.rename("measured_excess_db"),
            rain_rate_series.rename("rain_rate_mm_h"),
        ],
        axis=1,
    ).dropna()

    data = data[data["rain_rate_mm_h"] > 0].copy()

    data["gamma_h_db_km"] = data["rain_rate_mm_h"].apply(
        lambda r: _gamma_R(r, freq_ghz, tau_deg=0.0)
    )

    data["gamma_v_db_km"] = data["rain_rate_mm_h"].apply(
        lambda r: _gamma_R(r, freq_ghz, tau_deg=90.0)
    )

    data["predicted_excess_db"] = (
        0.5
        * (
            data["gamma_h_db_km"]
            + data["gamma_v_db_km"]
        )
        * path_length_km
    )

    data["error_db"] = (
        data["predicted_excess_db"]
        - data["measured_excess_db"]
    )

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cml-dir",
        required=True,
    )
    parser.add_argument(
        "--cml-file",
        default="CML_20110427.dat",
    )
    parser.add_argument(
        "--rain-gauge-file",
        default="RG02_2011-04-27.dat",
    )
    parser.add_argument(
        "--freq-ghz",
        type=float,
        default=38.0,
    )
    parser.add_argument(
        "--path-km",
        type=float,
        default=1.85,
    )
    parser.add_argument(
        "--out",
        default="real_rf_validation.csv",
    )

    args = parser.parse_args()

    data_dir = Path(args.cml_dir)

    cml_path = data_dir / args.cml_file
    rain_path = data_dir / args.rain_gauge_file

    print(f"CML file: {cml_path}")
    print(f"Rain gauge: {rain_path}")

    cml = load_real_cml_path_loss(cml_path)
    rain = load_real_rain_gauge(rain_path)

    result = run_validation(
        cml,
        rain,
        freq_ghz=args.freq_ghz,
        path_length_km=args.path_km,
    )

    if result.empty:
        raise RuntimeError(
            "No overlapping rain and CML measurements were found."
        )

    measured = result["measured_excess_db"].values
    predicted = result["predicted_excess_db"].values

    error = predicted - measured

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    if len(result) > 1:
        corr = np.corrcoef(
            measured,
            predicted,
        )[0, 1]
    else:
        corr = np.nan

    print()
    print(f"Samples: {len(result):,}")
    print(f"MAE:     {mae:.3f} dB")
    print(f"RMSE:    {rmse:.3f} dB")
    print(f"Corr.:   {corr:.3f}")
    print(
        f"Measured peak:  "
        f"{measured.max():.3f} dB"
    )
    print(
        f"Predicted peak: "
        f"{predicted.max():.3f} dB"
    )

    result.to_csv(args.out, index=True)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()