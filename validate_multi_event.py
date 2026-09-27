"""

-----------------------

validate_multi_event.py

Multi event rain attenuation validation using CML and rain gauge

data from the CoMMon.

"""

import argparse

import os

import numpy as np

import pandas as pd

from itur.models import itu838

def gammaR(R, f_ghz, tau_deg=45.0):

    """Rain specific attenuation in dB/km."""

    gamma = itu838.rain_specific_attenuation(

        R=R, f=f_ghz, el=45.0, tau=tau_deg

    )

    return gamma.value if hasattr(gamma, "value") else float(gamma)

def load_cml_path_loss(cml_dat_path):

    df = pd.read_csv(cml_dat_path, sep=r"\s+", engine="python")

    df.columns = [str(c).strip() for c in df.columns]

    if "Time" not in df.columns:

        raise ValueError(f"There was no time column found in {cml_dat_path}")

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    df = df.dropna(subset=["Time"])

    df["H_loss"] = pd.to_numeric(df["Tx_H"], errors="coerce") - pd.to_numeric(

        df["Rx_H"], errors="coerce"

    )

    df["V_loss"] = pd.to_numeric(df["Tx_V"], errors="coerce") - pd.to_numeric(

        df["Rx_V"], errors="coerce"

    )

    df["path_loss"] = df[["H_loss", "V_loss"]].mean(axis=1)

    df = df.set_index("Time")

    return df["path_loss"].resample("1min").mean()

def load_rain_gauge(rain_gauge_dat_path):

    df = pd.read_csv(rain_gauge_dat_path, sep=r"\s+", engine="python")

    df.columns = [str(c).strip() for c in df.columns]

    if "Time" not in df.columns:

        raise ValueError(f"There was no time column found in {rain_gauge_dat_path}")

    rain_col = next(

        c for c in df.columns

        if "rain" in c.lower()

    )

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce")

    df = df.dropna(subset=["Time", rain_col])

    df = df.set_index("Time")

    return df[rain_col].resample("1min").mean()

def validate_one_event(cml_path_loss, rain_rate, freq_ghz=38.0, path_km=1.85):

    baseline = cml_path_loss.median()

    measured_excess = (cml_path_loss - baseline).clip(lower=0.0)

    data = pd.concat(

        [measured_excess.rename("measured_excess"),

         rain_rate.rename("rain_rate")],

        axis=1

    ).dropna()

    data = data[data["rain_rate"] > 0].copy()

    data["gamma_h"] = data["rain_rate"].apply(

        lambda R: gammaR(R, freq_ghz, tau_deg=0.0)

    )

    data["gamma_v"] = data["rain_rate"].apply(

        lambda R: gammaR(R, freq_ghz, tau_deg=90.0)

    )

    data["predicted_excess"] = (

        0.5  (data["gamma_h"] + data["gamma_v"])  path_km

    )

    data["error"] = data["predicted_excess"] - data["measured_excess"]

    return data

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--extracted-dir", required=True)

    parser.add_argument("--events-csv", required=True)

    parser.add_argument("--freq-ghz", type=float, default=38.0)

    parser.add_argument("--path-km", type=float, default=1.85)

    parser.add_argument("--out", default="multi_event_validation.csv")

    parser.add_argument("--summary-out", default="multi_event_summary.csv")

    args = parser.parse_args()

    events = pd.read_csv(args.events_csv)

    all_results = []

    summaries = []

    for _, event in events.iterrows():

        date = str(event["Date"]).zfill(8)

        gauge = str(event["Gauge"])

        cml_path = os.path.join(

            args.extracted_dir,

            date,

            "CML_20110427.dat"

        )

        rain_path = os.path.join(

            args.extracted_dir,

            date,

            f"{gauge}.dat"

        )

        if not os.path.exists(cml_path) or not os.path.exists(rain_path):

            print(f"Skipping {date}: missing CML or rain-gauge file")

            continue

        try:

            cml = load_cml_path_loss(cml_path)

            rain = load_rain_gauge(rain_path)

            result = validate_one_event(

                cml,

                rain,

                freq_ghz=args.freq_ghz,

                path_km=args.path_km,

            )

        except Exception as e:

            print(f"Skipping {date}: {e}")

            continue

        result["event_date"] = date

        result["gauge"] = gauge

        all_results.append(result)

        if len(result):

            measured_peak = result["measured_excess"].max()

            predicted_peak = result["predicted_excess"].max()

            errors = result["error"]

            summaries.append({

                "event_date": date,

                "gauge": gauge,

                "samples": len(result),

                "peak_rain_mm_h": result["rain_rate"].max(),

                "measured_peak_db": measured_peak,

                "predicted_peak_db": predicted_peak,

                "mae_db": np.mean(np.abs(errors)),

                "rmse_db": np.sqrt(np.mean(errors ** 2)),

                "correlation": result["measured_excess"].corr(

                    result["predicted_excess"]

                ),

            })

    if not all_results:

        raise RuntimeError("No events were successfully processed.")

    combined = pd.concat(all_results, ignore_index=True)

    summary = pd.DataFrame(summaries)

    combined.to_csv(args.out, index=False)

    summary.to_csv(args.summary_out, index=False)

    print(f"Saved event results to {args.out}")

    print(f"Saved event summary to {args.summary_out}")

    pooled_error = combined["error"]

    pooled_mae = np.mean(np.abs(pooled_error))

    pooled_rmse = np.sqrt(np.mean(pooled_error ** 2))

    pooled_corr = combined["measured_excess"].corr(

        combined["predicted_excess"]

    )

    print()

    print("Pooled results:")

    print(f"  Events:       {len(summary)}")

    print(f"  Samples:      {len(combined)}")

    print(f"  MAE:          {pooled_mae:.3f} dB")

    print(f"  RMSE:         {pooled_rmse:.3f} dB")

    print(f"  Correlation:  {pooled_corr:.3f}")

    print()

    print(summary.sort_values("event_date").to_string(index=False))

if name == "__main__":

    main()

