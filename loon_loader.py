"""
Load and sample Loon stratospheric flight telemetry.
"""

import numpy as np
import pandas as pd


_MIN_HAPS_ALT_M = 15_000.0
_MAX_HAPS_ALT_M = 22_000.0


def load_loon_flights(
    csv_path,
    min_alt_m=_MIN_HAPS_ALT_M,
    max_alt_m=_MAX_HAPS_ALT_M,
):
    """Load the Loon telemetry and keep samples in the HAPS altitude range."""
    usecols = [
        "flight_id",
        "latitude",
        "longitude",
        "altitude",
        "temperature",
        "pressure",
        "velocity_u",
        "velocity_v",
        "solar_elevation",
        "is_daytime",
    ]

    df = pd.read_csv(csv_path, usecols=usecols)

    df = df.dropna(
        subset=[
            "altitude",
            "temperature",
            "pressure",
            "velocity_u",
            "velocity_v",
        ]
    )

    df = df[
        (df["altitude"] >= min_alt_m)
        & (df["altitude"] <= max_alt_m)
    ]

    out = pd.DataFrame(
        {
            "flight_id": df["flight_id"].values,
            "latitude": df["latitude"].values,
            "longitude": df["longitude"].values,
            "altitude_km": df["altitude"].values / 1000.0,
            "temperature_c": df["temperature"].values - 273.15,
            "pressure_hpa": df["pressure"].values,
            "drift_speed_mps": np.sqrt(
                df["velocity_u"].values ** 2
                + df["velocity_v"].values ** 2
            ),
            "drift_heading_deg": (
                np.degrees(
                    np.arctan2(
                        df["velocity_u"].values,
                        df["velocity_v"].values,
                    )
                )
                % 360.0
            ),
            "solar_elevation_deg": df["solar_elevation"].values,
            "is_daytime": df["is_daytime"].values.astype(bool),
        }
    ).reset_index(drop=True)

    return out


class LoonScenarioSampler:
    """Sample HAPS scenarios directly from the Loon telemetry table."""

    def __init__(self, loon_df):
        if len(loon_df) == 0:
            raise ValueError(
                "loon_df is empty after filtering; "
                "check the input file and altitude band"
            )

        self.df = loon_df.reset_index(drop=True)
        self._n = len(self.df)

    @classmethod
    def from_csv(cls, csv_path, **kwargs):
        return cls(load_loon_flights(csv_path, **kwargs))

    def sample(self, n, rng=None):
        """Draw n telemetry rows with replacement."""
        if rng is None:
            rng = np.random.default_rng()

        idx = rng.integers(0, self._n, size=n)
        rows = self.df.iloc[idx]

        return {
            "haps_altitude_km": rows["altitude_km"].values.copy(),
            "ambient_temperature_c": rows["temperature_c"].values.copy(),
            "ambient_pressure_hpa": rows["pressure_hpa"].values.copy(),
            "haps_drift_mps": rows["drift_speed_mps"].values.copy(),
            "haps_drift_heading_deg": rows[
                "drift_heading_deg"
            ].values.copy(),
            "haps_latitude_deg": rows["latitude"].values.copy(),
            "haps_longitude_deg": rows["longitude"].values.copy(),
            "abs_latitude_deg": np.abs(
                rows["latitude"].values.copy()
            ),
            "solar_elevation_deg": rows[
                "solar_elevation_deg"
            ].values.copy(),
            "is_daytime": rows["is_daytime"].values.copy(),
            "source_flight_id": rows["flight_id"].values.copy(),
        }

    def summary(self):
        """Return a short summary of the loaded flight data."""
        d = self.df

        lines = [
            f"Loon scenario pool: {len(d):,} rows from "
            f"{d['flight_id'].nunique()} flights",
            f"  altitude_km:   {d.altitude_km.min():.2f} - "
            f"{d.altitude_km.max():.2f} "
            f"(mean {d.altitude_km.mean():.2f})",
            f"  temperature_c: {d.temperature_c.min():.1f} - "
            f"{d.temperature_c.max():.1f} "
            f"(mean {d.temperature_c.mean():.1f})",
            f"  pressure_hpa:  {d.pressure_hpa.min():.1f} - "
            f"{d.pressure_hpa.max():.1f} "
            f"(mean {d.pressure_hpa.mean():.1f})",
            f"  drift_speed:   {d.drift_speed_mps.min():.2f} - "
            f"{d.drift_speed_mps.max():.2f} m/s "
            f"(mean {d.drift_speed_mps.mean():.2f})",
        ]

        return "\n".join(lines)
