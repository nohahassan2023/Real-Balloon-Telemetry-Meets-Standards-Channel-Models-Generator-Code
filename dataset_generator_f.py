"""
dataset_generator.py

Generate a Monte Carlo HAPS channel dataset using HAPS flight telemetry
from the Loon stratospheric dataset together with the geometry, propagation,
and link-budget models.

When Loon data are provided, HAPS altitude, drift speed, drift heading, and
position are sampled from the telemetry data. Weather and user-side
parameters remain synthetic because the Loon dataset does not provide
ground-level weather or user mobility data.

Usage
-----
python dataset_generator.py --n 1000000 --out haps_dataset.csv \
    --loon-csv loon_data/loon-flights-2021Q2_csv
"""

import argparse
import time
import numpy as np
import pandas as pd

import geometry
import propagation
import link_budget
import loon_loader


ENVIRONMENTS = ["urban", "suburban", "rural"]
WEATHER_STATES = ["clear", "light_rain", "heavy_rain", "fog"]
CARRIER_FREQS_GHZ = [2.0, 3.5, 6.0, 28.0]


def sample_weather(n, rng):
    """
    Sample weather conditions and the associated environmental variables.

    Rain rates for the two rain states are drawn from log-normal
    distributions. Temperature and humidity are sampled independently,
    except for the higher humidity imposed for fog.
    """
    weather = rng.choice(
        WEATHER_STATES,
        size=n,
        p=[0.55, 0.25, 0.10, 0.10]
    )

    rain_rate = np.zeros(n)
    cloud_lwp = np.zeros(n)
    temperature_c = rng.normal(15.0, 8.0, size=n)
    humidity_pct = rng.uniform(30.0, 95.0, size=n)

    is_light = weather == "light_rain"
    is_heavy = weather == "heavy_rain"
    is_fog = weather == "fog"
    is_clear = weather == "clear"

    rain_rate[is_light] = rng.lognormal(
        mean=np.log(3.0), sigma=0.5, size=is_light.sum()
    )
    rain_rate[is_heavy] = rng.lognormal(
        mean=np.log(20.0), sigma=0.6, size=is_heavy.sum()
    )
    rain_rate = np.clip(rain_rate, 0.0, 150.0)

    cloud_lwp[is_clear] = rng.uniform(
        0.0, 0.1, size=is_clear.sum()
    )
    cloud_lwp[is_light] = rng.uniform(
        0.1, 0.5, size=is_light.sum()
    )
    cloud_lwp[is_heavy] = rng.uniform(
        0.3, 0.8, size=is_heavy.sum()
    )
    cloud_lwp[is_fog] = rng.uniform(
        0.05, 0.3, size=is_fog.sum()
    )

    humidity_pct[is_fog] = rng.uniform(
        85.0, 100.0, size=is_fog.sum()
    )

    return weather, rain_rate, cloud_lwp, temperature_c, humidity_pct


def generate_dataset(
    n_samples,
    seed=42,
    loon_csv_path=None,
    loon_sampler=None,
    min_elevation_deg=10.0,
    bandwidth_options_mhz=(5, 10, 20, 40, 100),
    tx_power_range_dbm=(30.0, 43.0),
    interference_fraction=0.3,
    haps_altitude_range_km=(17.0, 25.0)
):
    """
    Generate the HAPS channel dataset.

    Parameters
    ----------
    n_samples : int
        Number of channel realizations.
    seed : int
        Random seed.
    loon_csv_path : str, optional
        Path to the Loon telemetry CSV. If provided, HAPS platform
        parameters are sampled from the telemetry data.
    loon_sampler : loon_loader.LoonScenarioSampler, optional
        Pre-built Loon sampler. Used instead of loon_csv_path when supplied.
    haps_altitude_range_km : tuple
        Altitude range used when no Loon data are provided.

    Returns
    -------
    pandas.DataFrame
        Generated HAPS channel dataset.
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    use_real_platform_data = (
        loon_sampler is not None or loon_csv_path is not None
    )

    if use_real_platform_data and loon_sampler is None:
        loon_sampler = loon_loader.LoonScenarioSampler.from_csv(
            loon_csv_path
        )

    # System and platform parameters
    if use_real_platform_data:
        real = loon_sampler.sample(n, rng=rng)

        haps_altitude_km = real["haps_altitude_km"]
        haps_drift_mps = real["haps_drift_mps"]
        haps_drift_heading_deg = real["haps_drift_heading_deg"]
        haps_latitude_deg = real["haps_latitude_deg"]
        haps_longitude_deg = real["haps_longitude_deg"]
        source_flight_id = real["source_flight_id"]
    else:
        haps_altitude_km = rng.uniform(
            *haps_altitude_range_km, size=n
        )
        haps_drift_mps = rng.uniform(0.0, 15.0, size=n)
        haps_drift_heading_deg = rng.uniform(0.0, 360.0, size=n)

        # No platform location is available in synthetic mode.
        haps_latitude_deg = np.full(n, np.nan)
        haps_longitude_deg = np.full(n, np.nan)
        source_flight_id = np.full(
            n, "synthetic", dtype=object
        )

    environment = rng.choice(
        ENVIRONMENTS,
        size=n,
        p=[0.3, 0.35, 0.35]
    )
    carrier_freq_ghz = rng.choice(CARRIER_FREQS_GHZ, size=n)
    bandwidth_mhz = rng.choice(
        bandwidth_options_mhz, size=n
    ).astype(float)

    tx_power_dbm = rng.uniform(
        *tx_power_range_dbm, size=n
    )
    tx_antenna_gain_dbi = rng.uniform(
        15.0, 35.0, size=n
    )
    rx_antenna_gain_dbi = rng.uniform(
        0.0, 10.0, size=n
    )
    noise_figure_db = rng.uniform(
        1.5, 5.0, size=n
    )

    # Geometry
    ground_distance_km, azimuth_deg = geometry.sample_user_positions(
        n,
        haps_altitude_km,
        min_elevation_deg=min_elevation_deg,
        rng=rng
    )

    slant_km, elevation_deg = geometry.slant_range_km(
        haps_altitude_km,
        ground_distance_km
    )

    # Mobility
    # HAPS motion comes from telemetry when Loon data are available.
    # User mobility is sampled synthetically.
    is_mobile = rng.uniform(size=n) < 0.5

    user_speed_mps = np.where(
        is_mobile,
        rng.uniform(0.5, 33.0, size=n),
        0.0
    )

    relative_heading_deg = rng.uniform(
        0.0, 360.0, size=n
    )

    doppler_hz = geometry.doppler_shift_hz(
        carrier_freq_ghz * 1e9,
        haps_drift_mps,
        user_speed_mps,
        elevation_deg,
        relative_heading_deg
    )

    # Weather
    (
        weather_state,
        rain_rate_mm_hr,
        cloud_lwp_kg_m2,
        temperature_c,
        humidity_pct
    ) = sample_weather(n, rng)

    # LoS state, shadow fading, and small-scale fading
    is_los = np.zeros(n, dtype=bool)
    shadow_db = np.zeros(n)
    shadow_sigma = np.zeros(n)
    small_scale_db = np.zeros(n)
    rician_k_db = np.zeros(n)

    for env in ENVIRONMENTS:
        mask = environment == env

        if not mask.any():
            continue

        is_los[mask] = propagation.sample_los_state(
            elevation_deg[mask],
            env,
            rng=rng
        )

        s_db, sigma = propagation.shadow_fading_db(
            elevation_deg[mask],
            env,
            is_los[mask],
            rng=rng
        )

        shadow_db[mask] = s_db
        shadow_sigma[mask] = sigma

        small_scale_db[mask] = (
            propagation.sample_small_scale_fading_db(
                elevation_deg[mask],
                env,
                is_los[mask],
                rng=rng
            )
        )

        rician_k_db[mask] = propagation.rician_k_factor_db(
            elevation_deg[mask],
            env
        )

    # Propagation losses
    fspl_db = propagation.free_space_path_loss_db(
        slant_km,
        carrier_freq_ghz
    )

    gas_atten_db = propagation.gaseous_attenuation_db(
        elevation_deg,
        carrier_freq_ghz,
        temperature_c,
        humidity_pct
    )

    cloud_atten_db = propagation.cloud_attenuation_db(
        elevation_deg,
        carrier_freq_ghz,
        cloud_lwp_kg_m2,
        temperature_c
    )

    rain_atten_db = propagation.rain_attenuation_db(
        elevation_deg,
        carrier_freq_ghz,
        rain_rate_mm_hr,
        haps_altitude_km
    )

    # Link budget
    rx_power_dbm = link_budget.received_power_dbm(
        tx_power_dbm,
        tx_antenna_gain_dbi,
        rx_antenna_gain_dbi,
        fspl_db,
        gas_atten_db,
        cloud_atten_db,
        rain_atten_db,
        shadow_db,
        small_scale_db,
        misc_loss_db=2.0
    )

    noise_dbm = link_budget.thermal_noise_power_dbm(
        bandwidth_mhz * 1e6,
        noise_figure_db
    )

    snr = link_budget.snr_db(
        rx_power_dbm,
        noise_dbm
    )

    # Simple co-channel interference model
    has_interference = (
        rng.uniform(size=n) < interference_fraction
    )

    inr_db = np.where(
        has_interference,
        rng.uniform(-5.0, 15.0, size=n),
        -99.0
    )

    interference_dbm = noise_dbm + inr_db

    sinr = link_budget.sinr_db(
        rx_power_dbm,
        noise_dbm,
        interference_power_dbm=interference_dbm
    )

    spectral_eff = link_budget.shannon_capacity_bps_hz(sinr)
    capacity_mbps = spectral_eff * bandwidth_mhz

    ber = link_budget.approximate_ber_qpsk(snr)
    outage = link_budget.outage_indicator(
        sinr,
        sinr_threshold_db=0.0
    )

    # Assemble the dataset
    df = pd.DataFrame({
        # Geometry
        "haps_altitude_km": haps_altitude_km,
        "haps_latitude_deg": haps_latitude_deg,
        "haps_longitude_deg": haps_longitude_deg,
        "ground_distance_km": ground_distance_km,
        "slant_range_km": slant_km,
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_deg,

        # Environment and weather
        "environment": environment,
        "weather_state": weather_state,
        "rain_rate_mm_per_hr": rain_rate_mm_hr,
        "cloud_lwp_kg_per_m2": cloud_lwp_kg_m2,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,

        # System configuration
        "carrier_freq_ghz": carrier_freq_ghz,
        "bandwidth_mhz": bandwidth_mhz,
        "tx_power_dbm": tx_power_dbm,
        "tx_antenna_gain_dbi": tx_antenna_gain_dbi,
        "rx_antenna_gain_dbi": rx_antenna_gain_dbi,
        "noise_figure_db": noise_figure_db,

        # Mobility
        "is_mobile": is_mobile,
        "user_speed_mps": user_speed_mps,
        "haps_drift_mps": haps_drift_mps,
        "haps_drift_heading_deg": haps_drift_heading_deg,
        "doppler_hz": doppler_hz,

        # Channel state
        "is_los": is_los,
        "rician_k_db": rician_k_db,
        "shadow_fading_db": shadow_db,
        "shadow_fading_sigma_db": shadow_sigma,
        "small_scale_fading_db": small_scale_db,

        # Propagation losses
        "fspl_db": fspl_db,
        "gas_atten_db": gas_atten_db,
        "cloud_atten_db": cloud_atten_db,
        "rain_atten_db": rain_atten_db,

        # Interference
        "has_interference": has_interference,
        "interference_dbm": np.where(
            has_interference,
            interference_dbm,
            np.nan
        ),

        # Link-level outputs
        "rx_power_dbm": rx_power_dbm,
        "noise_power_dbm": noise_dbm,
        "snr_db": snr,
        "sinr_db": sinr,
        "spectral_efficiency_bps_hz": spectral_eff,
        "capacity_mbps": capacity_mbps,
        "ber_qpsk": ber,
        "outage": outage,

        # Platform data provenance
        "platform_data_source_flight_id": source_flight_id,
    })

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate HAPS channel dataset"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1_000_000,
        help="Number of samples"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--out",
        type=str,
        default="haps_dataset.csv"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "hdf5", "parquet"]
    )

    parser.add_argument(
        "--loon-csv",
        type=str,
        default=None,
        help=(
            "Path to the Loon telemetry CSV. If provided, "
            "HAPS altitude and drift parameters are sampled "
            "from the flight data."
        )
    )

    args = parser.parse_args()

    t0 = time.time()

    df = generate_dataset(
        args.n,
        seed=args.seed,
        loon_csv_path=args.loon_csv
    )

    t1 = time.time()

    print(
        f"Generated {len(df):,} samples in "
        f"{t1 - t0:.2f} s"
    )

    if args.loon_csv:
        print(
            "Platform parameters sampled from Loon flight data: "
            f"{args.loon_csv}"
        )

    if args.format == "csv":
        df.to_csv(args.out, index=False)
    elif args.format == "hdf5":
        df.to_hdf(
            args.out,
            key="haps_dataset",
            mode="w"
        )
    elif args.format == "parquet":
        df.to_parquet(args.out, index=False)

    print(
        f"Saved to {args.out} "
        f"({df.shape[0]:,} rows x {df.shape[1]} columns)"
    )

    print(df.describe(include="all").T)


if __name__ == "__main__":
    main()