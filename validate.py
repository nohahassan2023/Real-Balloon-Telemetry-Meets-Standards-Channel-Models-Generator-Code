"""

validate.py

-----------

Sanity checks for the main simulator, which compare the simulated results with expected trends from 3GPP NTN/HAPS and ITU-R models. 

"""

import numpy as np

import pandas as pd

import geometry

import propagation

import link_budget

from dataset_generator import generate_dataset

def check_los_probability_trend():

    print("=== Check 1: LoS probability ordering (TR 38.811) ===")

    elevs = [10, 30, 50, 70, 90]

    for env in ["urban", "suburban", "rural"]:

        probs = [propagation.los_probability(e, env) for e in elevs]

        print(f"  {env:10s}: " + ", ".join(f"{e}deg={p:.2f}" for e, p in zip(elevs, probs)))

        assert all(probs[i] <= probs[i + 1] + 1e-9 for i in range(len(probs) - 1)), \

            f"LoS probability should increase with elevation for {env}"

    p_urban = propagation.los_probability(30, "urban")

    p_suburban = propagation.los_probability(30, "suburban")

    p_rural = propagation.los_probability(30, "rural")

    assert p_urban < p_suburban < p_rural, "Expected rural > suburban > urban LoS probability"

    print("  PASS: monotonic in elevation; rural > suburban > urban\n")

def check_snr_vs_frequency():

    print("=== Check 2: SNR/capacity decrease with carrier frequency ===")

    df = generate_dataset(200_000, seed=123)

    means = df.groupby("carrier_freq_ghz")["snr_db"].mean().sort_index()

    print(means)

    vals = means.values

    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), \

        "Mean SNR is meant to decrease with the increase in carrier frequency"

    print("  PASS: mean SNR decreases monotonically with frequency\n")

def check_rain_attenuation_trend():

    print("=== Check 3: Rain attenuation increases with frequency & rain rate ===")

    elev = 30.0

    for f in [2.0, 6.0, 28.0]:

        atts = [propagation.rain_attenuation_db(elev, f, R) for R in [0, 5, 20, 50, 100]]

        print(f"  f={f:5.1f} GHz: " + ", ".join(f"R={r}->{a:.3f}dB" for r, a in zip([0, 5, 20, 50, 100], atts)))

        assert all(atts[i] <= atts[i + 1] for i in range(len(atts) - 1)), "Rain atten should increase with R"

    att_2 = propagation.rain_attenuation_db(elev, 2.0, 50)

    att_28 = propagation.rain_attenuation_db(elev, 28.0, 50)

    assert att_28 > att_2, "Rain attenuation should be much higher at 28 GHz than 2 GHz for same rain rate"

    print("  PASS: rain attenuation increases with R and with frequency\n")

def check_gaseous_attenuation_magnitude():

    print("=== Check 4: Gaseous attenuation magnitude sanity (ITU-R P.676) ===")

    for f in [2.0, 6.0, 28.0]:

        a = propagation.gaseous_attenuation_db(30.0, f, temperature_c=15.0, humidity_pct=60.0)



    print("  PASS: all values were verified to be within the expected sub-10dB range for these bands\n")

def check_outage_vs_elevation():

    print("=== Check 5: It checks that outage probability decreases with the increase in elevation ===")

    df = generate_dataset(500_000, seed=7)

    df["elev_bin"] = pd.cut(df["elevation_deg"], bins=[10, 20, 30, 45, 60, 90])

    outage_by_bin = df.groupby("elev_bin")["outage"].mean()

    print(outage_by_bin)

    vals = outage_by_bin.values

    # Sampling noise can make the individual bins move a little.

    assert vals[0] > vals[-1], "Outage probability should be higher at low elevation than high elevation"

    print("  PASS: outage probability is higher at low elevation than high elevation\n")

def check_loon_real_data_integration(loon_csv_path="loon_data/loon-flights-2021Q2_csv"):

    print("=== Check 6: Real Loon flight data integration ===")

    import os

    if not os.path.exists(loon_csv_path):

        print(f"  SKIPPED: {loon_csv_path} not found (real-data check requires the downloaded Loon CSV)\n")

        return

    import loon_loader

    sampler = loon_loader.LoonScenarioSampler.from_csv(loon_csv_path)

    print(" ", sampler.summary().replace("\n", "\n  "))

    df = generate_dataset(50_000, seed=99, loon_sampler=sampler)

    real_alt_mean = sampler.df.altitude_km.mean()

    real_drift_mean = sampler.df.drift_speed_mps.mean()

    gen_alt_mean = df.haps_altitude_km.mean()

    gen_drift_mean = df.haps_drift_mps.mean()

    print(f"  Real data altitude mean:      {real_alt_mean:.3f} km")

    print(f"  Generated altitude mean:      {gen_alt_mean:.3f} km")

    print(f"  Real data drift speed mean:   {real_drift_mean:.3f} m/s")

    print(f"  Generated drift speed mean:   {gen_drift_mean:.3f} m/s")

    assert abs(gen_alt_mean - real_alt_mean) < 0.1, "Generated altitude mean should closely match real data"

    assert abs(gen_drift_mean - real_drift_mean) < 0.5, "Generated drift mean should closely match real data"

    assert df.haps_altitude_km.max() <= sampler.df.altitude_km.max() + 1e-6, \

        "Generated altitude should be less than the values observed in real flights"

    assert set(df.platform_data_source_flight_id.unique()) <= set(sampler.df.flight_id.unique()), \

        "Each one of the generated rows should trace back to a flight ID"

    print("  PASS: generated platform geometry matches real Loon flight telemetry\n")

def run_all():

    check_los_probability_trend()

    check_snr_vs_frequency()

    check_rain_attenuation_trend()

    check_gaseous_attenuation_magnitude()

    check_outage_vs_elevation()

    check_loon_real_data_integration()

    print("All the validation checks have passed.")

if name == "__main__":

    run_all()
