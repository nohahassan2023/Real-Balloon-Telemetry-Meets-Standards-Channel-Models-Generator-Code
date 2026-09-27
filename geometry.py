"""
geometry.py

Geometry and mobility functions for HAPS links.
"""

import numpy as np


EARTH_RADIUS_KM = 6371.0


def slant_range_km(
    haps_altitude_km,
    ground_distance_km,
    earth_radius_km=EARTH_RADIUS_KM
):
    """
    Slant range and elevation angle for a spherical Earth.
    """
    haps_altitude_km = np.asarray(haps_altitude_km, dtype=float)
    ground_distance_km = np.asarray(ground_distance_km, dtype=float)

    Re = earth_radius_km
    h = haps_altitude_km
    d_ground = ground_distance_km

    # Ground distance -> angle at the centre of the Earth.
    central_angle = d_ground / Re

    r_haps = Re + h

    slant = np.sqrt(
        Re**2
        + r_haps**2
        - 2 * Re * r_haps * np.cos(central_angle)
    )

    # Use atan2 here rather than the equivalent arcsin form.
    elevation_rad = np.arctan2(
        r_haps * np.cos(central_angle) - Re,
        r_haps * np.sin(central_angle)
    )

    elevation_deg = np.degrees(elevation_rad)
    elevation_deg = np.clip(elevation_deg, 0.0, 90.0)

    return slant, elevation_deg


def elevation_from_slant_range(
    haps_altitude_km,
    slant_range_km_,
    earth_radius_km=EARTH_RADIUS_KM
):
    """
    Recover elevation from altitude and slant range.
    """
    Re = earth_radius_km
    h = np.asarray(haps_altitude_km, dtype=float)
    s = np.asarray(slant_range_km_, dtype=float)

    r_haps = Re + h

    cos_central = (
        Re**2 + r_haps**2 - s**2
    ) / (2 * Re * r_haps)

    cos_central = np.clip(cos_central, -1.0, 1.0)
    central_angle = np.arccos(cos_central)

    elevation_rad = np.arctan2(
        r_haps * np.cos(central_angle) - Re,
        r_haps * np.sin(central_angle) + 1e-12
    )

    return np.degrees(
        np.clip(elevation_rad, 0.0, np.pi / 2)
    )


def max_ground_distance_km(
    haps_altitude_km,
    min_elevation_deg=5.0,
    earth_radius_km=EARTH_RADIUS_KM
):
    """
    Ground distance corresponding to the minimum elevation angle.
    """
    Re = earth_radius_km
    h = np.asarray(haps_altitude_km, dtype=float)
    r_haps = Re + h
    min_elev_rad = np.radians(min_elevation_deg)

    theta_lo = np.zeros_like(h)
    theta_hi = np.full_like(h, np.pi / 2)

    # Binary search for the central angle.
    for _ in range(60):
        theta_mid = 0.5 * (theta_lo + theta_hi)

        elev = np.arctan2(
            r_haps * np.cos(theta_mid) - Re,
            r_haps * np.sin(theta_mid) + 1e-12
        )

        mask = elev > min_elev_rad

        theta_lo = np.where(mask, theta_mid, theta_lo)
        theta_hi = np.where(mask, theta_hi, theta_mid)

    return Re * theta_lo


def sample_user_positions(
    n_samples,
    haps_altitude_km,
    min_elevation_deg=5.0,
    max_coverage_km=None,
    rng=None
):
    """
    Generate random user positions within the coverage area.
    """
    if rng is None:
        rng = np.random.default_rng()

    r_max = max_ground_distance_km(
        haps_altitude_km,
        min_elevation_deg
    )

    if max_coverage_km is not None:
        r_max = np.minimum(r_max, max_coverage_km)

    # sqrt(u) gives uniform density over area.
    u = rng.uniform(0.0, 1.0, size=n_samples)
    ground_distance_km = r_max * np.sqrt(u)

    azimuth_deg = rng.uniform(
        0.0, 360.0, size=n_samples
    )

    return ground_distance_km, azimuth_deg


def doppler_shift_hz(
    carrier_freq_hz,
    haps_velocity_mps,
    user_velocity_mps,
    elevation_deg,
    relative_heading_deg=0.0,
    speed_of_light=2.998e8
):
    """
    Approximate Doppler shift from HAPS and user motion.
    """
    elev_rad = np.radians(elevation_deg)
    heading_rad = np.radians(relative_heading_deg)

    # Radial component of the relative velocity.
    v_rel = (
        haps_velocity_mps * np.cos(elev_rad)
        - user_velocity_mps
        * np.cos(elev_rad)
        * np.cos(heading_rad)
    )

    return (v_rel / speed_of_light) * carrier_freq_hz
