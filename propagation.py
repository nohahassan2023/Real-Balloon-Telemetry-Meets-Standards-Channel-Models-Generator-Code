"""
propagation.py
--------------
Propagation models for HAPS links.

The module includes free-space loss, atmospheric attenuation, LoS
probability, shadow fading, and small-scale fading. ITU-R propagation
models are evaluated through the itur package, while NTN channel
parameters are taken from 3GPP TR 38.811.
"""

import numpy as np

import itur.models.itu676 as _itu676
import itur.models.itu838 as _itu838
import itur.models.itu840 as _itu840


C_LIGHT = 2.998e8  # m/s


# ---------------------------------------------------------------------------
# Free-space path loss
# ---------------------------------------------------------------------------

def free_space_path_loss_db(distance_km, freq_ghz):
    """
    Calculate free-space path loss using the Friis equation.

    Distance is given in km and frequency in GHz.
    """
    distance_km = np.asarray(distance_km, dtype=float)
    freq_ghz = np.asarray(freq_ghz, dtype=float)

    return (
        20.0 * np.log10(distance_km)
        + 20.0 * np.log10(freq_ghz)
        + 92.45
    )


# ---------------------------------------------------------------------------
# ITU-R P.676: gaseous attenuation
# ---------------------------------------------------------------------------

def _water_vapor_density_g_m3(temperature_c, humidity_pct):
    """
    Calculate water-vapour density from temperature and relative humidity.
    """
    T_k = np.asarray(temperature_c, dtype=float) + 273.15
    RH = np.asarray(humidity_pct, dtype=float)
    Tc = np.asarray(temperature_c, dtype=float)

    es_hpa = 6.1121 * np.exp(
        (17.502 * Tc) / (Tc + 240.97)
    )

    e_hpa = RH / 100.0 * es_hpa

    rho = (
        (e_hpa * 100.0)
        / (461.5 * T_k)
        * 1000.0
    )

    return rho


def _build_gas_attenuation_lut(
    freq_ghz_values,
    elev_grid_deg=None,
    rho_grid=None,
    T_k_grid=None,
    pressure_hpa=1013.25,
):
    """
    Build lookup tables for gaseous attenuation.
    """
    from scipy.interpolate import RegularGridInterpolator

    if elev_grid_deg is None:
        elev_grid_deg = np.linspace(5.0, 90.0, 35)

    if rho_grid is None:
        rho_grid = np.linspace(0.5, 25.0, 25)

    if T_k_grid is None:
        T_k_grid = np.linspace(253.15, 320.0, 12)

    lut = {}

    for f in freq_ghz_values:
        table = np.empty(
            (
                len(elev_grid_deg),
                len(rho_grid),
                len(T_k_grid),
            )
        )

        for ti, T_k in enumerate(T_k_grid):
            for ri, rho in enumerate(rho_grid):
                vals = _itu676.gaseous_attenuation_slant_path(
                    f,
                    elev_grid_deg,
                    rho=rho,
                    P=pressure_hpa,
                    T=T_k,
                    mode="approx",
                )

                table[:, ri, ti] = np.asarray(vals).ravel()

        lut[float(f)] = RegularGridInterpolator(
            (elev_grid_deg, rho_grid, T_k_grid),
            table,
            bounds_error=False,
            fill_value=None,
        )

    return lut


_GAS_LUT_CACHE = {}


def gaseous_attenuation_db(
    elevation_deg,
    freq_ghz,
    temperature_c=15.0,
    humidity_pct=60.0,
    pressure_hpa=1013.25,
):
    """
    Calculate slant-path gaseous attenuation using ITU-R P.676.

    A lookup table is built for each carrier frequency and reused for
    subsequent calls.
    """
    elevation_deg = np.atleast_1d(
        np.asarray(elevation_deg, dtype=float)
    )
    freq_ghz = np.atleast_1d(
        np.asarray(freq_ghz, dtype=float)
    )
    temperature_c = np.atleast_1d(
        np.asarray(temperature_c, dtype=float)
    )
    humidity_pct = np.atleast_1d(
        np.asarray(humidity_pct, dtype=float)
    )

    elev_clipped = np.clip(elevation_deg, 5.0, 90.0)
    T_k = temperature_c + 273.15
    rho = _water_vapor_density_g_m3(
        temperature_c,
        humidity_pct,
    )

    elev_b, f_b, T_b, rho_b = np.broadcast_arrays(
        elev_clipped,
        freq_ghz,
        T_k,
        rho,
    )

    cache_key = round(pressure_hpa, 2)

    if cache_key not in _GAS_LUT_CACHE:
        _GAS_LUT_CACHE[cache_key] = {}

    lut_for_pressure = _GAS_LUT_CACHE[cache_key]

    out = np.empty(elev_b.shape, dtype=float)

    unique_freqs = np.unique(f_b)

    missing = [
        f for f in unique_freqs
        if float(f) not in lut_for_pressure
    ]

    if missing:
        lut_for_pressure.update(
            _build_gas_attenuation_lut(
                missing,
                pressure_hpa=pressure_hpa,
            )
        )

    for f in unique_freqs:
        mask = f_b == f
        interpolator = lut_for_pressure[float(f)]

        pts = np.stack(
            [
                elev_b[mask],
                rho_b[mask],
                T_b[mask],
            ],
            axis=-1,
        )

        out[mask] = interpolator(pts)

    return out if out.size > 1 else float(out.ravel()[0])


# ---------------------------------------------------------------------------
# ITU-R P.840: cloud and fog attenuation
# ---------------------------------------------------------------------------

_CLOUD_KL_CACHE = {}


def cloud_attenuation_db(
    elevation_deg,
    freq_ghz,
    liquid_water_path_kg_m2=0.0,
    temperature_c=0.0,
):
    """
    Calculate cloud and fog attenuation using ITU-R P.840.
    """
    elevation_deg = np.atleast_1d(
        np.asarray(elevation_deg, dtype=float)
    )
    freq_ghz = np.atleast_1d(
        np.asarray(freq_ghz, dtype=float)
    )
    L = np.atleast_1d(
        np.asarray(liquid_water_path_kg_m2, dtype=float)
    )
    T = np.atleast_1d(
        np.asarray(temperature_c, dtype=float)
    )

    elev_b, f_b, L_b, T_b = np.broadcast_arrays(
        elevation_deg,
        freq_ghz,
        L,
        T,
    )

    elev_clip = np.clip(elev_b, 5.0, 90.0)

    T_grid = np.linspace(-25.0, 50.0, 76)

    Kl_b = np.empty(elev_b.shape, dtype=float)

    unique_freqs = np.unique(f_b)

    for f in unique_freqs:
        key = float(f)

        if key not in _CLOUD_KL_CACHE:
            kl_vals = np.array([
                float(
                    np.asarray(
                        _itu840.specific_attenuation_coefficients(
                            f=key,
                            T=t,
                        )
                    ).squeeze()
                )
                for t in T_grid
            ])

            _CLOUD_KL_CACHE[key] = kl_vals

        kl_vals = _CLOUD_KL_CACHE[key]
        mask = f_b == f

        Kl_b[mask] = np.interp(
            T_b[mask],
            T_grid,
            kl_vals,
        )

    zenith_atten = Kl_b * L_b

    result = np.maximum(
        zenith_atten
        / np.sin(np.radians(elev_clip)),
        0.0,
    )

    return result if result.size > 1 else float(result.ravel()[0])


# ---------------------------------------------------------------------------
# ITU-R P.618 / P.838: rain attenuation
# ---------------------------------------------------------------------------

def _build_rain_specific_atten_lut(
    freq_ghz_values,
    elev_grid_deg=None,
    R_grid=None,
    polarization_tau_deg=45.0,
):
    """
    Build lookup tables for ITU-R P.838 rain specific attenuation.
    """
    from scipy.interpolate import RegularGridInterpolator

    if elev_grid_deg is None:
        elev_grid_deg = np.linspace(5.0, 90.0, 35)

    if R_grid is None:
        R_grid = np.concatenate([
            [0.0],
            np.geomspace(0.1, 150.0, 40),
        ])

    lut = {}

    for f in freq_ghz_values:
        table = np.empty(
            (
                len(elev_grid_deg),
                len(R_grid),
            )
        )

        for ri, R in enumerate(R_grid):
            if R <= 0:
                table[:, ri] = 0.0
                continue

            vals = _itu838.rain_specific_attenuation(
                R=R,
                f=f,
                el=elev_grid_deg,
                tau=polarization_tau_deg,
            )

            table[:, ri] = np.asarray(vals).ravel()

        lut[float(f)] = RegularGridInterpolator(
            (elev_grid_deg, R_grid),
            table,
            bounds_error=False,
            fill_value=None,
        )

    return lut


_RAIN_LUT_CACHE = {}


def rain_attenuation_db(
    elevation_deg,
    freq_ghz,
    rain_rate_mm_per_hr,
    haps_altitude_km=20.0,
    polarization_tau_deg=45.0,
    rain_height_km=4.0,
):
    """
    Calculate rain attenuation along the HAPS slant path.

    Rain specific attenuation is obtained from ITU-R P.838. The effective
    path length is calculated using the P.618 approach.
    """
    elevation_deg = np.atleast_1d(
        np.asarray(elevation_deg, dtype=float)
    )
    freq_ghz = np.atleast_1d(
        np.asarray(freq_ghz, dtype=float)
    )
    R = np.atleast_1d(
        np.asarray(rain_rate_mm_per_hr, dtype=float)
    )

    elev_b, f_b, R_b = np.broadcast_arrays(
        elevation_deg,
        freq_ghz,
        R,
    )

    elev_clip = np.clip(elev_b, 5.0, 90.0)
    R_safe = np.maximum(R_b, 0.0)

    cache_key = round(polarization_tau_deg, 2)

    if cache_key not in _RAIN_LUT_CACHE:
        _RAIN_LUT_CACHE[cache_key] = {}

    lut_for_tau = _RAIN_LUT_CACHE[cache_key]

    gamma_R = np.zeros(elev_b.shape, dtype=float)

    unique_freqs = np.unique(f_b)

    missing = [
        f for f in unique_freqs
        if float(f) not in lut_for_tau
    ]

    if missing:
        lut_for_tau.update(
            _build_rain_specific_atten_lut(
                missing,
                polarization_tau_deg=polarization_tau_deg,
            )
        )

    for f in unique_freqs:
        mask = f_b == f
        interpolator = lut_for_tau[float(f)]

        pts = np.stack(
            [
                elev_clip[mask],
                R_safe[mask],
            ],
            axis=-1,
        )

        gamma_R[mask] = np.maximum(
            interpolator(pts),
            0.0,
        )

    elev_rad = np.radians(elev_clip)

    Ls = rain_height_km / np.sin(elev_rad)

    with np.errstate(invalid="ignore"):
        r001 = 1.0 / (
            1.0
            + 0.78
            * np.sqrt(
                np.maximum(
                    Ls * gamma_R
                    / np.maximum(f_b, 1e-6),
                    0.0,
                )
            )
            - 0.38 * (1.0 - np.exp(-2.0 * Ls))
        )

    r001 = np.clip(r001, 0.1, 1.0)

    Lr = Ls * r001
    attenuation = gamma_R * Lr

    attenuation = np.where(
        R_safe <= 0,
        0.0,
        attenuation,
    )

    return np.maximum(attenuation, 0.0)


# ---------------------------------------------------------------------------
# 3GPP TR 38.811: LoS probability
# ---------------------------------------------------------------------------

_LOS_PROB_TABLE = {
    "urban": np.array([
        24.6, 38.6, 49.3, 61.3, 72.6,
        80.5, 91.9, 96.8, 99.2,
    ]) / 100.0,
    "suburban": np.array([
        78.2, 86.9, 91.3, 92.9, 95.0,
        96.0, 97.5, 98.1, 98.5,
    ]) / 100.0,
    "rural": np.array([
        95.0, 97.0, 97.6, 98.4, 98.7,
        99.0, 99.6, 99.9, 100.0,
    ]) / 100.0,
}

_LOS_PROB_ELEV_GRID = np.array(
    [10, 20, 30, 40, 50, 60, 70, 80, 90],
    dtype=float,
)


def los_probability(elevation_deg, environment="suburban"):
    """
    Return the LoS probability from 3GPP TR 38.811.

    Values between the tabulated elevation angles are linearly
    interpolated.
    """
    environment = environment.lower()

    if environment not in _LOS_PROB_TABLE:
        raise ValueError(
            f"environment must be one of {list(_LOS_PROB_TABLE)}"
        )

    elevation_deg = np.clip(
        np.asarray(elevation_deg, dtype=float),
        10.0,
        90.0,
    )

    return np.interp(
        elevation_deg,
        _LOS_PROB_ELEV_GRID,
        _LOS_PROB_TABLE[environment],
    )


def sample_los_state(elevation_deg, environment, rng=None):
    """Draw LoS/NLoS states using the elevation-dependent LoS probability."""
    if rng is None:
        rng = np.random.default_rng()

    p_los = los_probability(
        elevation_deg,
        environment,
    )

    u = rng.uniform(
        0.0,
        1.0,
        size=np.shape(p_los),
    )

    return u < p_los


# ---------------------------------------------------------------------------
# Shadow fading
# ---------------------------------------------------------------------------

_SHADOW_SIGMA_LOS = {
    "urban": np.array([
        4.0, 3.5, 3.0, 2.7, 2.5,
        2.3, 2.2, 2.1, 2.0,
    ]),
    "suburban": np.array([
        2.0, 1.8, 1.6, 1.5, 1.4,
        1.3, 1.2, 1.1, 1.0,
    ]),
    "rural": np.array([
        1.5, 1.4, 1.3, 1.2, 1.1,
        1.0, 1.0, 0.9, 0.9,
    ]),
}

_SHADOW_SIGMA_NLOS = {
    "urban": np.array([
        10.0, 9.5, 9.0, 8.5, 8.2,
        8.0, 7.8, 7.6, 7.5,
    ]),
    "suburban": np.array([
        6.0, 5.7, 5.5, 5.3, 5.1,
        5.0, 4.9, 4.8, 4.7,
    ]),
    "rural": np.array([
        5.0, 4.8, 4.6, 4.4, 4.3,
        4.2, 4.1, 4.0, 4.0,
    ]),
}


def shadow_fading_db(
    elevation_deg,
    environment,
    is_los,
    rng=None,
):
    """
    Draw shadow-fading samples using the TR 38.811 standard deviations.

    Returns
    -------
    shadow_db : ndarray
        Shadow-fading sample in dB.
    sigma : ndarray
        Standard deviation used for each sample.
    """
    if rng is None:
        rng = np.random.default_rng()

    environment = environment.lower()

    elevation_deg = np.clip(
        np.asarray(elevation_deg, dtype=float),
        10.0,
        90.0,
    )

    is_los = np.asarray(
        is_los,
        dtype=bool,
    )

    sigma_los = np.interp(
        elevation_deg,
        _LOS_PROB_ELEV_GRID,
        _SHADOW_SIGMA_LOS[environment],
    )

    sigma_nlos = np.interp(
        elevation_deg,
        _LOS_PROB_ELEV_GRID,
        _SHADOW_SIGMA_NLOS[environment],
    )

    sigma = np.where(
        is_los,
        sigma_los,
        sigma_nlos,
    )

    return rng.normal(0.0, sigma), sigma


# ---------------------------------------------------------------------------
# Small-scale fading
# ---------------------------------------------------------------------------

_RICIAN_K_DB = {
    "urban": np.array([
        8.0, 9.0, 10.0, 10.5, 11.0,
        11.5, 12.0, 12.5, 13.0,
    ]),
    "suburban": np.array([
        10.0, 11.0, 12.0, 12.5, 13.0,
        13.5, 14.0, 14.5, 15.0,
    ]),
    "rural": np.array([
        12.0, 13.0, 14.0, 14.5, 15.0,
        15.5, 16.0, 16.5, 17.0,
    ]),
}


def rician_k_factor_db(elevation_deg, environment):
    """Return the Rician K-factor in dB."""
    environment = environment.lower()

    elevation_deg = np.clip(
        np.asarray(elevation_deg, dtype=float),
        10.0,
        90.0,
    )

    return np.interp(
        elevation_deg,
        _LOS_PROB_ELEV_GRID,
        _RICIAN_K_DB[environment],
    )


def sample_small_scale_fading_db(
    elevation_deg,
    environment,
    is_los,
    rng=None,
    nakagami_m_nlos=2.0,
):
    """
    Draw small-scale fading samples.

    LoS samples use a Rician distribution. NLoS samples use a
    Nakagami-m distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_los = np.asarray(
        is_los,
        dtype=bool,
    )

    n = np.shape(is_los)

    K_db = rician_k_factor_db(
        elevation_deg,
        environment,
    )

    K_lin = 10.0 ** (K_db / 10.0)

    s = np.sqrt(
        K_lin / (K_lin + 1.0)
    )

    sigma_scatter = np.sqrt(
        1.0 / (2.0 * (K_lin + 1.0))
    )

    x = (
        s
        + rng.normal(0.0, 1.0, size=n)
        * sigma_scatter
    )

    y = (
        rng.normal(0.0, 1.0, size=n)
        * sigma_scatter
    )

    rician_power = x ** 2 + y ** 2

    rician_db = 10.0 * np.log10(
        np.maximum(rician_power, 1e-12)
    )

    nakagami_power = rng.gamma(
        shape=nakagami_m_nlos,
        scale=1.0 / nakagami_m_nlos,
        size=n,
    )

    nakagami_db = 10.0 * np.log10(
        np.maximum(nakagami_power, 1e-12)
    )

    return np.where(
        is_los,
        rician_db,
        nakagami_db,
    )
