"""
link_budget.py
--------------
Link-budget calculations used by the simulator.

The functions below combine the different propagation terms to obtain
received power, noise power, SNR, SINR, capacity, BER, and outage.
"""

import numpy as np

K_BOLTZMANN = 1.380649e-23  # J/K


def thermal_noise_power_dbm(bandwidth_hz, noise_figure_db=3.0, temperature_k=290.0):
    """
    Thermal noise power in dBm.
    """
    bandwidth_hz = np.asarray(bandwidth_hz, dtype=float)
    noise_power_w = K_BOLTZMANN * temperature_k * bandwidth_hz
    noise_power_dbm = 10 * np.log10(noise_power_w) + 30.0
    return noise_power_dbm + noise_figure_db


def received_power_dbm(tx_power_dbm, tx_antenna_gain_dbi, rx_antenna_gain_dbi,
                        fspl_db, gas_atten_db, cloud_atten_db, rain_atten_db,
                        shadow_fading_db, small_scale_fading_db, misc_loss_db=0.0):
    """
    Calculate the received power from the link-budget terms.
    """
    return (tx_power_dbm + tx_antenna_gain_dbi + rx_antenna_gain_dbi
            - fspl_db - gas_atten_db - cloud_atten_db - rain_atten_db
            - shadow_fading_db + small_scale_fading_db - misc_loss_db)


def snr_db(rx_power_dbm, noise_power_dbm):
    return rx_power_dbm - noise_power_dbm


def sinr_db(rx_power_dbm, noise_power_dbm, interference_power_dbm=None):
    """
    Calculate SINR. If no interference is supplied, this is just SNR.
    """
    noise_lin = 10 ** (noise_power_dbm / 10.0)

    if interference_power_dbm is None:
        denom_lin = noise_lin
    else:
        interf_lin = 10 ** (np.asarray(interference_power_dbm) / 10.0)
        denom_lin = noise_lin + interf_lin

    denom_dbm = 10 * np.log10(denom_lin)
    return rx_power_dbm - denom_dbm


def shannon_capacity_bps_hz(sinr_db_, implementation_efficiency=0.75):
    """
    Spectral efficiency based on the Shannon expression.
    """
    sinr_lin = 10 ** (np.asarray(sinr_db_) / 10.0)
    return implementation_efficiency * np.log2(1.0 + sinr_lin)


def approximate_ber_qpsk(snr_db_):
    """
    Uncoded QPSK BER approximation for an AWGN channel.
    """
    from scipy.special import erfc

    snr_lin = 10 ** (np.asarray(snr_db_) / 10.0)
    ber = 0.5 * erfc(np.sqrt(np.maximum(snr_lin, 0.0)))
    return np.clip(ber, 0.0, 0.5)


def outage_indicator(sinr_db_, sinr_threshold_db=0.0):
    """
    Mark samples for which SINR is below the selected threshold.
    """
    return (np.asarray(sinr_db_) < sinr_threshold_db).astype(int)


def link_margin_db(rx_power_dbm, sensitivity_dbm):
    """
    Received power relative to the receiver sensitivity.
    """
    return rx_power_dbm - sensitivity_dbm