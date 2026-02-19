"""
Noise and background models shared across notebooks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from modules.constants import KelvinToeV, SecondToInveV
from modules.planet_env import R_SUN_KM, SUN_ENV

Z0_OHM = 376.730313668
DEFAULT_L_EFF = 5.0  # meters
PI = np.pi
AU_IN_KM = 1.496e8


def ma_to_freq(ma_eV: np.ndarray) -> np.ndarray:
    """Convert axion mass (eV) to frequency in Hz."""
    return np.asarray(ma_eV, dtype=float) / (2.0 * PI / SecondToInveV)


def freq_to_ma(freq: np.ndarray) -> np.ndarray:
    """Convert frequency in Hz to axion mass (eV)."""
    return np.asarray(freq, dtype=float) * (2.0 * PI / SecondToInveV)


def galactic_background(ma_eV: np.ndarray, B0: float = 1.38e-19) -> np.ndarray:
    """
    Isotropic Galactic radio background brightness [W/m^2/Hz] as a function of mass.
    Formula matches the notebooks: B0 * f^{-0.76} * exp(-3.28 f^{-0.64}) with f in MHz.
    """
    freq_mhz = ma_to_freq(ma_eV) * 1e-6
    return B0 * freq_mhz ** (-0.76) * np.exp(-3.28 * freq_mhz ** (-0.64))


def noise_rms(noise: np.ndarray, bandwidth_Hz: float, time_hour: float) -> np.ndarray:
    """Radiometer-equation RMS given brightness noise, bandwidth (Hz), and integration time (hours)."""
    return np.asarray(noise, dtype=float) / np.sqrt(time_hour * 3600.0 * bandwidth_Hz)


def _qtn_brightness(ma_eV, n_cm3, T_K, coeff, prefactor, L_meter):
    ma_eV = np.asarray(ma_eV, dtype=float)
    n_cm3 = np.asarray(n_cm3, dtype=float)
    nu_Hz = ma_to_freq(ma_eV)
    Vsqd_QTN = coeff * n_cm3 * T_K / (nu_Hz ** 3 * L_meter)
    return prefactor * Vsqd_QTN / (Z0_OHM * L_meter ** 2)


# --- Jupiter ---
NE_JUPITER_CM3 = 0.253
TE_JUPITER_K = 2.73 / KelvinToeV  # convert 2.73 eV to Kelvin
QTN_COEFF_JUPITER = 4e-5
QTN_PREF_JUPITER = 2.0 / (8.0 * PI / 3.0)


def brightness_qtn_jupiter(ma_eV, L_meter: float = DEFAULT_L_EFF) -> np.ndarray:
    """QTN brightness for Jovian environment at ~5 AU."""
    return _qtn_brightness(ma_eV, NE_JUPITER_CM3, TE_JUPITER_K, QTN_COEFF_JUPITER, QTN_PREF_JUPITER, L_meter)


# --- Earth (1 AU solar wind environment) ---
NE_EARTH_CM3 = 7.12
TE_EARTH_K = 4.89 / KelvinToeV
QTN_COEFF_EARTH = 5e-5
QTN_PREF_EARTH = 3.0 / (4.0 * PI)


def brightness_qtn_earth(ma_eV, L_meter: float = DEFAULT_L_EFF) -> np.ndarray:
    """QTN brightness for the terrestrial environment at 1 AU."""
    return _qtn_brightness(ma_eV, NE_EARTH_CM3, TE_EARTH_K, QTN_COEFF_EARTH, QTN_PREF_EARTH, L_meter)


# --- Sun (radial profile) ---
TE_SUN_K = 4.89 / KelvinToeV
QTN_COEFF_SUN = 5e-5
QTN_PREF_SUN = 3.0 / (4.0 * PI)


def brightness_qtn_sun(
    ma_eV,
    r_Rsun: Optional[float] = None,
    r_AU: Optional[float] = None,
    T_K: Optional[float] = None,
    L_meter: float = DEFAULT_L_EFF,
) -> np.ndarray:
    """
    QTN brightness for heliocentric distances.

    Parameters
    ----------
    ma_eV : array-like
        DM mass in eV.
    r_Rsun : float, optional
        Distance from Sun in solar radii.
    r_AU : float, optional
        Distance in AU (used if r_Rsun is None).
    T_K : float, optional
        Electron temperature in Kelvin (default TE_SUN_K).
    """
    if r_Rsun is None:
        if r_AU is None:
            r_AU = 1.0
        r_Rsun = (r_AU * AU_IN_KM) / R_SUN_KM
    altitude_km = (np.asarray(r_Rsun, dtype=float) - 1.0) * R_SUN_KM
    n_cm3 = SUN_ENV.density_profile(altitude_km)
    temperature = TE_SUN_K if T_K is None else T_K
    return _qtn_brightness(ma_eV, n_cm3, temperature, QTN_COEFF_SUN, QTN_PREF_SUN, L_meter)


__all__ = [
    "ma_to_freq",
    "galactic_background",
    "noise_rms",
    "brightness_qtn_jupiter",
    "brightness_qtn_earth",
    "brightness_qtn_sun",
]
