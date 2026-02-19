"""
Shared conversion utilities for axion/DP calculations across notebooks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from modules.constants import GaussToSqrdeV, JouleToeV, MeterToInveV, alpha, m_e
from modules.planet_env import PlanetEnv

DEFAULT_RHO_DM = 0.3e15  # eV / m^3
DEFAULT_V0_MS = 2e5
C_SI = 299_792_458.0
DEFAULT_V0_FRAC = DEFAULT_V0_MS / C_SI

Species = Literal["axion", "dp"]


def plasma_freq(ne_cm3: ArrayLike) -> np.ndarray:
    """Natural-unit plasma frequency from an electron density [cm^-3]."""
    ne_natural = np.asarray(ne_cm3, dtype=float) * 1e6 / (MeterToInveV**3)
    return np.sqrt(4.0 * np.pi * ne_natural * alpha / (m_e))


def density_for_mass(ma_eV: ArrayLike) -> np.ndarray:
    """Electron density (cm^-3) required for resonance with mass ma."""
    ma = np.asarray(ma_eV, dtype=float)
    ne_natural = (m_e) * ma**2 / (4.0 * np.pi * alpha)
    return ne_natural / 1e6 * (MeterToInveV**3)


def _bracket_resonance(
    env: PlanetEnv,
    mass_eV: float,
    h_min_km: float,
    h_max_km: float,
    expand_factor: float = 2.0,
    max_expand: float = 1e6,
) -> Tuple[float, float]:
    """Find a bracket [h_min, h_max] with opposite signs for f(h)=ω_pl-ma."""

    def f(h_km: float) -> float:
        ne_cm3 = env.density_profile(h_km)
        return plasma_freq(ne_cm3) - mass_eV

    f_min = f(h_min_km)
    if not np.isfinite(f_min):
        return np.nan, np.nan

    h_high = h_max_km
    for _ in range(50):
        f_high = f(h_high)
        if np.isnan(f_high):
            return np.nan, np.nan
        if f_min * f_high <= 0:
            return h_min_km, h_high
        h_high *= expand_factor
        if h_high > max_expand:
            break
    return np.nan, np.nan


def resonance_altitude(
    env: PlanetEnv,
    ma_eV: ArrayLike,
    h_min_km: float | None = None,
    h_max_km: float | None = None,
) -> np.ndarray:
    """
    Altitude (km) where ω_pl(h) = m_a. Returns np.nan when no resonance exists.
    """
    if h_min_km is None:
        h_min_km = env.metadata.get("resonance_h_min_km", 0.0)
    if h_max_km is None:
        h_max_km = env.metadata.get("resonance_h_max_km", 2e4)

    masses = np.asarray(ma_eV, dtype=float)
    out = np.full_like(masses, np.nan, dtype=float)

    ne0 = env.density_profile(h_min_km)
    ma_max = plasma_freq(ne0)

    it = np.nditer(masses, flags=["multi_index"])
    while not it.finished:
        m = float(it[0])
        if np.isfinite(m) and 0 < m < ma_max:
            bracket = _bracket_resonance(env, m, h_min_km, h_max_km)
            if bracket[0] == bracket[0]:  # not nan
                try:
                    h_c = brentq(
                        lambda h: plasma_freq(env.density_profile(h)) - m,
                        bracket[0],
                        bracket[1],
                        maxiter=200,
                    )
                    out[it.multi_index] = h_c
                except ValueError:
                    out[it.multi_index] = np.nan
        it.iternext()
    return out


def resonance_radius(env: PlanetEnv, ma_eV: ArrayLike) -> np.ndarray:
    """Radius (in units of env.radius_km) where resonance occurs."""
    h_c = resonance_altitude(env, ma_eV)
    return env.radius_km + h_c


def log_plasma_gradient(
    env: PlanetEnv,
    h_km: ArrayLike,
    delta_km: float = 1.0,
) -> np.ndarray:
    """
    Numerical derivative d/dh (ln n_e) evaluated at altitude h_km.
    Returned gradient is per kilometer.
    """
    if env.log_density_grad is not None:
        return env.log_density_grad(h_km)

    h = np.asarray(h_km, dtype=float)
    delta = max(delta_km, 1e-3)
    h_plus = h + delta
    h_minus = np.clip(h - delta, 0.0, None)
    n_plus = env.density_profile(h_plus)
    n_minus = env.density_profile(h_minus)

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_plus = np.log(n_plus)
        ln_minus = np.log(n_minus)
    grad = (ln_plus - ln_minus) / (h_plus - h_minus)
    return grad


def _mixing_angle(
    species: Species,
    coupling: np.ndarray,
    B_eV2: np.ndarray,
    ma_eV: np.ndarray,
) -> np.ndarray:
    if species == "axion":
        return (coupling * 1e-9) * B_eV2 / ma_eV
    if species == "dp":
        return coupling
    raise ValueError("species must be 'axion' or 'dp'")


def conversion_probability(
    env: PlanetEnv,
    ma_eV: ArrayLike,
    coupling: ArrayLike,
    species: Species = "axion",
    velocity_frac: float = DEFAULT_V0_FRAC,
) -> np.ndarray:
    """
    WKB conversion probability for axions or dark photons.
    """
    ma = np.asarray(ma_eV, dtype=float)
    g = np.asarray(coupling, dtype=float)

    h_c = resonance_altitude(env, ma)
    valid = np.isfinite(h_c)
    B_gauss = env.magnetic_profile(h_c)
    B_eV2 = np.abs(B_gauss) * GaussToSqrdeV

    grad = log_plasma_gradient(env, h_c)
    grad_m = grad / 1e3  # per meter
    with np.errstate(divide="ignore"):
        inv_grad = np.where(grad_m != 0, 1.0 / np.abs(grad_m), np.nan)
    inv_grad *= MeterToInveV  # convert meters to eV^-1

    theta = _mixing_angle(species, g, B_eV2, ma)
    prefactor = np.pi * (ma / velocity_frac)
    if species == "dp":
        prefactor *= 2.0 / 3.0

    P = prefactor * theta**2 * inv_grad
    P = np.where(valid, P, np.nan)
    return P


def specific_brightness(
    env: PlanetEnv,
    ma_eV: ArrayLike,
    coupling: ArrayLike,
    bandwidth_Hz: float,
    species: Species = "axion",
    rho_dm: float = DEFAULT_RHO_DM,
    velocity_frac: float = DEFAULT_V0_FRAC,
) -> np.ndarray:
    """
    Specific brightness B_nu [W m^-2 Hz^-1 sr^-1] for DM-induced signal.
    """
    P = conversion_probability(env, ma_eV, coupling, species, velocity_frac)
    v_SI = velocity_frac * C_SI
    B_nu = (rho_dm / JouleToeV) * v_SI / (4.0 * np.pi) * P / bandwidth_Hz
    return 2.0 * B_nu


def spectral_flux_density(
    env: PlanetEnv,
    ma_eV: ArrayLike,
    coupling: ArrayLike,
    distance_m: float,
    bandwidth_Hz: float,
    species: Species = "axion",
    rho_dm: float = DEFAULT_RHO_DM,
    velocity_frac: float = DEFAULT_V0_FRAC,
    mask_invalid: bool = True,
) -> np.ndarray:
    """
    Spectral flux density F_nu = B_nu * Omega_proj with Omega = pi (r_c / d)^2.
    """
    B_nu = specific_brightness(
        env,
        ma_eV,
        coupling,
        bandwidth_Hz,
        species=species,
        rho_dm=rho_dm,
        velocity_frac=velocity_frac,
    )
    r_c_km = resonance_radius(env, ma_eV)
    r_c_m = r_c_km * 1e3
    Omega_proj = np.pi * (r_c_m / distance_m) ** 2

    F_nu = B_nu * Omega_proj
    if mask_invalid:
        F_nu = np.where(r_c_m < distance_m, F_nu, np.nan)
    return F_nu


def beam_filling_factor(
    env: PlanetEnv,
    ma_eV: ArrayLike,
    distance_m: float,
    beam_solid_angle: float = 8.0 * np.pi / 3.0,
    invalid: str = "nan",
) -> np.ndarray:
    """
    Beam filling factor φ_b = Ω_src / Ω_beam using the resonance radius as source size.
    """
    if beam_solid_angle <= 0:
        raise ValueError("beam_solid_angle must be > 0")

    rc_km = resonance_radius(env, ma_eV)
    rc_m = rc_km * 1e3
    omega_src = np.pi * (rc_m / distance_m) ** 2
    phi = omega_src / beam_solid_angle
    mask = (rc_m < distance_m) & np.isfinite(rc_m)

    if invalid == "nan":
        phi = np.where(mask, phi, np.nan)
    elif invalid == "zero":
        phi = np.where(mask, phi, 0.0)
    else:
        raise ValueError("invalid must be 'nan' or 'zero'")
    return phi


__all__ = [
    "plasma_freq",
    "density_for_mass",
    "resonance_altitude",
    "resonance_radius",
    "log_plasma_gradient",
    "conversion_probability",
    "specific_brightness",
    "spectral_flux_density",
    "beam_filling_factor",
]
