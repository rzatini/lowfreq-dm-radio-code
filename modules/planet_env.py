from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]
R_SUN_KM = 6.957e5


def _evaluate(func: Callable[[np.ndarray], np.ndarray], values: ArrayLike) -> ArrayLike:
    """Helper: vectorize call and return scalar if scalar input."""
    arr = np.asarray(values, dtype=float)
    result = func(arr)
    if np.isscalar(values):
        return float(np.asarray(result))
    return result


@dataclass
class Profile:
    """Generic callable profile (density, magnetic field, etc.)."""

    name: str
    evaluator: Callable[[np.ndarray], np.ndarray]
    description: str = ""

    def __call__(self, values: ArrayLike) -> ArrayLike:
        return _evaluate(self.evaluator, values)


@dataclass
class PlanetEnv:
    """
    Minimal container for planetary environments used in the notebooks.

    Attributes
    ----------
    name : str
        Short identifier (e.g. \"jupiter\", \"earth\").
    radius_km : float
        Mean radius in km.
    density_profile : Profile
        Electron density profile n_e(h) [cm^-3] with h in km unless noted.
    magnetic_profile : Profile
        Transverse magnetic-field profile B_T(h) [Gauss].
    metadata : Dict[str, float | str]
        Free-form dictionary for extra parameters (scale height, references, etc.).
    """

    name: str
    radius_km: float
    density_profile: Profile
    magnetic_profile: Profile
    log_density_grad: Optional[Callable[[np.ndarray], np.ndarray]] = None
    metadata: Dict[str, Union[str, float]] = field(default_factory=dict)

    @property
    def radius_m(self) -> float:
        return self.radius_km * 1e3


# -------------------------------------------------------------------------
# Profile factories
# -------------------------------------------------------------------------


def exponential_density_profile(
    n0_cm3: float, scale_height_km: float
) -> Profile:
    """n_e(h) = n0 * exp(-h / H)."""

    def _profile(h_km: np.ndarray) -> np.ndarray:
        return n0_cm3 * np.exp(-h_km / scale_height_km)

    return Profile(
        name="exponential",
        evaluator=_profile,
        description=f"n0={n0_cm3:.2e} cm^-3, H={scale_height_km:.0f} km",
    )


def chapman_density_profile(
    n_max: float, z_max_km: float, H_km: float
) -> Profile:
    """Chapman-like profile used for the terrestrial ionosphere."""

    def _profile(h_km: np.ndarray) -> np.ndarray:
        x = (h_km - z_max_km) / H_km
        return n_max * np.exp(0.5 * (1 - x - np.exp(-x)))

    return Profile(
        name="chapman",
        evaluator=_profile,
        description=f"n_max={n_max:.2e} cm^-3, z_max={z_max_km:.0f} km, H={H_km:.0f} km",
    )


def dipole_field_profile(B0_G: float, radius_km: float) -> Profile:
    """Pure dipole transverse magnetic field."""

    def _profile(h_km: np.ndarray) -> np.ndarray:
        return B0_G * (radius_km / (radius_km + h_km)) ** 3

    return Profile(
        name="dipole",
        evaluator=_profile,
        description=f"B0={B0_G:.2f} G @ surface, radius={radius_km:.0f} km",
    )


def constant_field_profile(B0_G: float) -> Profile:
    """Constant magnetic field (used for simplified solar hotspots)."""

    def _profile(h_km: np.ndarray) -> np.ndarray:
        return np.full_like(h_km, B0_G, dtype=float)

    return Profile(
        name="constant",
        evaluator=_profile,
        description=f"B={B0_G:.2f} G (constant)",
    )


def _wexler_terms(h_km: np.ndarray, radius_km: float, delta_min: float = 1e-6):
    r = 1.0 + h_km / radius_km
    r = np.maximum(r, 1.0)
    delta = np.maximum(r - 1.0, delta_min)
    return r, delta


def wexler2019_density_profile(
    radius_km: float = R_SUN_KM, delta_min: float = 1e-6
) -> Profile:
    """Quiet-Sun density profile (Wexler et al. 2019) as a function of altitude in km."""

    def _profile(h_km: np.ndarray) -> np.ndarray:
        h = np.asarray(h_km, dtype=float)
        r_Rsun, delta = _wexler_terms(h, radius_km, delta_min)
        return 1e6 * (65 * r_Rsun ** (-5.94) + 0.768 * delta ** (-2.25))

    return Profile(
        name="wexler2019",
        evaluator=_profile,
        description="Wexler+2019 eq.21, h measured from solar surface",
    )


def wexler2019_log_grad(
    radius_km: float = R_SUN_KM, delta_min: float = 1e-6
) -> Callable[[np.ndarray], np.ndarray]:
    """Analytic d(ln n_e)/dh in km^-1 for the Wexler+2019 density."""

    density_profile = wexler2019_density_profile(radius_km, delta_min)

    def _grad(h_km: np.ndarray) -> np.ndarray:
        h = np.asarray(h_km, dtype=float)
        r_Rsun, delta = _wexler_terms(h, radius_km, delta_min)
        dens = density_profile(h)
        dr_dh = 1.0 / radius_km

        term1 = 65 * (-5.94) * r_Rsun ** (-6.94)
        delta_mask = (r_Rsun - 1.0) > delta_min
        term2 = np.where(
            delta_mask, 0.768 * (-2.25) * delta ** (-3.25), 0.0
        )
        dn_dr = 1e6 * (term1 + term2)
        dn_dh = dn_dr * dr_dh
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = np.divide(dn_dh, dens, out=np.full_like(dn_dh, np.nan), where=dens != 0)
        return grad

    return _grad


def dqcs_field_profile(
    radius_km: float = R_SUN_KM,
    M: float = 1.789,
    Q: float = 1.5,
    a1: float = 1.538,
    K: float = 1.0,
) -> Profile:
    """
    DQCS model for the solar magnetic field projected on the ecliptic plane.

    Parameters follow the nomenclature of Banaszkiewicz et al. (1998).
    Returns B_z in Gauss as a function of altitude h above the photosphere.
    """

    def _profile(h_km: np.ndarray) -> np.ndarray:
        r_Rsun = 1.0 + h_km / radius_km
        r_Rsun = np.maximum(r_Rsun, 1.0)
        term1 = -1.0 / r_Rsun**3
        term2 = (9.0 * Q) / (8.0 * r_Rsun**5)
        term3 = K / (a1**2 + r_Rsun**2) ** 1.5
        return M * (term1 + term2 + term3)

    return Profile(
        name="dqcs",
        evaluator=_profile,
        description="DQCS B_z along ecliptic (Gauss)",
    )


# -------------------------------------------------------------------------
# Pre-defined environments
# -------------------------------------------------------------------------

JUPITER_ENV = PlanetEnv(
    name="jupiter",
    radius_km=71_492.0,
    density_profile=exponential_density_profile(n0_cm3=1e6, scale_height_km=700.0),
    magnetic_profile=dipole_field_profile(B0_G=4.2, radius_km=71_492.0),
    log_density_grad=lambda h: np.full_like(np.asarray(h, dtype=float), -1.0 / 700.0),
    metadata={
        "reference": "Kurth et al. 2025",
        "scale_height_km": 700.0,
        "n0_cm3": 1e6,
        "resonance_h_min_km": 0.0,
        "resonance_h_max_km": 5_000.0,
    },
)

EARTH_ENV = PlanetEnv(
    name="earth",
    radius_km=6_371.0,
    density_profile=chapman_density_profile(n_max=1e6, z_max_km=300.0, H_km=100.0),
    magnetic_profile=dipole_field_profile(B0_G=0.3, radius_km=6_371.0),
    log_density_grad=lambda h: (
        (-1.0 + np.exp(-(np.asarray(h, dtype=float) - 300.0) / 100.0))
        / (2.0 * 100.0)
    ),
    metadata={
        "reference": "Chapman layer (F-region)",
        "n_max_cm3": 1e6,
        "z_max_km": 300.0,
        "H_km": 100.0,
        "resonance_h_min_km": 300.0,
        "resonance_h_max_km": 2_000.0,
    },
)

SUN_ENV = PlanetEnv(
    name="sun",
    radius_km=R_SUN_KM,
    density_profile=wexler2019_density_profile(),
    magnetic_profile=dqcs_field_profile(),
    log_density_grad=wexler2019_log_grad(),
    metadata={
        "reference": "Wexler+2019 density & DQCS magnetic field",
        "M": 1.789,
        "Q": 1.5,
        "a1": 1.538,
        "K": 1.0,
        "resonance_h_min_km": 10.0,
        "resonance_h_max_km": 4000.0 * R_SUN_KM,
    },
)

__all__ = [
    "ArrayLike",
    "Profile",
    "PlanetEnv",
    "exponential_density_profile",
    "chapman_density_profile",
    "dipole_field_profile",
    "constant_field_profile",
    "wexler2019_density_profile",
    "wexler2019_log_grad",
    "dqcs_field_profile",
    "JUPITER_ENV",
    "EARTH_ENV",
    "SUN_ENV",
]
