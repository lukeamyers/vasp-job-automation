"""
This EOS fitting code is based on the following paper:

Shun-Li Shang et al., Computational Materials Science, 47, 4, (2010).
https://doi.org/10.1016/j.commatsci.2009.12.006

It includes the following equations of state: 4-parameter Teter-Shang modified Birch-Murnaghan (mBM4),
5-parameter Teter-Shang modified Birch-Murnaghan (mBM5),
4-parameter Birch-Murnaghan (BM4),
5-parameter Birch-Murnaghan (BM5),
4-parameter Natural (LOG4),
5-parameter Natural (LOG5),
4-parameter Murnaghan,
4-parameter Vinet,
and 4-parameter Morse.
"""

# Related third party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from distinctipy import get_colors
from scipy.optimize import fsolve, curve_fit

# Conversion factor
EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa


# mBM4 EOS Functions
def mBM4_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float
) -> float | np.ndarray:
    """mBM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        a + b * (volume) ** (-1 / 3) + c * (volume) ** (-2 / 3) + d * (volume) ** (-1)
    )
    return energy


def mBM4_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Derivative of mBM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b * (-1 / 3) * (volume) ** (-4 / 3)
        + c * (-2 / 3) * (volume) ** (-5 / 3)
        + d * (-1) * (volume) ** (-2)
    )
    return energy_derivative


def mBM4_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Second derivative of mBM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        b * (4 / 9) * (volume) ** (-7 / 3)
        + c * (10 / 9) * (volume) ** (-8 / 3)
        + d * (2) * (volume) ** (-3)
    )
    return energy_derivative2


def mBM4_eos_parameters(
    a: float, b: float, c: float, d: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, and d.

    Args:
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = (
        -(
            4 * c**3
            - 9 * b * c * d
            + np.sqrt((c**2 - 3 * b * d) * (4 * c**2 - 3 * b * d) ** 2)
        )
        / b**3
    )
    E0 = mBM4_equation(V0, a, b, c, d)
    B = (
        (2 * d) / V0**3 + (10 * c) / (9 * V0 ** (8 / 3)) + (4 * b) / (9 * V0 ** (7 / 3))
    ) * V0
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (54 * d * V0 ** (1 / 3) + 25 * c * V0 ** (2 / 3) + 8 * b * V0) / (
        27 * d * V0 ** (1 / 3) + 15 * c * V0 ** (2 / 3) + 6 * b * V0
    )
    B2P = (
        V0 ** (8 / 3)
        * (
            9 * d * (5 * c * V0 ** (2 / 3) + 8 * b * V0)
            + 2 * V0 ** (1 / 3) * (5 * c * (b * V0))
        )
    ) / (2 * (9 * d * V0 ** (1 / 3) + 5 * c * V0 ** (2 / 3) + 2 * b * V0) ** 3)
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA

    return V0, E0, B, BP, B2P


def mBM4(
    volume: float | np.ndarray, energy: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the mBM4 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d = curve_fit(mBM4_equation, volume, energy, p0=[100, 100, 100, 100])[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = mBM4_equation(volume_range, a, b, c, d)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * mBM4_derivative(volume_range, b, c, d)
    )
    V0, E0, B, BP, B2P = mBM4_eos_parameters(a, b, c, d)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, 0])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# mBM5 EOS Functions
def mBM5_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """mBM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        a
        + b * (volume) ** (-1 / 3)
        + c * (volume) ** (-2 / 3)
        + d * (volume) ** (-1)
        + e * (volume) ** (-4 / 3)
    )
    return energy


def mBM5_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Derivative of mBM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b * (-1 / 3) * (volume) ** (-4 / 3)
        + c * (-2 / 3) * (volume) ** (-5 / 3)
        + d * (-1) * (volume) ** (-2)
        + e * (-4 / 3) * (volume) ** (-7 / 3)
    )
    return energy_derivative


def mBM5_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Second derivative of mBM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        b * (4 / 9) * (volume) ** (-7 / 3)
        + c * (10 / 9) * (volume) ** (-8 / 3)
        + d * (2) * (volume) ** (-3)
        + e * (28 / 9) * (volume) ** (-10 / 3)
    )
    return energy_derivative2


def mBM5_eos_parameters(
    volume_range: np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, d, and e.

    Args:
        volume_range (np.ndarray): range of volumes
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = fsolve(mBM5_derivative, np.mean(volume_range), args=(b, c, d, e))[0]
    E0 = mBM5_equation(V0, a, b, c, d, e)
    B = (
        (28 * e) / (9 * V0 ** (10 / 3))
        + (2 * d) / V0**3
        + (10 * c) / (9 * V0 ** (8 / 3))
        + (4 * b) / (9 * V0 ** (7 / 3))
    ) * V0
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (98 * e + 54 * d * V0 ** (1 / 3) + 25 * c * V0 ** (2 / 3) + 8 * b * V0) / (
        42 * e + 27 * d * V0 ** (1 / 3) + 15 * c * V0 ** (2 / 3) + 6 * b * V0
    )
    B2P = (
        V0 ** (8 / 3)
        * (
            9 * d * (14 * e + 5 * c * V0 ** (2 / 3) + 8 * b * V0)
            + 2
            * V0 ** (1 / 3)
            * (126 * b * e * V0 ** (1 / 3) + 5 * c * (28 * e + b * V0))
        )
    ) / (2 * (14 * e + 9 * d * V0 ** (1 / 3) + 5 * c * V0 ** (2 / 3) + 2 * b * V0) ** 3)
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA

    return V0, E0, B, BP, B2P


def mBM5(
    volume: float | np.ndarray, energy: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the mBM5 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d, e = curve_fit(
        mBM5_equation, volume, energy, p0=[100, 100, 100, 100, 100]
    )[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = mBM5_equation(volume_range, a, b, c, d, e)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * mBM5_derivative(volume_range, b, c, d, e)
    )
    V0, E0, B, BP, B2P = mBM5_eos_parameters(volume_range, a, b, c, d, e)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, e])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# BM4 EOS Functions
def BM4_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float
) -> float | np.ndarray:
    """BM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: energy(s)
    """

    energy = (
        a + b * volume ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2)
    )
    return energy


def BM4_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Derivative of BM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b * (-2 / 3) * volume ** (-5 / 3)
        + c * (-4 / 3) * (volume) ** (-7 / 3)
        + d * (-2) * (volume) ** (-3)
    )
    return energy_derivative


def BM4_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Second derivative of BM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        b * (10 / 9) * volume ** (-8 / 3)
        + c * (28 / 9) * (volume) ** (-10 / 3)
        + d * (6) * (volume) ** (-4)
    )
    return energy_derivative2


def BM4_eos_parameters(
    a: float, b: float, c: float, d: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, and d.

    Args:
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = np.sqrt(
        -(
            (
                4 * c**3
                - 9 * b * c * d
                + np.sqrt((c**2 - 3 * b * d) * (4 * c**2 - 3 * b * d) ** 2)
            )
            / b**3
        )
    )
    E0 = BM4_equation(V0, a, b, c, d)
    B = (2 * (27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0**2)) / (
        9 * V0 ** (11 / 3)
    )
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (243 * d * V0 ** (2 / 3) + 98 * c * V0 ** (4 / 3) + 25 * b * V0**2) / (
        81 * d * V0 ** (2 / 3) + 42 * c * V0 ** (4 / 3) + 15 * b * V0**2
    )
    B2P = (
        4
        * V0 ** (13 / 3)
        * (
            27 * d * (7 * c * V0 ** (4 / 3) + 10 * b * V0**2)
            + V0 ** (2 / 3) * (7 * c * (5 * b * V0**2))
        )
    ) / (27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0**2) ** 3
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA

    return V0, E0, B, BP, B2P


def BM4(
    volume: float | np.ndarray, energy: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the BM4 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d = curve_fit(BM4_equation, volume, energy, p0=[100, 100, 100, 100])[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = BM4_equation(volume_range, a, b, c, d)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * BM4_derivative(volume_range, b, c, d)
    )
    V0, E0, B, BP, B2P = BM4_eos_parameters(a, b, c, d)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, 0])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# BM5 EOS Functions
def BM5_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """BM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        a
        + b * (volume) ** (-2 / 3)
        + c * (volume) ** (-4 / 3)
        + d * (volume) ** (-2)
        + e * (volume) ** (-8 / 3)
    )
    return energy


def BM5_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Derivative of BM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b * (-2 / 3) * (volume) ** (-5 / 3)
        + c * (-4 / 3) * (volume) ** (-7 / 3)
        + d * (-2) * (volume) ** (-3)
        + e * (-8 / 3) * (volume) ** (-11 / 3)
    )
    return energy_derivative


def BM5_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Second derivative of BM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        b * (10 / 9) * volume ** (-8 / 3)
        + c * (28 / 9) * (volume) ** (-10 / 3)
        + d * (6) * (volume) ** (-4)
        + e * (88 / 9) * (volume) ** (-14 / 3)
    )
    return energy_derivative2


def BM5_eos_parameters(
    volume_range: np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, d, and e.

    Args:
        volume_range (np.ndarray): range of volumes
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = fsolve(BM5_derivative, np.mean(volume_range), args=(b, c, d, e))[0]
    E0 = BM5_equation(V0, a, b, c, d, e)
    B = (
        2 * (44 * e + 27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0**2)
    ) / (9 * V0 ** (11 / 3))
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (
        484 * e + 243 * d * V0 ** (2 / 3) + 98 * c * V0 ** (4 / 3) + 25 * b * V0**2
    ) / (132 * e + 81 * d * V0 ** (2 / 3) + 42 * c * V0 ** (4 / 3) + 15 * b * V0**2)
    B2P = (
        4
        * V0 ** (13 / 3)
        * (
            27 * d * (22 * e + 7 * c * V0 ** (4 / 3) + 10 * b * V0**2)
            + V0 ** (2 / 3)
            * (990 * b * e * V0 ** (2 / 3) + 7 * c * (176 * e + 5 * b * V0**2))
        )
    ) / (
        44 * e + 27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0**2
    ) ** 3
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA

    return V0, E0, B, BP, B2P


def BM5(
    volume: float | np.ndarray, energy: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the BM5 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d, e = curve_fit(
        BM5_equation, volume, energy, p0=[100, 100, 100, 100, 100]
    )[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = BM5_equation(volume_range, a, b, c, d, e)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * BM5_derivative(volume_range, b, c, d, e)
    )
    V0, E0, B, BP, B2P = BM5_eos_parameters(volume_range, a, b, c, d, e)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, e])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# LOG4 EOS Functions
def LOG4_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float
) -> float | np.ndarray:
    """LOG4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = a + b * np.log(volume) + c * np.log(volume) ** 2 + d * np.log(volume) ** 3
    return energy


def LOG4_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Derivative of LOG4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b + 2 * c * np.log(volume) + 3 * d * np.log(volume) ** 2
    ) / volume
    return energy_derivative


def LOG4_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Second derivative of LOG4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        -b / (volume**2)
        - 2 * c * (np.log(volume) - 1) / (volume**2)
        - (3 * d * (np.log(volume) - 2) * np.log(volume)) / (volume**2)
    )
    return energy_derivative2


def LOG4_eos_parameters(
    volume_range: np.ndarray, a: float, b: float, c: float, d: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, and d.

    Args:
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = fsolve(LOG4_derivative, np.mean(volume_range), args=(b, c, d))[0]
    E0 = LOG4_equation(V0, a, b, c, d)
    B = -((b - 2 * c + 2 * (c - 3 * d) * np.log(V0) + 3 * d * np.log(V0) ** 2) / V0)
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (
        b - 4 * c + 6 * d + 2 * (c - 6 * d) * np.log(V0) + 3 * d * np.log(V0) ** 2
    ) / (b - 2 * c + 2 * (c - 3 * d) * np.log(V0) + 3 * d * np.log(V0) ** 2)
    B2P = (
        2
        * V0
        * (
            2 * c**2
            - 3 * b * d
            + 18 * d**2
            - 6 * c * d
            + 6 * (c * d - 3 * d**2) * np.log(V0)
            + 9 * d**2 * np.log(V0) ** 2
        )
    ) / (b - 2 * c + 2 * (c - 3 * d) * np.log(V0) + 3 * d * np.log(V0) ** 2) ** 3
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA
    return V0, E0, B, BP, B2P


def LOG4(
    volume: float | np.ndarray, energy: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the LOG4 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d = curve_fit(LOG4_equation, volume, energy, p0=[100, 100, 100, 100])[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = LOG4_equation(volume_range, a, b, c, d)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * LOG4_derivative(volume_range, b, c, d)
    )
    V0, E0, B, BP, B2P = LOG4_eos_parameters(volume_range, a, b, c, d)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, 0])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# LOG5 EOS Functions
def LOG5_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """LOG5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        a
        + b * np.log(volume)
        + c * np.log(volume) ** 2
        + d * np.log(volume) ** 3
        + e * np.log(volume) ** 4
    )
    return energy


def LOG5_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Derivative of LOG5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy = (
        b
        + 2 * c * np.log(volume)
        + 3 * d * np.log(volume) ** 2
        + 4 * e * np.log(volume) ** 3
    ) / volume
    return energy


def LOG5_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Second derivative of LOG5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        -b / (volume**2)
        - 2 * c * (np.log(volume) - 1) / (volume**2)
        - (3 * d * (np.log(volume) - 2) * np.log(volume)) / (volume**2)
        - (4 * e * (np.log(volume) - 3) * np.log(volume) ** 2) / (volume**2)
    )
    return energy_derivative2


def LOG5_eos_parameters(
    volume_range: np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, d, and e.

    Args:
        volume_range (np.ndarray): range of volumes
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = fsolve(LOG5_derivative, np.mean(volume_range), args=(b, c, d, e))[0]
    E0 = LOG5_equation(V0, a, b, c, d, e)

    B = -(
        (
            b
            - 2 * c
            + 2 * (c - 3 * d) * np.log(V0)
            + 3 * (d - 4 * e) * np.log(V0) ** 2
            + 4 * e * np.log(V0) ** 3
        )
        / V0
    )
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (
        b
        - 4 * c
        + 6 * d
        + 2 * (c - 6 * d + 12 * e) * np.log(V0)
        + 3 * (d - 8 * e) * np.log(V0) ** 2
        + 4 * e * np.log(V0) ** 3
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * np.log(V0)
        + 3 * (d - 4 * e) * np.log(V0) ** 2
        + 4 * e * np.log(V0) ** 3
    )
    B2P = (
        2
        * V0
        * (
            2 * c**2
            - 3 * b * d
            + 18 * d**2
            + 12 * b * e
            - 6 * c * (d + 4 * e)
            + 6 * (c * d - 3 * d**2 - 2 * b * e + 12 * d * e) * np.log(V0)
            + 9 * (d - 4 * e) ** 2 * np.log(V0) ** 2
            + 24 * (d - 4 * e) * e * np.log(V0) ** 3
            + 24 * e**2 * np.log(V0) ** 4
        )
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * np.log(V0)
        + 3 * (d - 4 * e) * np.log(V0) ** 2
        + 4 * e * np.log(V0) ** 3
    ) ** 3
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA
    return V0, E0, B, BP, B2P


def LOG5(
    volume: float | np.ndarray, energy: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the LOG5 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d, e = curve_fit(
        LOG5_equation, volume, energy, p0=[100, 100, 100, 100, 100]
    )[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = LOG5_equation(volume_range, a, b, c, d, e)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * LOG5_derivative(volume_range, b, c, d, e)
    )
    V0, E0, B, BP, B2P = LOG5_eos_parameters(volume_range, a, b, c, d, e)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, e])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# Murnaghan EOS Functions
def murnaghan_equation(
    volume: float | np.ndarray, V0: float, E0: float, B: float, BP: float
) -> float | np.ndarray:
    """Murnaghan EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        E0 (float): equilibrium energy
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        E0
        - (B * V0) / (BP - 1)
        + (B * volume / BP) * (1 + (V0 / volume) ** BP / (BP - 1))
    )
    return energy


def murnaghan_derivative(
    volume: float | np.ndarray, V0: float, B: float, BP: float
) -> float | np.ndarray:
    """Derivative of Murnaghan EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (B / BP) * (1 - (V0 / volume) ** BP)
    return energy_derivative


def murnaghan(
    volume: float | np.ndarray, energy: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the Murnaghan EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    volume_range = np.linspace(min(volume), max(volume), 1000)

    [eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos] = mBM4(
        volume, energy
    )
    initial_guess = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[2] / EV_PER_CUBIC_ANGSTROM_TO_GPA,
        eos_parameters[3],
    ]

    V0, E0, B, BP = curve_fit(murnaghan_equation, volume, energy, p0=initial_guess)[0]

    energy_eos = murnaghan_equation(volume_range, V0, E0, B, BP)
    eos_parameters = np.array([V0, E0, B * EV_PER_CUBIC_ANGSTROM_TO_GPA, BP, 0])
    eos_constants = np.array([0, 0, 0, 0, 0])
    pressure_eos = (
        -1
        * EV_PER_CUBIC_ANGSTROM_TO_GPA
        * murnaghan_derivative(volume_range, V0, B, BP)
    )

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# Vinet EOS Functions
def vinet_equation(
    volume: float | np.ndarray, V0: float, E0: float, B: float, BP: float
) -> float | np.ndarray:
    """Vinet EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        E0 (float): equilibrium energy
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        E0
        + (4 * B * V0) / (BP - 1) ** 2
        - (4 * B * V0)
        / (BP - 1) ** 2
        * (1 - 3 / 2 * (BP - 1) * (1 - (volume / V0) ** (1 / 3)))
        * (np.exp(3 / 2 * (BP - 1) * (1 - (volume / V0) ** (1 / 3))))
    )
    return energy


def vinet_derivative(volume: float | np.ndarray, V0, B, BP) -> float | np.ndarray:
    """Derivative of Vinet EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = -(
        3
        * B
        * (1 - (volume / V0) ** (1 / 3))
        / ((volume / V0) ** (2 / 3))
        * np.exp(3 / 2 * (BP - 1) * (1 - (volume / V0) ** (1 / 3)))
    )
    return energy_derivative


def vinet(
    volume: float | np.ndarray, energy: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the Vinet EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    volume_range = np.linspace(min(volume), max(volume), 1000)

    [eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos] = mBM4(
        volume, energy
    )
    initial_guess = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[2] / EV_PER_CUBIC_ANGSTROM_TO_GPA,
        eos_parameters[3],
    ]

    V0, E0, B, BP = curve_fit(vinet_equation, volume, energy, p0=initial_guess)[0]

    energy_eos = vinet_equation(volume_range, V0, E0, B, BP)

    B2P = (19 - 18 * BP - 9 * BP**2) / (36 * B)
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([0, 0, 0, 0, 0])
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * vinet_derivative(volume_range, V0, B, BP)
    )

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# Morse EOS Functions
def morse_equation(
    volume: float | np.ndarray, V0: float, E0: float, B: float, BP: float
) -> float | np.ndarray:
    """Morse EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        E0 (float): equilibrium energy
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: energy
    """

    a = E0 + (9 * B * V0) / (2 * (BP - 1) ** 2)
    b = (-9 * B * V0 * np.exp(BP - 1)) / (BP - 1) ** 2
    c = (9 * B * V0 * np.exp(2 * BP - 2)) / (2 * (BP - 1) ** 2)
    d = (1 - BP) / V0 ** (1 / 3)
    energy = (
        a + b * np.exp(d * volume ** (1 / 3)) + c * np.exp(2 * d * volume ** (1 / 3))
    )
    return energy


def morse_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Derivative of Morse EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = b * d * np.exp(d * volume ** (1 / 3)) / (
        3 * volume ** (2 / 3)
    ) + 2 * c * d * np.exp(2 * d * volume ** (1 / 3)) / (3 * volume ** (2 / 3))
    return energy_derivative


def morse(
    volume: float | np.ndarray, energy: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the Morse EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    volume_range = np.linspace(min(volume), max(volume), 1000)

    [eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos] = mBM4(
        volume, energy
    )
    initial_guess = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[2] / EV_PER_CUBIC_ANGSTROM_TO_GPA,
        eos_parameters[3],
    ]

    V0, E0, B, BP = curve_fit(morse_equation, volume, energy, p0=initial_guess)[0]

    energy_eos = morse_equation(volume_range, V0, E0, B, BP)

    B2P = (5 - 5 * BP - 2 * BP**2) / (9 * B)
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    eos_parameters = np.array([V0, E0, B, BP, B2P])

    a = E0 + (9 * B * V0) / (2 * (BP - 1) ** 2)
    b = (-9 * B * V0 * np.exp(BP - 1)) / (BP - 1) ** 2
    c = (9 * B * V0 * np.exp(2 * BP - 2)) / (2 * (BP - 1) ** 2)
    d = (1 - BP) / V0 ** (1 / 3)
    eos_constants = np.array([a, b, c, d, 0])

    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * morse_derivative(volume_range, b, c, d)
    )

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


def fit_to_all_eos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fits the volume and energies of configurations to all EOS functions and returns the results in a dataframe.

    Args:
        df: Dataframe with headers ['config', 'volume', 'energy', 'number_of_atoms'].

    Returns:
        tuple(eos_df, eos_parameters_df)
    """

    eos_functions = [mBM4, mBM5, BM4, BM5, LOG4, LOG5, murnaghan, vinet, morse]
    dataframes = []

    for config in df["config"].unique():
        config_df = df[df["config"] == config]
        volumes = config_df["volume"].values
        energies = config_df["energy"].values
        number_of_atoms = config_df["number_of_atoms"].values

        for eos_function in eos_functions:
            eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = (
                eos_function(volumes, energies)
            )
            eos_name = eos_function.__name__

            dataframes.append(
                pd.DataFrame(
                    [
                        [
                            config,
                            eos_name,
                            eos_constants[0],
                            eos_constants[1],
                            eos_constants[2],
                            eos_constants[3],
                            eos_constants[4],
                            eos_parameters[0],
                            eos_parameters[1],
                            eos_parameters[2],
                            eos_parameters[3],
                            eos_parameters[4],
                            volume_range,
                            energy_eos,
                            pressure_eos,
                            number_of_atoms,
                        ]
                    ],
                    columns=[
                        "config",
                        "EOS",
                        "a",
                        "b",
                        "c",
                        "d",
                        "e",
                        "V0",
                        "E0",
                        "B",
                        "BP",
                        "B2P",
                        "volumes",
                        "energies",
                        "pressures",
                        "number_of_atoms",
                    ],
                )
            )

    eos_df = pd.concat(dataframes, ignore_index=True)
    eos_parameters_df = eos_df.drop(
        columns=["volumes", "energies", "pressures", "number_of_atoms"]
    )

    return eos_df, eos_parameters_df


# TODO: Not sure if a list of dataframe will actually work. the function simply concats them all together
def plot_mv(df: pd.DataFrame, show_fig: bool = True) -> go.Figure:
    """Plot the magnetic moment vs volume

    Args:
        df (pd.DataFrame): single pandas data frame or a list of pandas dataframes with columns ['config', '# of ion', 'volume', 'tot']
        show_fig (bool, optional): Defaults to True.

    Returns:
        go.Figure: plotly figure
    """

    # Create a new dataframe where each 'mag_data' dataframe is associated with its corresponding 'volume' and 'config' values
    df_new = pd.concat(
        [
            df_mag.assign(volume=v, config=c)
            for v, c, df_mag in zip(df["volume"], df["config"], df["mag_data"])
        ]
    )
    fig = px.line(
        df_new.sort_values(
            [
                "# of ion",
                "volume",
            ]
        ),
        x="volume",
        y="tot",
        color="# of ion",
        symbol="# of ion",
        hover_data=["config", "# of ion", "volume", "tot"],
        template="plotly_white",
    )
    fig.update_layout(xaxis_title="Volume [A^3]", yaxis_title="Magnetic Moment [mu_B]")
    fig.update_yaxes(nticks=10)
    fig.update_xaxes(nticks=10)

    # Loop over each trace and update dash length
    for i, trace in enumerate(fig.data):
        dash_length = (
            f"{2+(i+1)}px,{2+2*(i+1)}px"  # Dash length changes with each iteration
        )
        fig.data[-i - 1].update(
            mode="markers+lines",
            marker=dict(size=8, line=dict(width=1), opacity=0.5),
            line=dict(width=3, dash=dash_length),
        )

    if show_fig:
        fig.show()
    return fig


def assign_colors_to_configs(
    df: pd.DataFrame, alpha: float = 1, cmap: str = "plotly"
) -> dict:
    unique_configs = df["config"].unique()

    if cmap == "plotly":
        colors = px.colors.qualitative.Plotly
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        colors = [
            f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})"
            for color in colors
        ]

    elif cmap == "distinctipy":
        colors = get_colors(len(unique_configs))
        colors = [
            f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})" for color in colors
        ]
    else:
        print("cmap must be 'plotly' or 'distinctipy'")

    config_colors = {
        config: colors[i % len(colors)] for i, config in enumerate(unique_configs)
    }
    return config_colors


def assign_marker_symbols_to_configs(df: pd.DataFrame) -> dict:
    unique_configs = df["config"].unique()
    symbols = [
        "circle",
        "square",
        "diamond",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "octagon",
        "star",
        "hexagram",
        "star-triangle-up",
        "star-triangle-down",
        "star-square",
        "star-diamond",
        "diamond-tall",
        "diamond-wide",
        "hourglass",
        "bowtie",
        "circle-cross",
        "circle-x",
        "square-cross",
        "square-x",
        "diamond-cross",
        "diamond-x",
        "cross-thin",
        "x-thin",
        "asterisk",
        "hash",
        "y-up",
        "y-down",
        "y-left",
        "y-right",
        "line-ew",
        "line-ns",
        "line-ne",
        "line-nw",
    ]
    config_symbols = {
        config: symbols[i % len(symbols)] for i, config in enumerate(unique_configs)
    }
    return config_symbols


def plot_ev(
    data,
    eos_fitting="BM4",
    highlight_minimum=True,
    per_atom=False,
    title=None,
    show_fig=True,
    cmap="plotly",
    marker_alpha=1,
    marker_size=10,
):
    """Plot the energy vs volume curves for each configuration.

    Args:
        data (pandas.DataFrame, list of pandas.DataFrame, or list of str): Data must be a pandas
        DataFrame or a list of pandas DataFrames.
        eos_fitting (str, optional): EOS name. Defaults to "BM4".
        highlight_minimum (bool, optional): Defaults to True.
        per_atom (bool, optional):Defaults to False.
        title (_type_, optional): Defaults to None.
        show_fig (bool, optional): Defaults to True.
        left_col (str, optional): Defaults to "volume".
        right_col (str, optional): Defaults to "energy".
        cmap (str, optional): Defaults to 'plotly'.
        marker_alpha (int, optional): Defaults to 1.
        marker_size (int, optional): Defaults to 10.

    Returns:
        fig (plotly.graph_objs._figure.Figure): A Plotly figure.
    """

    # Check if data is a pandas DataFrame or a list of pandas DataFrames
    if isinstance(data, pd.DataFrame):
        df = data

    # Check if each element of the list is the same type as the zeroth element
    elif isinstance(data, list) and all(
        type(element) == type(data[0]) for element in data
    ):
        if isinstance(data[0], pd.DataFrame):
            df = pd.concat(data, ignore_index=True)

    else:
        raise ValueError(
            "data must be a pandas DataFrame, list of pandas DataFrames, or a list of input_file names as strings"
        )

    # Create a data frame with the eos fits for each config
    if eos_fitting != None:
        eos_df, _ = fit_to_all_eos(df)

    # Assign colors and symbols
    config_colors = assign_colors_to_configs(df, alpha=marker_alpha, cmap=cmap)
    config_symbols = assign_marker_symbols_to_configs(df)

    # Plot the data
    fig = go.Figure()
    fig.update_layout(
        font=dict(
            family="Devaju Sans",
        )
    )

    for config in df["config"].unique():
        config_df = df[df["config"] == config]

        if isinstance(per_atom, bool):
            x = config_df["volume"]
            y = config_df["energy"]

            if per_atom:
                x = x / config_df["number_of_atoms"]
                y = y / config_df["number_of_atoms"]
        else:
            raise ValueError("per_atom must be True or False")

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=config_colors[config],
                    symbol=config_symbols[config],
                ),
                legendgroup="EOS",
                name=f"{config}",
            )
        )

    if isinstance(per_atom, bool):
        atom_suffix = "/atom" if per_atom else ""
        fig.update_xaxes(
            title=dict(
                text=f"Volume (Å<sup>3</sup>{atom_suffix})",
                font=dict(size=22, color="rgb(0,0,0)"),
            )
        )
        fig.update_yaxes(
            title=dict(
                text=f"Energy (eV{atom_suffix})", font=dict(size=22, color="rgb(0,0,0)")
            )
        )

    # Loop over configs in the eos data frame and plot the eos fits
    if eos_fitting != None:
        for config in eos_df["config"].unique():
            eos_config_df = eos_df[eos_df["config"] == config]
            if eos_fitting in eos_config_df["EOS"].unique():
                eos_name_df = eos_config_df[eos_config_df["EOS"] == eos_fitting]

                x = eos_name_df["volumes"].values[0]
                y = eos_name_df["energies"].values[0]

                if per_atom:
                    num_atoms = eos_name_df["number_of_atoms"].values[0][0]
                    x = x / num_atoms
                    y = y / num_atoms

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{eos_fitting} fit",
                        line=dict(width=1.75, color=config_colors[config]),
                        legendgroup="data",
                        showlegend=False,
                    )
                )

                # Plot the equilibrium energy and volume for each config
                if highlight_minimum == True:
                    min_energy = min(eos_name_df["energies"].values[0])
                    volume_at_min_energy = eos_name_df["volumes"].values[0][
                        np.where(eos_name_df["energies"].values[0] == min_energy)[0][0]
                    ]

                    x = eos_name_df["V0"].values[0]
                    y = eos_name_df["E0"].values[0]

                    if per_atom:
                        num_atoms = eos_name_df["number_of_atoms"].values[0][0]
                        x = x / num_atoms
                        y = y / num_atoms

                    fig.add_trace(
                        go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers",
                            name=f"{eos_fitting} min energy",
                            marker=dict(
                                color="black", size=marker_size, symbol="cross"
                            ),
                            legendgroup="minimum",
                            showlegend=False,
                        )
                    )

                elif highlight_minimum == False:
                    pass

                else:
                    raise ValueError("highlight_minimum must be True or False")

    axis_params = dict(
        showline=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
        mirror="allticks",
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        tickfont=dict(color="rgb(0,0,0)", size=20),
    )

    fig.update_layout(
        plot_bgcolor="white",
        width=840,
        height=600,
        legend=dict(font=dict(size=20, color="black")),
        xaxis=axis_params,
        yaxis=axis_params,
    )

    if title != None:
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(color="rgb(0,0,0)", size=30))
        )

    if show_fig:
        fig.show()

    return fig


def plot_energy_difference(
    df,
    reference_config,
    per_atom=False,
    show_fig=True,
    convert_to_mev=False,
    volume_precision=0.0001,
    title=None,
    marker_alpha=1,
    cmap="plotly",
    marker_size=10,
):
    """Takes a dataframe and plots the energy difference with respect to a reference configuration
    as a function of volume. Utilizes plot_ev() for the actual plotting.

    Args:
        df (pandas.DataFrame): dataframe containing the volumes, energies, and number of atoms of each
        configuration.
        reference_config (str): name of the configuration to be used as the reference state
        per_atom (bool, optional): Defaults to False.
        show_fig (bool, optional): Defaults to True.
        convert_to_mev (bool, optional): Defaults to False.
        title (_type_, optional): Defaults to None.
        marker_alpha (int, optional): Defaults to 1.
        cmap (str, optional): Defaults to 'plotly'.
        marker_size (int, optional): Defaults to 10.

    Returns:
        fig (plotly.graph_objs._figure.Figure): A Plotly figure.
    """

    df_list = []
    for config in df["config"].unique():
        df_list.append(df[df["config"] == config].reset_index(drop=True))
        if config == reference_config:
            reference_df = df[df["config"] == config].reset_index(drop=True)

    # Subtract reference energies
    missing_volumes = []
    for df_el in df_list:
        for i, row in df_el.iterrows():
            try:
                reference_energy = reference_df[
                    np.isclose(
                        reference_df["volume"], row["volume"], atol=volume_precision
                    )
                ]["energy"].values[0]
                df_el.at[i, "energy"] = row["energy"] - reference_energy
            except Exception as e:
                missing_volumes.append((row["config"], row["volume"]))
    if missing_volumes:
        print(f"Warning: Missing volumes for configurations: {missing_volumes}")

    energy_difference_df = pd.concat(df_list)

    if convert_to_mev == True:
        energy_difference_df["energy"] *= 1000

    # plot energy difference vs volume
    fig = plot_ev(
        energy_difference_df,
        eos_fitting=None,
        per_atom=per_atom,
        show_fig=False,
        title=title,
        cmap=cmap,
        marker_alpha=marker_alpha,
        marker_size=marker_size,
    )

    fig.update_layout(
        font=dict(
            family="Devaju Sans",
        )
    )

    unit = "meV" if convert_to_mev else "eV"
    per_atom_suffix = "/atom" if per_atom else ""
    title_text = f"ΔEnergy ({unit}{per_atom_suffix})"

    fig.update_yaxes(
        title=dict(text=title_text, font=dict(size=22, color="rgb(0,0,0)"))
    )

    # Plot a horizontal line a y=0
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=min(fig.data[0].x * 0.95),
        x1=max(fig.data[0].x * 1.05),
        y0=0,
        y1=0,
        line=dict(
            color="black",
            width=2,
            dash="dash",
        ),
    )
    if show_fig:
        fig.show()

    return fig


def plot_config_energy(
    df,
    max_rank=10000,
    inset_max_rank=10,
    show_inset=True,
    color_assignment=None,
    show_fig=True,
):
    data = []
    for i, el in enumerate([max_rank, inset_max_rank]):
        new_df = df
        new_df["energy_per_atom"] = new_df["energy"] / new_df["number_of_atoms"]
        new_df = df.nsmallest(el + 1, "energy_per_atom").copy()
        new_df["energy_difference"] = (
            new_df["energy_per_atom"] - new_df["energy_per_atom"].min()
        ) * 1000
        new_df = new_df.reset_index(drop=True)
        new_df["rank"] = new_df["energy_difference"].rank(method="min") - 1
        max_energy_difference = new_df["energy_difference"].max()
        # Get the order of magnitude of the max_energy_difference
        rounding_order_of_magnitude = 10 ** (len(str(int(max_energy_difference))) - 2)

        if i == 0:
            types_of_magnetic_ordering = new_df["magnetic_ordering"].unique()
        # add a new color column to new_df that corresponds to the magnetic ordering
        colors = px.colors.qualitative.Plotly.copy()  # plotly colors
        print(types_of_magnetic_ordering)
        if color_assignment == None:
            assignment = zip(types_of_magnetic_ordering, colors)
        else:
            assignment = color_assignment
        new_df["color"] = new_df["magnetic_ordering"].map(dict(assignment))

        if i == 0:
            for trace_number, mo in enumerate(types_of_magnetic_ordering):
                single_mo_df = new_df[new_df["magnetic_ordering"] == mo]
                data.append(
                    go.Scatter(
                        x=single_mo_df["rank"],
                        y=single_mo_df["energy_difference"],
                        mode="markers",
                        marker=dict(
                            size=7,
                            symbol="cross-thin-open",
                            color=single_mo_df["color"],
                        ),
                        hovertext=[
                            f"config={config}, <br>magnetic ordering={mo}"
                            for config, mo in zip(
                                single_mo_df["config"],
                                single_mo_df["magnetic_ordering"],
                            )
                        ],
                        name=mo,
                    )
                )
        else:
            data.append(
                go.Scatter(
                    x=new_df["rank"],
                    y=new_df["energy_difference"],
                    xaxis="x2",
                    yaxis="y2",
                    mode="markers",
                    marker=dict(
                        size=7, symbol="cross-thin-open", color=new_df["color"]
                    ),
                    hovertext=[
                        f"config={config}, <br>magnetic ordering={mo}"
                        for config, mo in zip(
                            new_df["config"], new_df["magnetic_ordering"]
                        )
                    ],
                    name=mo,
                )
            )
    for el in data[trace_number + 1 :]:
        el.showlegend = False  # don't show legend for traces in inset data
    layout = go.Layout(
        font=dict(family="Devaju Sans", size=20, color="black"),
        xaxis=dict(
            title="Energy rank",
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            mirror="allticks",
            tickwidth=1,
            tickcolor="black",
            showgrid=False,
            tickfont=dict(color="rgb(0,0,0)", size=20),
        ),
        yaxis=dict(
            title="Energy difference (meV/atom)",
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            mirror="allticks",
            tickwidth=1,
            tickcolor="black",
            showgrid=False,
            tickfont=dict(color="rgb(0,0,0)", size=20),
        ),
        xaxis2=dict(
            domain=[0.1, 0.5],
            anchor="y2",
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            mirror="allticks",
            tickwidth=1,
            tickcolor="black",
            showgrid=False,
            tickfont=dict(color="rgb(0,0,0)", size=20),
        ),
        yaxis2=dict(
            domain=[0.55, 0.95],
            anchor="x2",
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            mirror="allticks",
            tickwidth=1,
            tickcolor="black",
            showgrid=False,
            tickfont=dict(color="rgb(0,0,0)", size=20),
        ),
        plot_bgcolor="white",
        width=680,
        height=600,
        margin=dict(l=80, r=30, t=30, b=80),
        showlegend=True,
    )
    if show_inset == False:
        for i in range(trace_number + 1, len(data)):
            data.pop(i)
    fig = go.Figure(data=data, layout=layout)

    if show_fig:
        fig.show()
    return fig


def plot_energy_histogram(df, nbins=None, show_fig=True):
    try:
        new_df = df.drop("# of ion", axis=1)
        new_df = new_df.drop("tot", axis=1)
        new_df = new_df.drop_duplicates()
        new_df["energy_per_atom"] = new_df["energy"] / new_df["number_of_atoms"]
    except Exception as e:
        print("possible error. Could not strip magnetic data: ", e)
        new_df["energy_per_atom"] = new_df["energy"] / new_df["number_of_atoms"]
    new_df["relative_energy"] = (
        new_df["energy_per_atom"] - new_df["energy_per_atom"].min()
    ) * 1000
    fig = px.histogram(
        new_df, x="relative_energy", nbins=nbins, template="plotly_white"
    )
    print(new_df)
    if show_fig:
        fig.show()
