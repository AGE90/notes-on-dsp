"""
Frequency Modulated Signals
"""

import numpy as np


def cfsig(
    t: np.ndarray,
    f: float,
    amp: float = 1.0,
    phi0: float = 0.0,
) -> np.ndarray:
    """
    Generates a complex valued sinusoid (Continuous Frequency Signal)

    Parameters
    ----------
    t : np.ndarray
        Time instants.
    f : float
        Frequency in [Hz].
    amp : float
        Amplitude of the signal, by default 1.0
    phi0 : float, optional
        Phase of the complex signal, by default 0.0

    Returns
    -------
    np.ndarray
        Complex valued signal
    """

    phi = phi0 + 2 * np.pi * f * t
    y = amp * np.exp(1j * (phi))

    return y


def lfmsig(
    t: np.ndarray,
    f0: float,
    f1: float,
    sweep_time: float,
    amp: float = 1.0,
    phi0: float = 0.0,
) -> np.ndarray:
    """
    Generates a complex valued linear frequency modulated signal.

    Parameters
    ----------
    t : np.ndarray
        Time instants.
    f0 : float
        Starting frequency in [Hz].
    f1 : float
        Final frequency in [Hz].
    sweep_time : float
        Sweeping time.
    amp : float
        Amplitude of the signal, by default 1.0
    phi0 : float, optional
        Phase of the complex signal, by default 0.0

    Returns
    -------
    np.ndarray
        Complex valued signal
    """
    c = (f1 - f0) / sweep_time
    phi = phi0 + 2 * np.pi * (0.5 * c * t**2 + f0 * t)
    y = amp * np.exp(1j * (phi))

    return y


def qfmsig(
    t: np.ndarray,
    f0: float,
    f1: float,
    sweep_time: float,
    amp: float = 1.0,
    phi0: float = 0.0,
) -> np.ndarray:
    """
    Generates a complex valued quadratic frequency modulated signal.

    Parameters
    ----------
    t : np.ndarray
        Time instants.
    f0 : float
        Starting frequency in [Hz].
    f1 : float
        Final frequency in [Hz].
    sweep_time : float
        Sweeping time.
    amp : float
        Amplitude of the signal, by default 1.0
    phi0 : float, optional
        Phase of the complex signal, by default 0.0

    Returns
    -------
    _type_
        Complex valued signal
    """
    c = (f1 - f0) / sweep_time**2
    phi = phi0 + 2 * np.pi * ((1 / 3) * c * t**3 + f0 * t)
    y = amp * np.exp(1j * (phi))

    return y


def efmsig(
    t: np.ndarray,
    f0: float,
    f1: float,
    sweep_time: float,
    amp: float = 1.0,
    phi0: float = 0.0,
) -> np.ndarray:
    """
    Generates a complex valued exponential frequency modulated signal.


    Parameters
    ----------
    t : np.ndarray
        Time instants.
    f0 : float
        Starting frequency in [Hz] (must be greater than 0).
    f1 : float
        Final frequency in [Hz].
    sweep_time : float
        Sweeping time.
    amp : float
        Amplitude of the signal, by default 1.0
    phi0 : float, optional
        Phase of the complex signal, by default 0.0

    Returns
    -------
    np.ndarray
        Complex valued signal
    """
    k = (f1 / f0) ** (1 / sweep_time)
    phi = phi0 + 2 * np.pi * f0 * ((k**t - 1) / np.log(k))
    y = amp * np.exp(1j * (phi))

    return y


def sfmsig(
    t: np.ndarray,
    f0: float,
    f1: float,
    fm: float,
    amp : float = 1.0,
    phi0: float = 0.0,
) -> np.ndarray:
    """
    Generates a complex valued sinusoidal frequency modulated signal.


    Parameters
    ----------
    t : np.ndarray
        Time instants.
    f0 : float
        Lower frequency in the modulation given in [Hz] (must be greater than 0).
    f1 : float
        Higher frequency in the modulation given in [Hz].
    fm : float
        Modulation frequency in [Hz].
    amp : float
        Amplitude of the signal, by default 1.0
    phi0 : float, optional
        Phase of the complex signal, by default 0.0

    Returns
    -------
    np.ndarray
        Complex valued signal
    """
    mod_amp = (f1 - f0) / 2
    fc = f0 + mod_amp
    phi = phi0 + 2 * np.pi * fc * t + (mod_amp / fm) * np.sin(2 * np.pi * fm * t)
    y = amp * np.exp(1j * (phi))

    return y
