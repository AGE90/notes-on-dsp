"""
Module to calculate the Fractional Fourier Transforms.
"""

import numpy as np
import scipy
import scipy.signal


def frft(f: np.ndarray, a: float) -> np.ndarray:
    """
    Compute the fast fractional Fourier transform (FRFT) of a signal.

    Parameters
    ----------
    f : np.ndarray
        Input signal to be transformed.
    a : float
        Fractional power (order) of the transform.

    Returns
    -------
    np.ndarray
        The transformed signal.

    References
    ----------
    .. [1] This algorithm implements `frft.m` from
       https://nalag.cs.kuleuven.be/research/software/FRFT/
    """
    # Use np.complex128 instead of deprecated np.complex
    ret = np.zeros_like(f, dtype=np.complex128)
    f = f.copy().astype(np.complex128)
    N = len(f)
    # Shift indices for FFT operations
    shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
    sN = np.sqrt(N)
    a = np.remainder(a, 4.0)

    # Handle special cases for fractional power
    if a == 0.0:
        return f
    if a == 2.0:
        return np.flipud(f)
    if a == 1.0:
        ret[shft] = np.fft.fft(f[shft]) / sN
        return ret
    if a == 3.0:
        ret[shft] = np.fft.ifft(f[shft]) * sN
        return ret

    # Reduce to interval 0.5 < a < 1.5
    if a > 2.0:
        a = a - 2.0
        f = np.flipud(f)
    if a > 1.5:
        a = a - 1
        f[shft] = np.fft.fft(f[shft]) / sN
    if a < 0.5:
        a = a + 1
        f[shft] = np.fft.ifft(f[shft]) * sN

    # General case for 0.5 < a < 1.5
    alpha = a * np.pi / 2
    tana2 = np.tan(alpha / 2)
    sina = np.sin(alpha)
    # Sinc interpolation and zero padding
    f = np.hstack((np.zeros(N - 1), sincinterp(f), np.zeros(N - 1))).T

    # Chirp premultiplication
    chrp = np.exp(-1j * np.pi / N * tana2 / 4 * np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
    f = chrp * f

    # Chirp convolution
    c = np.pi / N / sina / 4
    ret = scipy.signal.fftconvolve(
        np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
        f
    )
    ret = ret[4 * N - 4:8 * N - 7] * np.sqrt(c / np.pi)

    # Chirp post multiplication
    ret = chrp * ret

    # Normalizing constant
    ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]

    return ret


def ifrft(f: np.ndarray, a: float) -> np.ndarray:
    """
    Compute the inverse fast fractional Fourier transform (FRFT) of a signal.

    Parameters
    ----------
    f : np.ndarray
        Input signal to be transformed.
    a : float
        Fractional power (order) of the transform.

    Returns
    -------
    np.ndarray
        The transformed signal.
    """
    return frft(f, -a)


def sincinterp(x: np.ndarray) -> np.ndarray:
    """
    Sinc interpolation for signal upsampling.

    Parameters
    ----------
    x : np.ndarray
        Input signal to interpolate.

    Returns
    -------
    np.ndarray
        Interpolated signal.
    """
    N = len(x)
    y = np.zeros(2 * N - 1, dtype=x.dtype)
    y[:2 * N:2] = x
    # Convolve with sinc kernel for interpolation
    xint = scipy.signal.fftconvolve(
        y[:2 * N],
        np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
    )
    return xint[2 * N - 3: -2 * N + 3]
