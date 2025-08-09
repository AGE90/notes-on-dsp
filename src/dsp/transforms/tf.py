from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import as_strided


def stft(
    signal: np.ndarray,
    fs: float,
    window_len: int = 256,
    overlap: float = 0.5,
    nfft: int = None, # type: ignore
    window_fn: Callable = np.hanning,
    center=True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficient Short-Time Fourier Transform (STFT) with optional centering and zero-padding.

    Parameters:
    -----------
    signal : np.ndarray
        Input time-domain signal as 1D numpy array.
    fs : float
        Sampling frequency (Hz).
    window_len : int
        Window length in samples.
    overlap : float
        Overlap fraction between 0 and 1.
    nfft : int
        FFT length (default: next power of 2 >= window_len).
    window_fn : Callable
        Window function generator (default: np.hanning).
    center : bool
        If True, zero-pad so windows are centered (like Algorithm 02).

    Returns:
    --------
    stft_matrix : np.ndarray
        Complex STFT result (freq_bins x time_frames) as 2D numpy array.
    t : np.ndarray
        Time vector for frame centers.
    f : np.ndarray
        Frequency vector.
    """

    # Ensure array
    is_real = np.isrealobj(signal)
    signal = np.asarray(signal)

    # Zero-pad if centering
    if center:
        pad = window_len // 2
        signal = np.pad(signal, (pad, pad), mode='constant')

    hop_size = int(window_len * (1 - overlap))
    if nfft is None:
        nfft = int(2**np.ceil(np.log2(window_len)))

    # Create window
    window = window_fn(window_len)

    # Number of frames
    num_frames = 1 + (len(signal) - window_len) // hop_size

    # Stride trick for efficient framing
    shape = (num_frames, window_len)
    strides = (signal.strides[0] * hop_size, signal.strides[0])
    frames = as_strided(signal, shape=shape, strides=strides)

    # Apply window
    frames *= window[None, :]

    # FFT: rfft for real, fft for complex
    if is_real:
        stft_matrix = np.fft.rfft(frames, n=nfft, axis=1)
        f = np.fft.rfftfreq(nfft, 1/fs)
    else:
        stft_matrix = np.fft.fft(frames, n=nfft, axis=1)
        f = np.fft.fftfreq(nfft, 1/fs)

    # Time and frequency axes
    t = np.arange(num_frames) * hop_size / fs
    if center:
        t -= (window_len / 2) / fs  # Correct for centering shift

    return stft_matrix.T, t, f


# Example usage
if __name__ == "__main__":
    FS = 1000
    time_span = np.linspace(0, 1, FS, endpoint=False)
    x = np.sin(2*np.pi*50*time_span) + np.sin(2*np.pi*120*time_span)

    S, tt, ff = stft(x, fs=FS, window_len=128, overlap=0.5)

    import matplotlib.pyplot as plt
    plt.pcolormesh(tt, ff, 20*np.log10(np.abs(S)+1e-6), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram')
    plt.colorbar(label='Magnitude [dB]')
    plt.show()
