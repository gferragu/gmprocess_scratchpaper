# -*- coding: utf-8 -*-

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# def make_test_signal(A=1, f=5, t_min=0, t_max=60, samp_interv=0.001):
#     # sampling_rate = 1 / samp_interv
#     nsamp = int((t_max - t_min) / samp_interv)
#     t = np.linspace(t_min, t_max, nsamp)

#     signal = A*np.sin((2*np.pi)*t)

#     return t, signal

def get_impulse(tmin=0, tmax=100, fs=1, delta=0.01):
    signal.unit_impulse(100)
    imp = signal.unit_impulse((tmax-tmin), 'mid')
    t = np.arange(tmin, tmax)

    return t, imp

def get_psd(series, fs=1.0, nfft=None, nperseg=10, noverlap=None, scaling='density'):
    if nfft == None and noverlap == None:
        nfft = len(series)
        nperseg = len(series)/100
        noverlap = nperseg/2
    freqs, psd = signal.welch(series,fs=fs,nfft=nfft,nperseg=nperseg,noverlap=noverlap,scaling=scaling)
    return freqs, psd

def get_ps(series, fs=1.0, nfft=None, nperseg=10, noverlap=None, scaling='spectrum'):
    freqs, ps = signal.welch(series,fs=fs,nfft=nfft,nperseg=nperseg,noverlap=noverlap,scaling=scaling)
    return freqs, ps

def psd_to_fft_sqrt(spectrum):
    noise = np.zeros(len(spectrum))
    for idx,power in enumerate(spectrum):
        noise[idx] = np.sqrt(power)
    return noise

def psd_to_fft_normalize(spectrum, fs=1):
    nfft = len(spectrum)
    delta = 1/fs
    norm = nfft/(2*delta)
    fft_norm = np.zeros(len(spectrum))
    for idx,power in enumerate(spectrum):
        fft_norm[idx] = np.sqrt(norm*power)
    return fft_norm

def fft_to_t(spectrum):
    t_series = np.fft.ifft(spectrum)
    return t_series

# def ifft_of_psd(series, fs=1.0, scaling='density'):
#     reqs, psd = signal.welch(series,fs,scaling=scaling)

#     np.fft.fft()
#     ifft_psd = np.fft.ifft(psd)

#     return ifft_psd


def plot_unit_amplitude_impulse(t = None, series=None, save=False, path='./', filename='impulse.png'):
    """


    Parameters
    ----------
    t : TYPE, optional
        DESCRIPTION. The default is None.
    series : TYPE, optional
        DESCRIPTION. The default is None.
    save : TYPE, optional
        DESCRIPTION. The default is False.
    path : TYPE, optional
        DESCRIPTION. The default is './'.
    filename : TYPE, optional
        DESCRIPTION. The default is 'impulse.png'.

    Returns
    -------
    None.

    """
    if series is None or t is None:
        t, imp = get_impulse()
    else:
        imp = series
        t = t
        if len(t) != len(series):
            print("signal and time arrays have differing lengths, exiting ...")
            return

    plt.plot(t, imp)
    plt.title("Unit Impulse")
    plt.margins(0.1, 0.1)
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    # plt.show()
    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/impulse/' + filename, dpi=500)
    plt.close()

def plot_unit_amplitude_impulse_fft(t = None, series = None, save=False, path='./', filename='impulse_fft.png'):
    """


    Parameters
    ----------
    t : TYPE, optional
        DESCRIPTION. The default is None.
    series : TYPE, optional
        DESCRIPTION. The default is None.
    save : TYPE, optional
        DESCRIPTION. The default is False.
    path : TYPE, optional
        DESCRIPTION. The default is './'.
    filename : TYPE, optional
        DESCRIPTION. The default is 'impulse_fft.png'.

    Returns
    -------
    None.

    """
    t, imp = get_impulse()
    fft_imp = np.abs(np.fft.fft(imp).real)
    freq = np.fft.fftfreq(len(fft_imp))

    plt.plot(freq, fft_imp)
    plt.title("NumPy FFT of Unit Impulse")
    plt.margins(0.1, 0.1)
    plt.xlabel('Period (s)')
    plt.ylabel('Amplitude')
    ax = plt.gca()
    ax.set_xscale('log')
    plt.grid(True)
    # plt.show()
    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/impulse/' + filename, dpi=500)
    plt.close()

def plot_unit_amplitude_impulse_psd(series, fs=1.0, scaling='density',save=False, path='./', filename='impulse_psd.png'):
    """
    Uses the SciPy Signal method "Welch" to construct the power spectral
    density from a time series using the method of Welch (1967)

    Units are in (m/s)^2 / Hz



    Parameters
    ----------
    series : TYPE
        DESCRIPTION.
    fs : TYPE, optional
        DESCRIPTION. The default is 1.0.
    scaling : TYPE, optional
        DESCRIPTION. The default is 'density'.

    Returns
    -------
    None.

    """
    freqs, psd = signal.welch(series,fs,scaling=scaling)

    plt.semilogx(freqs, psd)
    plt.margins(0.1, 0.1)
    plt.title('PSD: power spectral density - Scaling: "density"')
    plt.xlabel('Frequency')
    plt.ylabel('PSD [V**2/Hz]')
    # plt.tight_layout()
    plt.grid(True)
    # plt.show()
    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/impulse/' + filename, dpi=500)
    plt.close()

def plot_unit_amplitude_impulse_ps(series, fs=1.0, scaling='spectrum',save=False, path='./', filename='impulse_ps.png'):
    """
    Uses the SciPy Signal method "Welch" to construct the power spectral
    density from a time series using the method of Welch (1967).

    Units are in (m/s)^2 / Hz



    Parameters
    ----------
    series : TYPE
        DESCRIPTION.
    fs : TYPE, optional
        DESCRIPTION. The default is 1.0.
    scaling : TYPE, optional
        DESCRIPTION. The default is 'density'.

    Returns
    -------
    None.

    """
    freqs, psd = signal.welch(series,fs,scaling=scaling)

    plt.semilogx(freqs, psd)
    plt.margins(0.1, 0.1)
    plt.title('Power - Scaling: "spectrum"')
    plt.xlabel('Frequency')
    plt.ylabel('Linear Spectrum [V RMS]')
    # plt.tight_layout()
    plt.grid(True)
    # plt.show()
    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/impulse/' + filename, dpi=500)
    plt.close()

# def plot_noise(t = None, series = None, save=False, path='./', filename='impulse_fft.png'):

#     t, imp = get_impulse()
#     fft_imp = np.abs(np.fft.fft(imp).real)
#     freq = np.fft.fftfreq(len(fft_imp))


#     plt.plot(freq, fft_imp)
#     plt.margins(0.1, 0.1)
#     plt.xlabel('Period (s)')
#     plt.ylabel('Amplitude')
#     ax = plt.gca()
#     ax.set_xscale('log')
#     plt.grid(True)
#     plt.show()
#     if save:
#         plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/impulse/' + filename, dpi=500)
#     plt.close()

# def plot_noise_fft(spectrum = None, save=False, path='./', filename='impulse_fft.png'):


#     if noise is None:

#     noise_fft = np.abs(np.fft.fft(imp).real)
#     freq = np.fft.fftfreq(len(fft_imp))


#     plt.plot(freq, fft_imp)
#     plt.margins(0.1, 0.1)
#     plt.xlabel('Period (s)')
#     plt.ylabel('Amplitude')
#     ax = plt.gca()
#     ax.set_xscale('log')
#     plt.grid(True)
#     plt.show()
#     if save:
#         plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/impulse/' + filename, dpi=500)
#     plt.close()



# %%
# plt.close()

# # Make a test signal
# np.random.seed(0)
# delta = .01
# fs = 1/ delta
# t = np.arange(0, 70, delta)

# # A signal with a small frequency chirp
# sig = np.sin(0.5 * np.pi * t * (1 + .1 * t))

# # An unit impulse
# t, imp = get_impulse(tmin=0,tmax=100)

# plot_unit_amplitude_impulse(save=True)
# plot_unit_amplitude_impulse_fft(save=True)
# plot_unit_amplitude_impulse_psd(imp, save=True)
# plot_unit_amplitude_impulse_ps(imp, save=True)

# # plot_unit_amplitude_impulse_psd(sig,fs)


#%%

# # PSD of the impulse
# freqs_psd, psd = get_psd(imp)
# plt.semilogx(freqs_psd, psd)
# plt.title("Test plot of Impulse PSD")
# plt.show()
# plt.close()

# # PS of the impulse
# freqs_ps, ps = get_ps(imp)
# plt.semilogx(freqs_ps, ps)
# plt.title("Test plot of Impulse PS")
# plt.show()
# plt.close()

# # What does the ifft of the PSD look like?
# convert_psd_to_fft = psd_to_fft(psd)
# plt.semilogx(convert_psd_to_fft)
# plt.title("PSD to FFT by just using sqrt()")
# plt.show()
# plt.close()

# ifft_impulse_psd = fft_to_t(convert_psd_to_fft)
# plt.plot(ifft_impulse_psd)
# plt.title("Inverse FFT of sqrt(PSD)")
# plt.show()
# plt.close()

# # Convert directly from PSD to time domain
# psd_to_time = np.fft.ifft(psd)
# plt.plot(psd_to_time)
# plt.title("Inverse FFT of PSD (no square root)")
# plt.show()
# plt.close()

#%%
# Make a test signal
tmin=0
tmax=100
np.random.seed(0)
delta = .01
fs = 1/ delta
t = np.arange(tmin, tmax, delta)
wndw_factor=500
overlap_factor=2
nfft = len(t)
nperseg=len(t)/wndw_factor
noverlap=nperseg/overlap_factor

# A signal with a small frequency chirp
sig = np.sin(0.5 * np.pi * t * (1 + .1 * t))

# Just plot the input signal
plt.plot(t,sig)
plt.title("Original sine() test signal")
plt.show()
plt.close()

#%%
# Make unit impulse
tmin=0
tmax=100
delta = .01
fs = 1/ delta
t = np.arange(tmin, tmax, delta)
overlap_factor=2
nfft = len(t)
nperseg=len(t)/wndw_factor
noverlap=nperseg/overlap_factor

# A unit impulse test signal
t, sig = get_impulse(tmin=tmin, tmax=tmax, fs=fs, delta= delta)

# Just plot the input signal
plt.plot(t,sig)
plt.title("Unit Impulse")
plt.show()
plt.close()

#%%
# FFT of the test signal
# sine_fft = np.abs(np.fft.fft(np.sin(t)))
sine_fft = np.abs(np.fft.fft(np.sin(t)).real)
sine_freqs = np.fft.fftfreq(t.shape[-1], delta)
plt.semilogx(sine_freqs, sine_fft)
plt.title("Taking only real, positive components of the standard FFT")
plt.show()

# rFFT of the test signal (real)
# rsine_fft = np.abs(np.fft.rfft(np.sin(t)))
rsine_fft = np.abs(np.fft.rfft(np.sin(t)).real)
rsine_freqs = np.fft.rfftfreq(t.shape[-1], delta)
plt.semilogx(rsine_freqs, rsine_fft)
plt.title("Taking only real, positive components of the rFFT (redundant)")
plt.show()

# # hFFT of the test signal (hermitian)
# hsine_fft = np.fft.hfft(np.sin(t), n=nfft)        # Need to specify n
# hsine_freqs = np.fft.fftfreq(t.shape[-1], delta)
# plt.semilogx(hsine_freqs, hsine_fft)
# plt.show()

#%% Plot individually

# # PSD of the test signal
# freqs_psd, psd = get_psd(sig, fs=fs)
# plt.semilogx(freqs_psd, psd)
# plt.title("Test plot of a sine() signal's PSD")
# plt.show()
# plt.close()

# # PS of the impulse
# freqs_ps, ps = get_ps(sig)
# plt.semilogx(freqs_ps, ps)
# plt.title("Test plot of a sine() signal's PS")
# plt.show()
# plt.close()

# # What does the ifft of the PSD look like?
# convert_psd_to_fft = psd_to_fft_sqrt(psd)
# plt.semilogx(convert_psd_to_fft)
# plt.title("PSD to FFT by just using sqrt()")
# plt.show()
# plt.close()

# ifft_impulse_psd = fft_to_t(convert_psd_to_fft)
# plt.plot(ifft_impulse_psd)
# plt.title("Inverse FFT of sqrt(PSD)")
# plt.show()
# plt.close()

# # Using the PSD -> FFT normalization
# convert_psd_to_fft_norm = psd_to_fft_normalize(psd, fs=fs)
# plt.semilogx(convert_psd_to_fft_norm)
# plt.title("PSD to FFT by just using normalized sqrt()")
# plt.show()
# plt.close()

# ifft_impulse_psd_norm = fft_to_t(convert_psd_to_fft_norm)
# plt.plot(ifft_impulse_psd_norm)
# plt.title("Inverse FFT of normalized sqrt(PSD)")
# plt.show()
# plt.close()

# # Convert directly from PSD to time domain
# psd_to_time = np.fft.ifft(psd)
# plt.plot(psd_to_time)
# plt.title("Inverse FFT of PSD (no square root)")
# plt.show()
# plt.close()

#%% Plot together

# PSD of the test signal
freqs_psd, psd = get_psd(sig, fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
plt.semilogx(freqs_psd, psd,label="PSD")
# plt.title("Test plot of a sine() signal's PSD")
# PS of the impulse
freqs_ps, ps = get_ps(sig, fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
plt.semilogx(freqs_ps, ps,label="PS")
plt.title("Test plot of a sine() signal's PS")
plt.legend()
plt.show()

# What does the ifft of the PSD look like?
convert_psd_to_fft = psd_to_fft_sqrt(psd)
plt.semilogx(convert_psd_to_fft)
plt.title("PSD to FFT by just using sqrt()")
plt.show()
plt.close()

# Using the PSD -> FFT normalization
convert_psd_to_fft_norm = psd_to_fft_normalize(psd, fs=fs)
plt.semilogx(convert_psd_to_fft_norm)
plt.title("PSD to FFT by just using normalized sqrt()")
plt.show()
plt.close()

# What does the ifft of the PSD look like?
ifft_impulse_psd = fft_to_t(convert_psd_to_fft)
plt.plot(ifft_impulse_psd)
plt.title("Inverse FFT of sqrt(PSD)")
plt.show()
plt.close()

ifft_impulse_psd_norm = fft_to_t(convert_psd_to_fft_norm)
plt.plot(ifft_impulse_psd_norm)
plt.title("Inverse FFT of normalized sqrt(PSD)")
plt.show()
plt.close()

# Convert directly from PSD to time domain
psd_to_time = np.fft.ifft(psd)
plt.plot(psd_to_time)
plt.title("Inverse FFT of PSD (no square root)")
plt.show()
plt.close()