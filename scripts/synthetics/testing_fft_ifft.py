# -*- coding: utf-8 -*-

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

fig_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/spectra/"

#%%

def get_fft(series, fs, nfft):
    ft = np.fft.fft(series,fs)
    freq = np.fft.fftfreq(nfft)

    return freq, ft

def get_ifft(series, fs, nfft, real=True):
    if real:
        ift = np.fft.ifft(series,fs).real
    else:
        ift = np.fft.ifft(series,fs)


    return ift

def ps_psd_difference(ps,psd):
    diff = []
    if len(ps) == len(psd):
        diff = np.zeros(len(ps))
        for i in range(len(ps)):
            diff[i] = psd[i]/ps[i]
    else:
        print("Spectra must be of the same size")

    return diff

#%%

# Make a test signal
np.random.seed(0)
delta = .01
fs = 1/ delta
time_vec = np.arange(0, 70, delta)

#%% Sine wave

delta = .01
fs = 1/ delta
t = np.arange(0,256,delta)
wndw_factor=500
overlap_factor=2
nfft = len(t)
nperseg=len(t)/wndw_factor
noverlap=nperseg/overlap_factor

# filename = "sine_wave_input-del_.01-nfft_256k"
# plt.plot(t,np.sin(t))
# plt.title("Test sine wave, ∆=0.01, N=256000")
# plt.savefig(fname=fig_path + filename + ".png", dpi=500)
# plt.show()

#%% Calculate PSD of test signal with Welch's Method
freqs_psd, psd = signal.welch(np.sin(t),fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap, scaling="density")
# freqs_psd, psd = signal.welch(np.sin(t),fs=fs, nfft=nfft, nperseg=nperseg, scaling="density")
# freqs_psd, psd = signal.welch(np.sin(t),fs=fs, nfft=nfft, scaling="density")
# freqs_psd, psd = signal.welch(np.sin(t),fs=fs, scaling="density")
# freqs_psd, psd = signal.welch(np.sin(t), scaling="density")

freqs_ps, ps = signal.welch(np.sin(t),fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
# freqs_ps, ps = signal.welch(np.sin(t),fs=fs, nfft=nfft, nperseg=nperseg, scaling="spectrum")
# freqs_ps, ps = signal.welch(np.sin(t),fs=fs, nfft=nfft, scaling="spectrum")
# freqs_ps, ps = signal.welch(np.sin(t),fs=fs,scaling="spectrum")
# freqs_ps, ps = signal.welch(np.sin(t),scaling="spectrum")

diff = ps_psd_difference(ps,psd)

# Note: freqs are the same for both psd and ps

# filename = "sine_wave-default_fs_and_nfft"
# filename = "sine_wave-default_fs_and_nfft_nperseg"
# filename = "sine_wave-set_fs_and_default_nfft_nperseg"
filename = "sine_wave-using_fs_and_nfft_nperseg->" + str(wndw_factor) + "->" + str(100/overlap_factor) + "perc_overlap"

# No Scaling
plt.plot(freqs_psd, psd, label="PSD")
plt.plot(freqs_ps, ps, "--", label="PS")
plt.title("Parameter Testing for Welch's Method")
plt.legend()
plt.savefig(fname=fig_path + filename + "-no_scaling.png", dpi=500)
plt.show()

# Log x
plt.semilogx(freqs_psd, psd, label="PSD")
plt.semilogx(freqs_ps, ps, "--", label="PS")
plt.title("Parameter Testing for Welch's Method")
plt.legend()
plt.savefig(fname=fig_path + filename + "-logx.png", dpi=500)
plt.show()

# Log Y
plt.semilogy(freqs_psd, psd, label="PSD")
plt.semilogy(freqs_ps, ps, "--", label="PS")
plt.title("Parameter Testing for Welch's Method")
plt.legend()
plt.savefig(fname=fig_path + filename + "-logy.png", dpi=500)
plt.show()


#%% Scaling differences

# PSD / PS scaling ratios
# plt.figure(figsize=(4, 6))
plt.semilogy(freqs_psd, diff, label="PSD/PS ratio")
ax = plt.gca()
# plt.annotate("Variation due to rounding error", xy=(2, 1), xytext=(3, 1.5))
plt.title("Differencing PSD and PS")
plt.legend()
plt.savefig(fname=fig_path + "diff-" + filename + "-logy.png", dpi=500)
plt.show()

#%% Calculate PSD of test signal with just a periodogram

# freqs_ps, ps = signal.periodogram(np.sin(t),fs)
# freqs_psd, psd = signal.periodogram(np.sin(t),fs,scaling="density")

# # No Scaling
# plt.plot(freqs_psd, psd, label="PSD")
# plt.plot(freqs_ps, ps, "--", label="PS")
# plt.legend()
# plt.show()
# # Log x
# plt.semilogx(freqs_psd, psd, label="PSD")
# plt.semilogx(freqs_ps, ps, "--", label="PS")
# plt.legend()
# plt.show()
# # Log Y
# plt.semilogy(freqs_psd, psd, label="PSD")
# plt.semilogy(freqs_ps, ps, "--", label="PS")
# plt.legend()
# plt.show()



#%% Forward FFT

# # Forward transform
# t = np.arange(256)
# sp = np.fft.fft(np.sin(t))
# freq = np.fft.fftfreq(t.shape[-1])
# # plt.plot(freq, sp.real, freq, sp.imag)
# plt.plot(freq, sp.real)
# plt.show()

# # Forward transform, ortho normalization
# t = np.arange(256)
# sp = np.fft.fft(np.sin(t),norm='ortho')
# freq = np.fft.fftfreq(t.shape[-1])
# # plt.plot(freq, sp.real, freq, sp.imag)
# plt.plot(freq, sp.real)
# plt.show()

# Forward transform finer spacing (larger NFFT)
t = np.arange(0,256,0.1)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1],0.1)
# plt.plot(freq, sp.real, freq, sp.imag)
# plt.plot(freq, sp.real)
# plt.show()

# # Forward transform, finer spacing, ortho normalization
# t = np.arange(0,256,0.1)
# sp = np.fft.fft(np.sin(t),norm='ortho')
# freq = np.fft.fftfreq(t.shape[-1],0.1)
# # plt.plot(freq, sp.real, freq, sp.imag)
# plt.plot(freq, sp.real)
# plt.show()


#
# Inverse FFT
#


# # Inverse transform
# t = np.arange(256)
# sig = np.fft.ifft(sp)
# plt.plot(t, sig)
# # plt.show()

# # Inverse transform, ortho normalization
# t = np.arange(256)
# sig = np.fft.ifft(sp, norm='ortho')
# plt.plot(t, sig)
# plt.show()

# Inverse transform, finer spacing (larger NFFT)
t = np.arange(0,256,0.1)
sig = np.fft.ifft(sp)
plt.plot(t, sig)
plt.show()

# Inverse transform, finer spacing, ortho normalization
# t = np.arange(0,256,0.1)
# sig = np.fft.ifft(sp, norm='ortho')
# plt.plot(t, sig)
# plt.show()
