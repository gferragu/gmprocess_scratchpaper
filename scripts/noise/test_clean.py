#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:15:55 2020

@author: gabriel
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

# from models import NHNM
# from models import get_uniform_rand_phase
# from models import rand_phase_PSD_signal
# from models import get_time_array

plt.style.use('seaborn')

NHNM_coeffs = pd.read_csv('./noise_models/NHNM-coeffs.txt')
NHNM_SI = pd.read_csv('./noise_models/NHNM.csv')

A = np.array(NHNM_coeffs["A"])
B = np.array(NHNM_coeffs["B"])

NHNM_period = np.array(NHNM_coeffs["P"])
NHNM_acc = A + B * (np.log10(NHNM_period))

interp = interp1d(NHNM_period, NHNM_acc, kind="linear", fill_value='extrapolate')

# FT_npts = 10
FT_npts = 1798 * 2
sig_npts = int(FT_npts / 2)

npts = int(1798)

# Altering ... need spacing and time range right if I save noise in this script
# dt = 0.01
dt = 0.02559485
# stop = (FT_npts - 1) / dt
stop = (FT_npts - 1) * dt
# stop = (sig_npts - 1) * dt


def generate_noise(npts=1798):
    FT_npts = npts
    # Create a random time series (can mod std. dev. of this)
    x = np.random.normal(0, 1, FT_npts)
    t = np.linspace(0, stop, FT_npts)
    # x = np.random.normal(0, 1, sig_npts) #Should have same # npts as synthetics?
    # t = np.linspace(0, stop, sig_npts)

    # Construct array of indeces we need
    fft_idx = np.arange(int(FT_npts / 2), FT_npts)

    # Take the FFT and shift zero freq term
    sig_fft = (np.fft.fft(x) * dt)
    freq = (np.fft.fftfreq(len(sig_fft), dt))

    sig_fft = np.fft.fftshift(sig_fft)
    freq = np.fft.fftshift(freq)

    # Manually doing indeces stuff
    ind_zero = np.where(freq == 0.0)
    freq_temp = freq.copy()
    freq_temp[ind_zero] = 100

    # Take only positive freq terms and convert to period
    freq_abs = np.abs(freq_temp)
    period_freq_abs = (1 / freq_abs)
    period_freq_abs[ind_zero] = 0

    # Get the noise model and convert from dB to (presumably) (m/s^2) / Hz
    NHNM_acc_i_dB = interp(period_freq_abs)
    NHNM_acc_i = 10**(NHNM_acc_i_dB / 10)

    # Get the mean square average
    msa = np.sqrt(np.mean(np.abs(sig_fft)**2))

    # Normalize the FFT of the signal by this
    sig_fft_norm = sig_fft / msa

    # Multiply noise model and normalized FT
    mod_fft = sig_fft_norm * NHNM_acc_i

    # Transfer back to the time domain
    sim_ifft = np.fft.ifft(mod_fft)

    # Take the real component only
    sim = np.real(sim_ifft)

    # Check the FT and phase of the signal
    sim_fft = np.fft.fft(sim)
    sim_fft_abs = np.abs(sim_fft)
    sim_fft_phase = np.angle(sim_fft)

    # Returns + freq portions of
    return [t, sim, freq[fft_idx], NHNM_acc_i[fft_idx], sim_fft_abs[fft_idx], sim_fft_phase[fft_idx]]
    # return [t[fft_idx], sim[fft_idx], freq[fft_idx], NHNM_acc_i[fft_idx], sim_fft_abs[fft_idx], sim_fft_phase[fft_idx]]
    # return [t, sim, freq, NHNM_acc_i, sim_fft_abs, sim_fft_phase]



nsim = 100

npts = 1798
npts2 = npts * 2
# npts = 10000

# Try this a whole bunch of times and build a set to get stats from
matrix1 = np.zeros((nsim, int(npts / 2)))
matrix2 = np.zeros((nsim, int(npts2 / 2)))


freq1 = []
freq2 = []

for i in range(nsim):
    [t, x, freq1, NHNM, sim_abs, sim_phase] = generate_noise(npts)
    matrix1[i, :] = sim_abs

    [t2, x2, freq2, NHNM2, sim_abs2, sim_phase2] = generate_noise(npts2)
    matrix2[i, :] = sim_abs2


# Get meaningful measures of the time series: Mean, Std. Dev., Residuals
# amp_mean = np.mean(matrix, axis=0)
# amp_std = np.std(matrix, axis=0)
# amp_plus_1std = (amp_mean + amp_std)
# amp_min_1std = (amp_mean - amp_std)

# amp_diff = (amp_mean - NHNM)
# amp_scale = (NHNM / amp_mean)
# avg_scale = np.mean(amp_scale)

amp_mean = np.mean(matrix1, axis=0)
amp_std = np.std(matrix1, axis=0)
amp_plus_1std = (amp_mean + amp_std)
amp_min_1std = (amp_mean - amp_std)

amp_diff = (amp_mean - NHNM)
amp_scale = (NHNM / amp_mean)
avg_scale = np.mean(amp_scale)

amp_mean2 = np.mean(matrix2, axis=0)
amp_std2 = np.std(matrix2, axis=0)
amp_plus_1std2 = (amp_mean2 + amp_std2)
amp_min_1std2 = (amp_mean2 - amp_std2)

amp_diff2 = (amp_mean2 - NHNM2)
amp_scale2 = (NHNM2 / amp_mean2)
avg_scale2 = np.mean(amp_scale2)

#%%###########################################################
# Plotting
##############################################################

san_check = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/figs/sanity_checks/noise_generation/"

plt.plot(t, x, label=str(npts) + ' FFT Points')
plt.plot(t2, x2, label=str(npts2) + ' FFT Points')
plt.title("Plotting t and x")
plt.savefig(san_check + "Boore Method - Input White Noise.png")
plt.show()


# plt.plot(freq, sim_phase, 'ko')
# # plt.hist(sim_phase)
# # plt.show()
# plt.savefig(san_check + "Boore Method - Phase Check.png")
# plt.show()


# plt.loglog(freq, amp_mean, label='amp_mean')
# plt.loglog(freq, NHNM, label='NHNM')
# plt.title("Plotting amp_mean and NHNM")
# plt.legend()
# # plt.show()
# # plt.savefig(san_check + "Boore Method - Model vs. Modified White Noise.png")
# plt.show()

plt.loglog(freq1, amp_mean, label='amp_mean, npts: ' + str(npts))
plt.loglog(freq2, amp_mean2, '--', label='amp_mean2, npts: ' + str(npts2))
plt.loglog(freq2, NHNM2, label='NHNM')
plt.title("Plotting amp_mean and NHNM")
plt.legend()
# plt.show()
# plt.savefig(san_check + "Boore Method - Model vs. Modified White Noise.png")
plt.show()


# plt.loglog(freq, amp_mean, label='amp_mean')
# plt.loglog(freq, amp_std, label='amp_std')
# plt.loglog(freq, NHNM, label='NHNM')
# plt.title("Plotting amp_mean, amp_std, and NHNM ")
# plt.legend()
# # plt.show()
# plt.savefig(san_check + "Boore Method - Model vs. Modified White Noise - Check Std Dev.png")
# plt.show()


# plt.loglog(freq, amp_plus_1std, label='+ 1 Std. Dev.')
# plt.loglog(freq, amp_std, label='amp_std')
# plt.loglog(freq, amp_min_1std, label='- 1 Std. Dev.')
# plt.title("Plotting Std. Dev. of Amplitude (amp_std)")
# plt.legend()
# # plt.xlim(0.1, 0.11)
# # plt.show()
# plt.savefig(san_check + "Boore Method - Modified White Noise and Std Dev.png")
# plt.show()

# plt.plot(freq, amp_diff, label='residual')
# plt.title("Residual Between Mean Amp. of Modified Signal and True NHNM")
# plt.xlabel("Frequency (Hz)")
# plt.legend()
# # plt.show()
# plt.savefig(san_check + "Boore Method - Residuals from Modified White Noise Mean Amplitude and True NHNM.png")
# plt.show()

# plt.plot(freq, amp_scale, label='NHNM / Modified-Noise Ratio')
# plt.title("Ratio of Amplitude Scales: NHNM to Noise")
# plt.hlines(avg_scale, min(freq), max(freq), colors='r',
#            label='Average Value: ' + str(avg_scale), zorder=10)
# plt.legend()
# # plt.show()
# plt.savefig(san_check + "Boore Method - Amplitude Ratio of True NHNM to Modified White Noise Mean Amplitude.png")
# plt.show()

##############################################################
