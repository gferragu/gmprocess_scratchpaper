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


NHNM_coeffs = pd.read_csv('./noise_models/NHNM-coeffs.txt')
NHNM_SI = pd.read_csv('./noise_models/NHNM.csv')

A = np.array(NHNM_coeffs["A"])
B = np.array(NHNM_coeffs["B"])

NHNM_period = np.array(NHNM_coeffs["P"])
NHNM_acc = A + B * (np.log10(NHNM_period))

interp = interp1d(NHNM_period, NHNM_acc, kind="linear", fill_value='extrapolate')

# npts = 1800
npts = 3600

dt = 0.01
stop = (npts - 1) / dt

def generate_noise():

    x = np.random.normal(0, 1, npts)
    t = np.linspace(0, stop, npts)

    fft_idx = np.arange(int(npts / 2), npts)

    # Built in way to shift #
    sig_fft = (np.fft.fft(x) * dt)
    freq = (np.fft.fftfreq(len(sig_fft), dt))

    sig_fft = np.fft.fftshift(sig_fft)
    freq = np.fft.fftshift(freq)

    # Manually doing stuff #
    ind_zero = np.where(freq == 0.0)
    freq_temp = freq.copy()
    freq_temp[ind_zero] = 100

    freq_abs = np.abs(freq_temp)
    period_freq_abs = (1 / freq_abs)
    period_freq_abs[ind_zero] = 0

    # Checking if this works
    # period_freq_abs_fft_idx = (1 / freq_abs[fft_idx])

    # T = 1 / freq[fft_idx]
    # period, acc = NHNM()

    NHNM_acc_i_dB = interp(period_freq_abs)
    NHNM_acc_i = 10**(NHNM_acc_i_dB / 10)

    # This works
    msa = np.mean(np.abs(sig_fft)**2)

    sig_fft_norm = sig_fft / msa

    mod_fft = sig_fft_norm * NHNM_acc_i

    # random_phases = np.random.uniform(np.pi, (2 * np.pi), npts)
    # ifft_amp = interp(period_freq_abs)

    # ifft_real = ifft_amp * np.cos(random_phases)
    # ifft_imag = ifft_amp * np.sin(random_phases)
    # ifft_complex = ifft_real + ifft_imag * 1j
    # sim_ifft = np.fft.ifft(ifft_complex)

    sim_ifft = np.fft.ifft(mod_fft)

    sim = np.real(sim_ifft)

    sim_fft = np.fft.fft(sim)

    sim_fft_abs = np.abs(sim_fft)

    sim_fft_phase = np.angle(sim_fft)

    return [freq[fft_idx], NHNM_acc_i[fft_idx], sim_fft_abs[fft_idx], sim_fft_phase[fft_idx]]


nsim = 100

# Why do we have a 2D array again?
matrix = np.zeros((nsim, 1800))

for i in range(nsim):
    [freq, NHNM, sim_abs, sim_phase] = generate_noise()

    matrix[i, :] = sim_abs


amp_mean = np.mean(matrix, axis=0)
amp_std = np.std(matrix, axis=0)
amp_plus_1std = (amp_mean + amp_std)
amp_min_1std = (amp_mean - amp_std)

amp_diff = (amp_mean - NHNM)

#%%
##############################################################
# Plotting
##############################################################
# plt.plot(t, x)
# plt.title("Plotting t and x")
# plt.show()

plt.loglog(freq, amp_mean, label='amp_mean')
plt.loglog(freq, amp_std, label='amp_std')
plt.loglog(freq, NHNM, label='NHNM')
plt.title("Plotting amp_mean and NHNM")
plt.legend()
plt.show()

plt.loglog(freq, amp_std, label='+ 1 Std. Dev.')
plt.loglog(freq, amp_std, label='- 1 Std. Dev.')
# plt.loglog(freq, amp_std, label='amp_std')
# plt.plot(freq, amp_std, label='+ 1 Std. Dev.')
# plt.plot(freq, amp_std, label='- 1 Std. Dev.')
# # plt.plot(freq, amp_std, label='amp_std')
plt.title("Plotting Std. Dev. of Amplitude (amp_std)")
plt.legend()
plt.xlim(0.1, 0.11)
plt.show()

plt.plot(freq, sim_phase, 'ko')
# plt.hist(sim_phase)
plt.show()


#############
# Old Plots #
#############
# plt.plot(T)
# plt.title("Plotting T")
# plt.show()

# plt.plot(T)
# plt.semilogy("Plotting LogY T")
# plt.show()

# plt.semilogx((1/period), acc)
# plt.semilogx(NHNM_period, NHNM_acc)
# plt.plot(T, NHNM_acc_i)
# plt.title("Plotting T and NHNM_acc_i")
# plt.show()

# plt.semilogx(NHNM_period, NHNM_acc)
# plt.plot(T, NHNM_acc_i_dB)
# plt.title("Plotting T and NHNM_acc_i_dB")
# plt.show()
##############################################################

# # This works
# random_phases = np.random.uniform(np.pi, (2 * np.pi), npts)
# ifft_amp = interp(period_freq_abs)

# # Checking if this does
# # random_phases = np.random.uniform(np.pi, (2 * np.pi), len(period_freq_abs_fft_idx))
# # ifft_amp = interp(period_freq_abs_fft_idx)

# # img = np.sqrt(-1 + 0j)

# ifft_real = ifft_amp * np.cos(random_phases)
# ifft_imag = ifft_amp * np.sin(random_phases)
# ifft_complex = ifft_real + ifft_imag * 1j

# sim_ifft = np.fft.ifft(ifft_complex)

##############################################################
# Plotting
##############################################################
# sim_real = np.real(sim_ifft)
# plt.plot(sim_real)
# plt.title("Plotting sim_real")
# plt.show()

# sim_imag = np.imag(sim_ifft)
# plt.plot(sim_imag)
# plt.title("Plotting sim_imag")
# plt.show()
##############################################################

# Z = rand_phase_PSD_signal(T, psd, phase, interp_mode)

# signal = np.fft.irfft(Z)

# t = get_time_array(T, psd, signal, npts, delta)

