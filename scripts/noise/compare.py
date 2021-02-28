# -*- coding: utf-8 -*-

import os
import sys
import pkg_resources
import glob
import numpy as np
import matplotlib.pyplot as plt

from obspy import read
from obspy.core.stream import Stream
from obspy.signal.filter import bandpass

from gmprocess import filtering
from gmprocess.io.read import read_data
from gmprocess.streamcollection import StreamCollection
from gmprocess.config import get_config
from gmprocess.processing import process_streams
from gmprocess.event import get_event_object

from gmprocess.denoise import dwt


plt.style.use('seaborn')

#%%

path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/synthetics/data/miniseed/noisy/new_noise_models/"
save_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/synthetics/data/miniseed/processed/wavelets/"

fig_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/denoising/wavelets/"
bandpass_dat_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/synthetics/data/miniseed/processed/bandpass/"
bandpass_fig_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/denoising/bandpass/"

# model = "NHNM"
model = "NLNM"
# model = "stochastic"'


#%% gmprocess bandpassing

# sc = StreamCollection.from_directory(path + model + "/")
# sc.describe()

dir_list = glob.glob(path + model + "/*")
# for i in range(len(dir_list)):
#     dir_list[i] = dir_list[i].split("/")[-1]

freqmin = 0.08
# freqmax = 20  # Nyquist is 19.535179640718564
freqmax = 19

for i, file in enumerate(dir_list):
    print(i)
    st = read(dir_list[i])
    st.plot()

    synth = dir_list[i].split("/")[-1].split("-")[1]
    npts = st[0].stats.npts
    samp_rate = (1 / st[0].stats.delta)

    bandpassed = st.copy()
    for j,tr in enumerate(bandpassed):
        data = tr.data
        data_filtered = bandpass(data, freqmin, freqmax, samp_rate, corners=4, zerophase=True)
        tr.data = data_filtered

    filename = "signal-bandpass_denoised-" + str(freqmin) + "_" + str(freqmax) + "-""synth_" + synth + "-" + model + "_" + str(i)
    bandpassed.plot(outfile = bandpass_fig_path + "signal/" + model + "/" + filename + ".png")




    # filename = "signal-BP_denoised-synth_" + synth + "-" + model + "_" + str(i)
    # signal.write(save_path + "signal/" + model + "/" + filename + ".ms", format="MSEED")
    # signal.plot(outfile=fig_path + "signal/" + model + "/" + filename + ".png")

    # filename = "noise-BP_denoised-synth_" + synth + "-" + model + "_" + str(i)
    # noise.write(save_path + "noise/" + model + "/" + filename + ".ms", format="MSEED")
    # noise.plot(outfile=fig_path + "noise/" + model + "/" + filename + ".png")


#%% Wavelets, Processing & Writing

# dir_list = glob.glob(path + model + "/*")
# for i in range(len(dir_list)):
#     dir_list[i] = dir_list[i].split("/")[-1]


# for i, file in enumerate(dir_list):
#     st = read(dir_list[i])

#     synth = dir_list[i].split("/")[-1].split("-")[1]

#     denoised = dwt.denoise(st, store_noise=True)
#     signal = denoised["data"]
#     noise = denoised["noise"]

#     filename = "signal-wavelet_denoised-synth_" + synth + "-" + model + "_" + str(i)
#     signal.write(save_path + "signal/" + model + "/" + filename + ".ms", format="MSEED")
#     signal.plot(outfile=fig_path + "signal/" + model + "/" + filename + ".png")

#     filename = "noise-wavelet_denoised-synth_" + synth + "-" + model + "_" + str(i)
#     noise.write(save_path + "noise/" + model + "/" + filename + ".ms", format="MSEED")
#     noise.plot(outfile=fig_path + "noise/" + model + "/" + filename + ".png")




# %% Time series comparison plots

# clean = glob.glob(path + model + "/*")




# %% Double check directory contents with plots

dir_list = glob.glob(path + model + "/*")
# for i in range(len(dir_list)):
#     dir_list[i] = dir_list[i].split("/")[-1]


for i, file in enumerate(dir_list):
    st = read(dir_list[i])
    st.plot()


#%% Comparison Plots (copy pasted from models.py)

# This is probably a more appropriate place to do this

#%% Read noise in as ObsPy trace

san_check = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/figs/sanity_checks/noise_generation/"

from noise_generation import read_noise
# Comparing low and high noise models

# Read in noise using ObsPy
[NHNM_st, NLNM_st] = read_noise()


def plot_waveform_overlay(NHNM_st, NLNM_st, reverse_zorder=False):
    print("\nPlotting Waveform Overlay ...\n")

    plt.figure()
    plt.title("Noise Series Constructed From NHNM and NLNM:"
              "Anomalous Amplitudes", fontsize=13, fontweight="bold")

    if reverse_zorder:
        plt.plot(NLNM_st[0].times(), NLNM_st[0], label="NLNM")
        plt.plot(NHNM_st[0].times(), NHNM_st[0], label="NHNM")
    else:
        plt.plot(NHNM_st[0].times(), NHNM_st[0], label="NHNM")
        plt.plot(NLNM_st[0].times(), NLNM_st[0], label="NLNM")

    plt.xticks(np.arange(0, max(NHNM_st[0].times()), 5))
    plt.xlabel("Time (s)", fontweight="bold")
    plt.ylabel("Velocity (m/s)", fontweight="bold")
    plt.legend()
    plt.show()


plot_waveform_overlay(NHNM_st, NLNM_st, reverse_zorder=True)


def plot_fft_overlay(NHNM_st, NLNM_st, reverse_zorder=False):
    print("\nPlotting Periodogram Overlay ...\n")
    delta_h = NHNM_st[0].stats.delta
    delta_l = NLNM_st[0].stats.delta

    nfft_h = len(NHNM_st[0]) * 2
    nfft_l = len(NLNM_st[0]) * 2

    # fft_h = np.fft.fftshift(fft_h)
    fft_h = np.abs(np.fft.fftshift(np.fft.fft(NHNM_st[0], nfft_h)))
    freq_h = np.fft.fftfreq(nfft_h, delta_h)
    freq_h = np.fft.fftshift(freq_h)

    # fft_l = np.fft.fftshift(fft_l)
    fft_l = np.abs(np.fft.fftshift(np.fft.fft(NLNM_st[0], nfft_l)))
    freq_l = np.fft.fftfreq(nfft_l, delta_l)
    freq_l = np.fft.fftshift(freq_l)

    if reverse_zorder:
        plt.plot(freq_l, fft_l, label="NLNM")
        plt.plot(freq_h, fft_h, label="NHNM")
    else:
        plt.plot(freq_h, fft_h, label="NHNM")
        plt.plot(freq_l, fft_l, label="NLNM")

    plt.title("FFT of Noise Time Series Generated from NHNM/NLNM", fontsize=16, fontweight="bold")
    plt.ylabel('Spectral Amplitude', fontweight="bold")
    plt.xlabel('Frequency (Hz)', fontweight="bold")
    plt.xlim(0, 20)
    plt.legend()

    # plt.savefig(san_check + "NHNM-NLNM FFTs.png")
    plt.show()


plot_fft_overlay(NHNM_st, NLNM_st, reverse_zorder=True)


def plot_spectrograms(NHNM_st, NLNM_st):
    print("\nPlotting Spectrograms ...\n")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle("Spectrograms for NHNM and NLNM Time Series",
                 fontsize=16, fontweight="bold")

    NHNM_st[0].spectrogram(axes=axs[0])
    # NHNM_st[0].spectrogram(axes=axs[0], dbscale=True)
    # NHNM_st[0].spectrogram(axes=axs[0], log=True)
    axs[0].set_title("Time Series Constructed from NHNM", fontweight="bold")
    axs[0].set_xlabel("Time (s)", fontweight="bold")
    axs[0].set_ylabel("Frequency (Hz)", fontweight="bold")
    axs[0].set_xticks(np.arange(0, max(NHNM_st[0].times()), 5))

    NLNM_st[0].spectrogram(axes=axs[1])
    # NLNM_st[0].spectrogram(axes=axs[1], dbscale=True)
    # NLNM_st[0].spectrogram(axes=axs[1], log=True)
    axs[1].set_title("Time Series Constructed from NLNM", fontweight="bold")
    axs[1].set_xlabel("Time (s)", fontweight="bold")
    axs[1].set_ylabel("Frequency (Hz)", fontweight="bold")
    axs[1].set_xticks(np.arange(0, max(NHNM_st[0].times()), 5))

    # plt.savefig(san_check + "NHNM-NLNM Spectrograms.png")
    plt.show()


plot_spectrograms(NHNM_st, NLNM_st)


def plot_ppsd_welch(NHNM_st, NLNM_st):

    plt.figure()
    # segmt_lens = [32, 64, 128, 256, 512]
    segmt_lens = [32, 256]
    segmt_lens.reverse()

    for nperseg in segmt_lens:
        fs_h = NLNM_st[0].stats.sampling_rate
        fs_l = NLNM_st[0].stats.sampling_rate

        freq_wh, Pxx_wh = sig.welch(NHNM_st[0], fs_h, nperseg=nperseg)
        freq_wl, Pxx_wl = sig.welch(NLNM_st[0], fs_l, nperseg=nperseg)
        label_h = "NHNM, nperseg: " + str(nperseg)
        label_l = "NLNM, nperseg: " + str(nperseg)
        plt.semilogy(freq_wh, Pxx_wh, label=label_h)
        plt.semilogy(freq_wl, Pxx_wl, label=label_l)

    # plt.ylim([0.5e-3, 1])
    plt.title("Estimated PSD for NHNM/NLNM Time Sereis with Welch's Method",
              fontsize=13, fontweight="bold")
    plt.xlabel('frequency [Hz]', fontweight="bold")
    plt.ylabel('PSD [V**2/Hz]', fontweight="bold")
    plt.legend()
    plt.savefig(san_check + "PSD via Welch's Method - NHNM-NLNM - 32 and 256 nperseg.png")


plot_ppsd_welch(NHNM_st, NLNM_st)



#%% Try the Boore method with stochastic series multiplied by noise model FT