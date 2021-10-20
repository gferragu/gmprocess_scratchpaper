# -*- coding: utf-8 -*-

# %% Imports
import sys
import os
sys.path.append(os.path.abspath("/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/noise/"))
sys.path.append(os.path.abspath("/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/synthetics/"))

import glob
import numpy as np
import pandas as pd
import matplotlib as mplq
import matplotlib.pyplot as plt
from scipy import signal as sig
from obspy import read
import obspy.signal.filter
from obspy import Trace
from obspy.core.stream import Stream

# from synthetics.synthetics import read_synthetic_streams
# from synthetics import read_synthetic_streams                 # Used to work?
import synthetics as synth

# %% Functions


def gen_fft_noise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:(Np + 1)] *= phases
    f[-1:(-1 - Np):-1] = np.conj(f[1:(Np + 1)])
    return np.fft.ifft(f).real


def gen_band_limited_noise(min_freq, max_freq, samples=1024, sampling_rate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/sampling_rate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    noise = gen_fft_noise(f)
    if not any(noise != 0.0):
        print("Calculated noise is zero, exiting")
    else:
        return noise
    # return gen_fft_noise(f)

# def gen_boore_noise(model='NHNM', npts=1798):



def add_band_limited_noise(signal,noise):
    if len(signal) == len(noise):
        for i in range(len(signal)):
            sig.data[i] = signal.data[i] + noise[i]
    else:
        print("Signal and noise time series must of the same length")
        return

    return signal


def add_random_noise(signal, sigma=0.1):

    noise = np.zeros(len(signal))
    noisy_signal = np.zeros(len(signal))

    for i in range(len(signal)):
        noise[i] = np.random.normal(noise[i], sigma, 1)
        noisy_signal[i] = signal[i] + noise[i]

    return noisy_signal


def add_bandpassed_random_noise(signal, sigma=0.5,
                                min_freq=0.001, max_freq=40.0,
                                samples=1024, sampling_rate=1):
    noise = add_random_noise(signal, sigma)
    bandpassed_noise = obspy.signal.filter.bandpass(noise, min_freq,
                                                    max_freq, sampling_rate,
                                                    corners=5, zerophase=True)
    noisy_signal = signal + bandpassed_noise

    return noisy_signal


def generate_starting_signal(A=1, f=5, t_min=0, t_max=60, samp_interv=0.001):
    # sampling_rate = 1 / samp_interv
    nsamp = int((t_max - t_min) / samp_interv)
    t = np.linspace(t_min, t_max, nsamp)

    signal = A * np.sin((2 * np.pi) * t)

    return t, signal


def read_noise(quantity="vel", y_units="dB", filename="", path ="/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/noise/data/csv/"):

    NHNM_dir_list = glob.glob(path + "NHNM/*" + quantity + "*")
    NLNM_dir_list = glob.glob(path + "NLNM/*" + quantity + "*")

    for i in range(len(NHNM_dir_list)):
        NHNM_dir_list[i] = NHNM_dir_list[i].split("/")[-1]
    for i in range(len(NLNM_dir_list)):
        NLNM_dir_list[i] = NLNM_dir_list[i].split("/")[-1]

    dir_lists = [NHNM_dir_list, NLNM_dir_list]

    NHNM = Stream()
    NLNM = Stream()

    for dir_list in dir_lists:
        for idx, obj in enumerate(dir_list):

            if "high" in obj:
                noise = pd.read_csv(path + "NHNM/" + str(obj))

            elif "low" in obj:
                noise = pd.read_csv(path + "NLNM/" + str(obj))
            else:
                print("\nIssue identifying correct model. Exiting ...")
                break

            t = np.array(noise["time"])
            amp = np.array(noise["amplitude"])

            npts = len(t)
            delta = (max(t) - min(t)) / len(t)

            tr = Trace(amp)
            tr.stats.npts = npts
            tr.stats.delta = delta

            if "high" in obj:
                NHNM += tr
            elif "low" in obj:
                NLNM += tr

    return [NHNM, NLNM]

def write_noise(stream, filename, path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/noise/data/"):
    stream.write(path + filename + ".ms", format="MSEED")
    return

def plot_stream(stream):
    for tr in stream:
        return tr.plot()

def add_modeled_noise(signal, noise):
    noisy_signal = Stream()

    # Add in a check for sampling rate and resampling routine?
    # Oh wait I did that already, but need to do it for Lancsoz

    if type(signal) == obspy.core.stream.Stream:
        noisy_signal = signal.copy()
        for i in range(len(signal)):
            if len(signal[i]) == len(noise):
                for ii in range(len(signal[i])):
                    noisy_signal[i].data[ii] = (signal[i].data[ii] + noise.data[ii])
            else:
                noise_resamp = sig.resample(noise.data, len(signal))
                for i in range(len(signal)):
                    for ii in range(len(signal[i])):
                        noisy_signal[i].data[ii] = (signal[i].data[ii] + noise_resamp.data[ii])

        return noisy_signal

    elif type(signal) == obspy.core.trace.Trace:
        noisy_signal = signal.copy()
        if len(signal) == len(noise):
            for i in range(len(signal)):
                for ii in range(len(signal[i])):
                    noisy_signal[i].data[ii] = (signal[i].data[ii] + noise.data[ii])

        else:
            noise_resamp = sig.resample(noise.data, len(signal))
            for i in range(len(signal)):
                for ii in range(len(signal[i])):
                    noisy_signal[i] = (signal[i].data[ii] + noise_resamp.data[ii])

        return noisy_signal
    else:
        print("Data is neither an ObsPy stream nor trace and as such is unsupported")


def build_modeled_noise_dataset(model="", filename="", path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/synthetics/data/miniseed/noisy/new_noise_models/"):
    synths = synth.read_synthetic_streams()
    wasatch_1 = synths[0]
    wasatch_2 = synths[1]

    if model == "stochastic":
        print("Stochastic noise models under development")

    else:
        [NHNM, NLNM] = read_noise()

        for i in range(len(NHNM)):
            filename = "/NHNM/high_noise_wasatch_synth-"

            savename=(filename + "1" + "-noise_ID-" + str(i))
            noisy_high_1 = add_modeled_noise(wasatch_1, NHNM[i])
            write_noise(noisy_high_1, filename=savename, path=path)

            savename=(filename + "2" + "-noise_ID-" + str(i))
            noisy_high_2 = add_modeled_noise(wasatch_2, NHNM[i])
            write_noise(noisy_high_2, filename=savename, path=path)

        for i in range(len(NLNM)):
            filename = "/NLNM/low_noise_wasatch_synth-"

            savename=(filename + "1" + "-noise_ID-" + str(i))
            noisy_high_1 = add_modeled_noise(wasatch_1, NLNM[i])
            write_noise(noisy_high_1, filename=savename, path=path)

            savename=(filename + "2" + "-noise_ID-" + str(i))
            noisy_high_2 = add_modeled_noise(wasatch_2, NLNM[i])
            write_noise(noisy_high_2, filename=savename, path=path)


def plot_modeled_noise_dataset(path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/synthetics/data/miniseed/noisy/new_noise_models/"):
    save_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/synthetics/new_noise_models/"
    dir_list = glob.glob(path + "NHNM/*")
    dir_list = dir_list + glob.glob(path + "NLNM/*")
    for i in range(len(dir_list)):
        dir_list[i] = dir_list[i].split("/")[-1]

    for i in range(len(dir_list)):
        if "high" in dir_list[i]:
            filename = "NHNM/" + dir_list[i].split(".")[0]
            tmp_stream = read(path + "NHNM/" + dir_list[i])
            tmp_stream.plot(outfile=save_path + filename + ".png")
        elif "low" in dir_list[i]:
            filename = "NLNM/" + dir_list[i].split(".")[0]
            tmp_stream = read(path + "NLNM/" + dir_list[i])
            tmp_stream.plot(outfile=save_path + filename + ".png")
        else:
            break

# %% Adding noise to the data

samp_interv = 0.001
sampling_rate = 1 / samp_interv
t_min = 0
t_max = 60
nsamp = int((t_max - t_min) / samp_interv)

t = np.linspace(t_min, t_max, nsamp)
A = 1
f = 5

signal = A * np.sin((2 * np.pi) * t)
# signal_fft = np.fft(signal)

# Generate noise with FFT
# noisy_signal = gen_band_limited_noise(10, 25, nsamp, sampling_rate)
# noisy_signal = gen_band_limited_noise(1, 10, nsamp, sampling_rate)
# noisy_signal = gen_band_limited_noise(0.1, 40, nsamp, sampling_rate)
noisy_signal = gen_band_limited_noise(0.1, 5, nsamp, sampling_rate)


# Generate noise withy random Gaussian
# noisy_signal = add_random_noise(signal, sigma=1)

# noisy_signal = add_bandpassed_random_noise(signal,
#                                             min_freq=10.0, max_freq=25.0,
#                                             sampling_rate=sampling_rate)

# noisy_signal = add_bandpassed_random_noise(signal,
#                                             min_freq=1.0, max_freq=10.0,
#                                             sampling_rate=sampling_rate)

# noisy_signal = add_bandpassed_random_noise(signal,
#                                            min_freq=0.1, max_freq=49.0,
#                                            sampling_rate=sampling_rate)

# %% Plotting the results


# Plot the clean signal

# fig = plt.figure()
# plt.plot(t, signal)
# # plt.plot(t, noisy_signal)
# # plt.show()
# plt.xlim(0, 10)
# # plt.ylim(-1.2, 1.2)
# filename = "clean_sig.png"
# plt.title("Clean signal from sine()")
# plt.savefig(fig_path + filename, dpi=500)

# Plot the noisy signal
# fig = plt.figure()
# # plt.plot(t, signal)
# plt.plot(t, noisy_signal)
# # plt.show()
# plt.xlim(0, 10)
# # plt.ylim(-1.2, 1.2)
# filename = "clean_sig_random_noise_added-sigma1.png"
# plt.title("Added random Gaussian noise, sigma = 1")
# plt.savefig(fig_path + filename, dpi=500)

# # Plot the noisy bandpassed signal
# fig = plt.figure()
# # plt.plot(t, signal)
# plt.plot(t, noisy_signal)
# # plt.show()
# plt.xlim(0, 10)
# # plt.ylim(-1.2, 1.2)
# filename = "clean_sig_bandpassed_noise_10-25Hz_added-sigma0.5.png"
# plt.title("Added bandpassed Gaussian noise - 10-25Hz, sigma = 0.5")
# plt.savefig(fig_path + filename, dpi=500)

# Plot the noisy FFT signal
# fig = plt.figure()
# # plt.plot(t, signal)
# plt.plot(t, noisy_signal)
# # plt.show()
# plt.xlim(0, 10)
# # plt.ylim(-1.2, 1.2)
# filename = "clean_sig_FFT_noise_0.1-5Hz_added.png"
# plt.title("Added noise from FFT/iFFt - 0.1-5Hz")
# # plt.savefig(fig_path + filename, dpi=500)

# # Plot the residual (just the noise)
# fig = plt.figure()
# resid = np.zeros(len(noisy_signal))
# for i in range(len(noisy_signal)-1):
#     resid[i] = noisy_signal[i] - signal[i]

# resid2 = noisy_signal - signal

# plt.plot(t, resid)
# plt.xlim(0, 10)
# # plt.ylim(-1.2, 1.2)


# # %%
# noise_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/noise/data/"
# data_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/noise/data/miniseed/"

# quant = "vel"
# dir_list = glob.glob(noise_path + "*" + quant + "*")
# for i in range(len(dir_list)):
#     dir_list[i] = dir_list[i].split("/")[-1]

# NHNM_st = Stream()
# NLNM = Stream()

# for idx, obj in enumerate(dir_list):
#     noise = pd.read_csv(noise_path + str(obj))

#     t = np.array(noise["time"])
#     amp = np.array(noise["amplitude"])

#     npts = len(t)
#     delta = (max(t) - min(t)) / len(t)

#     tr = Trace(amp)
#     tr.stats.npts = npts
#     tr.stats.delta = delta

#     model = obj.split("_")[1]
#     if model == "NHNM":
#         NHNM_st += tr
#     elif model == "NLNM":
#         NLNM += tr

#     fname = obj.split(".")[0]
#     # tr.plot(outfile=fig_path + "NHNM_NLNM/" + fname + ".png")

# NHNM_st.write(data_path + "NHNM_" + quant + ".ms", format="MSEED")
# NLNM.write(data_path + "NLNM_" + quant + ".ms", format="MSEED")

# # %% Try random noise on the synthetic signals

# synth_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/test_data/synthetic/"
# fig_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/synthetics/"

# st1_filename="wasatch_synth1_horzXY"
# st2_filename="wasatch_synth2_horzXY"

# nsamp = 1798
# delta = 0.02559485
# sampling_rate = 1/delta

# synth1 = read(synth_path + st1_filename + ".ms")
# synth2 = read(synth_path + st2_filename + ".ms")

# noisy_synth1 = synth1
# noisy_synth2 = synth2

# # noise1 = gen_band_limited_noise(1, 3, nsamp, sampling_rate)
# # noise2 = gen_band_limited_noise(1, 3, nsamp, sampling_rate)
# noise1 = gen_band_limited_noise(0.1, 10, nsamp, sampling_rate)
# noise2 = gen_band_limited_noise(0.1, 10, nsamp, sampling_rate)

# noisy_synth1[0] = add_band_limited_noise(synth1[0], noise1)
# noisy_synth1[1] = add_band_limited_noise(synth1[1], noise1)

# noisy_synth2[0] = add_band_limited_noise(synth2[0], noise2)
# noisy_synth2[1] = add_band_limited_noise(synth2[1], noise2)


# synth1.plot()
# synth2.plot()
# plt.show()

# plt.plot(noise1)
# plt.plot(noise2)
# plt.show()

# # noisy_synth1.plot(outfile=fig_path + st1_filename + '1-3Hz_noise.png')
# # noisy_synth2.plot(outfile=fig_path + st2_filename + '1-3Hz_noise.png')
# noisy_synth1.plot(outfile=fig_path + st1_filename + '0.1-10Hz_noise.png')
# noisy_synth2.plot(outfile=fig_path + st2_filename + '0.1-10Hz_noise.png')

# #%% Resample noise and add to clean synthetic traces

# # Copy to prevent accessing same memory
# tr = synth1[0].copy()
# noisy_tr = synth1[0].copy()

# tr.stats.station = "CLEAN"
# noisy_tr.stats.station = "NOISY"

# npts = synth1[0].stats.npts

# NHNM_resamp = sig.resample(NHNM_st[0], npts)
# NHNM_resamp_scaled = np.zeros(len(NHNM_resamp))

# scale = 100

# for i in range(npts):
#     noise = scale * NHNM_resamp[i]
#     NHNM_resamp_scaled[i] = noise
#     noisy_tr.data[i] = tr.data[i] + noise


# # comparison = Stream()
# # comparison += tr
# # comparison += noisy_tr

# # comparison.plot()
# # plt.close()

# # tr.plot()
# # plt.close()
# # noisy_tr.plot()
# # plt.close()

# # plt.plot(tr.data, noisy_tr.data)
# # # plt.xlim(0, 0.25)
# # plt.show()


# plt.plot(tr)
# plt.plot(noisy_tr, "--")
# plt.xlim(750,1500)
# plt.ylim(-0.1,0.1)
# plt.show()

# plt.plot(NHNM_resamp_scaled)
# # plt.show()

# plt.plot(NHNM_resamp)
# plt.show()

#%% Convert noise csv to mseed
model = ""
quantity = "vel"
units = "dB"

noise = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/noise/data/miniseed/"
save_noisy = "/Users/gabriel/Documents/Research/USGS_Work//gmprocess/scripts/synthetics/data/miniseed/noisy/"

[NHNM, NLNM] = read_noise()

##Plot?
# plot_stream(NHNM)
# plot.stream(NLNM)

# write_noise(NHNM)
# write_noise(NLNM)

# for i, tr in enumerate(NHNM):
#     model = "NHNM"
#     filename = "noise-" + model + "_" + quantity + "_" + units + "_" + str(i)

#     write_noise(tr, filename, path=noise)

# for i, tr in enumerate(NLNM):
#     model = "NLNM"
#     filename = "noise-" + model + "_" + quantity + "_" + units + "_" + str(i)

#     write_noise(tr, filename, path=noise)

#%% Re-read to check
# noise = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/noise/data/miniseed/"
# noise = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/noise/data/miniseed/noisy/"
# save = "/Users/gabriel/Documents/Research/USGS_Work//gmprocess_scratchpaper/scripts/synthetics/data/miniseed/"

# read_test = read(noise + "NHNM-rand*.ms")


#%% Add noise and save miniseed files

# model = ""

# synths = synth.read_synthetic_streams()

# wasatch_1 = synths[0]
# wasatch_2 = synths[1]

# [NHNM, NLNM] = read_noise()

# wasatch_1_noisy_high = add_modeled_noise(wasatch_1, NHNM[0])
# wasatch_2_noisy_high = add_modeled_noise(wasatch_2, NHNM[0])

# wasatch_1_noisy_low = add_modeled_noise(wasatch_1, NLNM[0])
# wasatch_2_noisy_low = add_modeled_noise(wasatch_2, NLNM[0])

# wasatch_1_noisy_high.plot()
# wasatch_2_noisy_high.plot()

# wasatch_1_noisy_low.plot()
# wasatch_2_noisy_low.plot()



# Build and plot a database of noisy synthetic streams

# build_modeled_noise_dataset()
# plot_modeled_noise_dataset()



#%% Make stochastic noise dataset


#### Just turn into another function ####

# save_noise = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/noise/data/miniseed/stochastic/"
# save_fig = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/figs/noise_generation/stochastic/"

# N_series = 50
# nsamp = 1798
# samp_interv = 0.02559485

# noise = Stream()

# for i in range(N_series):
#     noise_amp = gen_band_limited_noise(0.01, 40, nsamp, sampling_rate)

#     tr = Trace(noise_amp)

#     tr.stats.npts = nsamp
#     tr.stats.delta = samp_interv

#     noise += tr

#     filename = "stochastic_noise-" + str(i)

#     tr.plot(outfile=save_fig + filename + ".png")
#     # tr.write(save_noise + filename + ".ms", format="MSEED")
