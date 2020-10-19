# -*- coding: utf-8 -*-

import os
import sys
import pkg_resources
import glob

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

