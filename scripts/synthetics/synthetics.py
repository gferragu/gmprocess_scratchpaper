# -*- coding: utf-8 -*-

# %% Imports

import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from obspy import Trace
from obspy.core.stream import Stream
from obspy import read
import obspy.signal.filter
import obspy.signal.spectral_estimation as spec_es

# from unit_impulse import get_psd

# %% Summary from Kyle

"""
I've extracted two stations from one of my Wasatch simulations (dipping normal fault, Mw~6.9) at a distance near the fualt and one farther away. The files are attached. You can load them in with something like this:

fileID4 = '/wasatch1.Hslice1seisxdec'
trup=np.fromfile(fileID4, dtype='float32', count=-1, sep='', offset=0)

fileID4 ='wasatch1.Hslice1seisydec'
trup1=np.fromfile(fileID4, dtype='float32', count=-1, sep='', offset=0)


 The time step is: 0.02559485 seconds.
The two files are the two horizontal components (x,y) concatenated for the two stations. Each station has 1798 time steps.
They are in velocity (m/s) and I haven't filtered them. They are computationally accurate up to 6-7 Hz.
"""


"""
This is a sampling rate of 39.07035985754947 Hz
Total time is 46.019540299999996 seconds
"""
# %% Functions
def load_synthetics(filename, dtype='float32', count=-1, offset=0):
    """


    Return.

    -------
    None.

    """
    signal = np.fromfile(filename, dtype='float32', count=-1, sep='', offset=0)
    return signal

def process_synthetics(signal, samp_interv=0.02559485, nsamp=1798, multistation=True):
    """


    Return.

    -------
    None.

    """
    if multistation:
        n_series = len(signal)/nsamp
        if (n_series%1.0) == 0.0:
            n_series = int(len(signal)/nsamp)
        else:
            print("Non-standard time series length encountered, exiting ...")
            exit
        print("With nsamp = " + str(nsamp) + ", there were " + str(n_series) + " time series found in the file")
        signals = []
        T = []
        for i in range(n_series):
            t_min=0
            t_max=nsamp*samp_interv
            t = np.linspace(t_min, t_max, nsamp)
            T.append(t)

            signals.append(signal[nsamp*i:nsamp*(i+1)])
            # signals.append(signal[nsamp:])
        return T, signals
    else:
        t_min=0
        t_max=nsamp*samp_interv
        t = np.linspace(t_min, t_max, nsamp)
        return t, signal

def read_synthetic_streams(filename="*.ms", path="/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/synthetics/data/miniseed/clean/"):
    print("\nNote: Synthetics are in m/s \n")
    dir_list = glob.glob(path + filename)
    # streams = np.zeros(len(dir_list), dtype=object)
    streams = []
    for i in range(len(dir_list)):
        tmp = read(dir_list[i])
        streams.append(tmp)

    return streams

def save_synthetics(stream, file_format="MSEED", filename="default", path="/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/synthetics/data/miniseed/clean/"):
    if file_format == "MSEED":
        stream.write(path + filename + ".", format="MSEED")
    else:
        print("Unsupported file format, try 'MSEED' instead")



# def plot_synthetics(t, signal, title="Synthetic Time Series", x_label="Time (s)", y_label="Velocity (m/s)", return_handle=False):
#     """


#     Return.

#     -------
#     None.

#     """
#     fig = plt.figure()
#     plt.plot(t, signal, 'k-', linewidth=0.5)
#     # plt.scatter(t, signal)
#     plt.title(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)

#     if return_handle:
#         return fig


# def calculate_amplitude_spectra():
#     """


#     Return.

#     -------
#     None.

#     """




# %% Script
data_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/test_data/synthetic/"
# fig_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/figs/noise_generation/"
fig_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/figs/synthetics/"


npts=1798
delta=0.02559485

fileID4 = 'wasatch1.Hslice1seisxdec'
# trup=load_synthetics(data_path + fileID4)
# trup=np.fromfile(data_path + fileID4, dtype='float32', count=-1, sep='', offset=0)
trup=np.fromfile(data_path + fileID4)


fileID4 ='wasatch1.Hslice1seisydec'
# trup1=load_synthetics(data_path + fileID4)
# trup1=np.fromfile(data_path + fileID4, dtype='float32', count=-1, sep='', offset=0)
trup1=np.fromfile(data_path + fileID4)



# [Xt, Xsignal] = process_synthetics(trup)
# [Yt, Ysignal] = process_synthetics(trup1)

st1 = Stream()
st2 = Stream()

# tr1_X = Trace(Xsignal[0])
# tr1_Y = Trace(Ysignal[0])
# tr1_X.stats.npts = tr1_Y.stats.npts = npts
# tr1_X.stats.delta = tr1_Y.stats.delta = delta
# tr1_X.stats.channel="X"
# tr1_Y.stats.channel="Y"
# tr1_X.stats.network = tr1_Y.stats.network = "SY"

# tr2_X = Trace(Xsignal[1])
# tr2_Y = Trace(Ysignal[1])
# tr2_X.stats.npts = tr2_Y.stats.npts = npts
# tr2_X.stats.delta = tr2_Y.stats.delta = delta
# tr2_X.stats.channel="X"
# tr2_Y.stats.channel="Y"
# tr2_X.stats.network = tr2_Y.stats.network = "SY"

# st1 += tr1_X ; st1 += tr1_Y
# st2 += tr2_X ; st2 += tr2_Y

# st1_filename="wasatch_synth1_horzXY"
# st2_filename="wasatch_synth2_horzXY"

# st1.plot(outfile=fig_path + st1_filename + '.png')
# st2.plot(outfile=fig_path + st2_filename + '.png')

# st1.write(data_path + st1_filename + ".ms",format="MSEED")
# st2.write(data_path + st2_filename + ".ms",format="MSEED")


# %% Test Reading miniseed back in

test1 = read("/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/test_data/synthetic/wasatch_synth1_horzXY.ms")
test2 = read("/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/test_data/synthetic/wasatch_synth2_horzXY.ms")

test1.plot()
test2.plot()

# %% Manual Plotting stuff
# plot_synthetics(Xt[1], Xsignal[1])
# plot_synthetics(Xsignal[0], Xsignal[1])
# plot_synthetics(Xt[0], Yt[0])

# # What am I overthinking here ...
# t = np.linspace(0,1797,1798)s
# plot_synthetics(t, Xsignal[0])

# fig = plt.figure
# # plt.plot(Xsignal[0])
# plot_synthetics(t, Xsignal[0])
# # plt.xlim(85,250)
# # plt.ylim(-0.17,0.17)
# # plt.savefig(fig_path + "synthetic_signal_zoomed.png", dpi=500)
# plt.savefig(fig_path + "synthetic_signal_fullview.png", dpi=500)

# %% Getting power spectra from synthetics

# Recall
npts=1798
delta=0.02559485
fs=1/delta

fig_name1 = "fft_synth_seismogram-"
fig_name2 = "psd_synth_seismogram-"


# %% McNmara / Buland

# Power def:
# P_k =  ((2*∆t)/N) * abs(Y_k)^2 , where Y_k is the discrete Fourier component

# Take FFT of synthetics

data = [st1, st2]
spectra = []
PSD = []

for idx, stream in enumerate(data):
    for tr in stream:
        nfft = tr.stats.npts
        delta = tr.stats.delta
        tmax = nfft * delta
        fs = tr.stats.sampling_rate
        stnm = str(idx + 1)
        chan = tr.stats.channel

        t = [0, tmax, delta]
        amp = tr.data

        ft = np.abs(np.fft.rfft(amp, nfft))
        freq = np.fft.rfftfreq(nfft)
        spectra.append([freq, ft, stnm, chan])

        power = []
        for iidx, Y_k in enumerate(ft):
            print(Y_k)
            P_k = ((2 * delta) / nfft) * abs(ft[iidx])**2
            print(P_k)
            power.append(P_k)

        PSD.append([freq, power, stnm, chan])

for dat in spectra:
    # Log x
    plt.semilogx(dat[0], dat[1], label="FFT of " + str(dat[3] + " comp."))
    plt.title("FFT of Synthetic Seismogram: Station " + dat[2])
    plt.ylabel("Amplitude (m/s)/Hz")
    plt.xlabel("Hz")
    plt.legend()
    plt.savefig(fname=fig_path + fig_name1 + "station" + dat[2] + "_" + dat[3] + "-logx.png", dpi=500)
    plt.show()

for dat in PSD:
    # Log x
    plt.semilogx(dat[0], dat[1], label="PSD of " + str(dat[3] + " comp."))
    plt.title("PSD after McNamara and Buland (2004): Station " + dat[2])
    plt.ylabel("PSD (non-decibel reference) (m/s)^2/Hz")
    plt.xlabel("Hz")
    plt.legend()
    plt.savefig(fname=fig_path + fig_name2 + "station" + dat[2] + "_" + dat[3] + "-logx.png", dpi=500)
    plt.show()


#%% Adding noise

test = read_synthetic_streams()
