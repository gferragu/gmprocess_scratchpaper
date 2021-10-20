#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:59:52 2021

@author: gabriel
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues March 1

@author: gabriel
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import obspy
# import cartopy
import scipy.constants as sp
from scipy import integrate
from obspy import read
from obspy.core import UTCDateTime
from obspy.core.stream import Stream
from obspy.core.stream import Trace
# from obspy.signal.interpolation import interpolate
from obspy.signal.interpolation import plot_lanczos_windows
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader
from obspy.clients.fdsn.mass_downloader import CircularDomain
from obspy.clients.fdsn.mass_downloader import RectangularDomain

plt.rcParams['font.size'] = 6
# plt.style.use('ggplot')
plt.style.use('seaborn')

sys.path.append(os.path.abspath("/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/"))
sys.path.append(os.path.abspath("/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/noise/"))
sys.path.append(os.path.abspath("/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/denoise/ "))
sys.path.append(os.path.abspath("/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/synthetics/"))
sys.path.append(os.path.abspath("/Users/gabriel/groundmotion-processing/gmprocess/"))

# import synthetics as synth                # Worked last time, now doesn't???
import synthetics.synthetics as synth       # Failed last time, now works??!

"""
    Attempting to identify any continuous strong motion accelerometers
    available in different ObsPy clients in order to grab noise windows

    An initial test case is centered on the USC campus with a radius of 1,
    and specifying stations with channels *N* (strong motion) that are
    designated "permanent" networks with endtime 3000-01-01

    Networks fitting those criteria are AM, CE, CI, CJ, FA, NP, and WR

"""
#%% Set paths
data_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/noise/data/miniseed/real/"
fig_path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/figs/real_noise/"


#%% Start building functions to make this reproducable and easy

def resample_for_synth(st, npts, dt):
    """


    Parameters
    ----------
    st : TYPE
        DESCRIPTION.
    npts : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    st_npts = st.stats.npts
    st_dt = st.stats.delta
    st_t_range = st_npts * st_dt

    # Unfinished

    return None


def get_arias(st, units='m/s/s'):
    """
    Basically stealing the code from gmprocess for this:
    "Performs calculation of arias intensity"

    Returns
    -------
            arias_intensities: Dictionary of arias intensity for each channel.
    """

    acc = []
    gal_2_pctg = (1 / (2 * sp.g))

    arias_st = Stream()

    for trace in st:
        dt = trace.stats.delta

        # Do I need to convert from cm/s to m/s?
        if units == 'm/s/s':
            acc = trace.data
        elif units == 'cm/s/s':
            acc = trace.data * 0.01

        intgrd_sq_acc = integrate.cumtrapz(acc * acc, dx=dt)
        arias_intens = intgrd_sq_acc * np.pi * gal_2_pctg

        stats = trace.stats.copy()
        # chan = stats.channel
        # stats.standard.units = 'vel'
        stats.npts = len(arias_intens)
        arias_tr = Trace(arias_intens)
        arias_tr.stats = stats
        arias_st += arias_tr

    return arias_st


def cut_earthquake():

    # Unfinished

    return None


def download_noise_windows(yr=2021, mnth=3, day=4, tod='morning'):

    morning = [UTCDateTime(yr, mnth, day, 6), UTCDateTime(yr, mnth, day, 9)]

    afternoon = [UTCDateTime(yr, mnth, day, 12), UTCDateTime(yr, mnth, day, 15)]

    evening = [UTCDateTime(yr, mnth, day, 18), UTCDateTime(yr, mnth, day, 21)]

    night = [UTCDateTime(yr, mnth, day, 22, 30),
             UTCDateTime(yr, mnth, day + 1, 1, 30)]

    restrictions = Restrictions(
        starttime=morning[0],
        endtime=morning[1],
        chunklength_in_sec=46,
        network=ntwk_query[2], station="USC", location="", channel="?N?",
        reject_channels_with_gaps=False,
        minimum_length=0.0,
    )

def check_stream_sampling_rate(st):
    N_traces = len(st)
    fs = np.zeros(N_traces)
    for i in range(N_traces):
        fs_tmp = (1 / st[i].stats.delta)
        fs[i] = fs_tmp

    # Check for fs that remain unset
    if any in fs == 0:
        print("\nWarning: Some sampling rates are 0\n")
    # Check if all fs are equal
    elif all(x == fs[0] for x in fs):
        print("\nNote: All sampling rates are equal: " + str(fs[0]) + " Hz\n")
        return fs[0]
    else:
        print("\nNote: Variable sampling rates, returning array of them \n")
        return fs


def check_stream_npts(st):
    N_traces = len(st)
    npts = np.zeros(N_traces)
    for i in range(N_traces):
        npts_tmp = st[i].stats.npts
        npts[i] = npts_tmp

    # Check for npts that remain unset
    if any in npts == 0:
        print("\nWarning: Some NPTS are 0\n")
    # Check if all npts are equal
    elif all(x == npts[0] for x in npts):
        print("\nNote: All NPTS are equal: " + str(npts[0]) + "\n")
        return npts[0]
    else:
        print("\nNote: Variable NPTS, returning array of them \n")
        bins = 24
        plt.figure()
        plt.hist(npts, bins=bins, label="# of Bins:"+str(bins))
        plt.title("Histogram of NPTS for Trace() objects in Stream()")
        plt.xlabel("NPTS in Trace")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        return npts

def check_stream_t(st):
    N_traces = len(st)
    times = np.zeros(N_traces)

    for i in range(N_traces):
        t_tmp = st[i].stats.endtime - st[i].stats.starttime
        times[i] = t_tmp

    # Check for time windows that remain unset
    if any in times == 0:
        print("\nWarning: Some time windows are 0 s\n")
    # Check if all fs are equal
    elif all(x == times[0] for x in times):
        print("\nNote: All time windows are equal: " + str(times[0]) + " s\n")
        return times[0]
    else:
        print("\nNote: Variable time windows, returning array of them \n")
        bins = 24
        plt.figure()
        plt.hist(times, bins=bins, label="# of Bins:"+str(bins))
        plt.title("Histogram of Time Windows for Trace() objects in Stream()")
        plt.xlabel("Time Window for Trace (s)")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        return times

    #%% Set up query parameters
    # Pick a time range
t1 = UTCDateTime(2019, 1, 1)
t2 = UTCDateTime(2020, 1, 1)
end = UTCDateTime(3000, 1, 1)          # Appears to be last possible date

# Investigate a single station, such as the one on USC campus
stnm = "USC1"

# Channel ID of ?N? corresponds to all strong motion accelerometers
chan = "*N*"  # "?N?"

# Coordinates of Interest (seismometer on USC campus)
lat = 34.023953
lon = -118.285117
r = 0.1

# Choose a client
SCEDC = Client("SCEDC")
#IRIS = Client("IRIS")
# USGS = Client("USGS")

# Possible networks (permanent, strong motion, long operations)
"""
    These are the networks showing stations within 1° of USC campus on the IRIS
    station browser that run until 3000-01-01 and have channel *N*

    Seems like most of these are from SCEDC
"""
ntwk_query = ['AM', 'CE', 'CI', 'CJ', 'FA', 'NP', 'WR']

#%% Retrieve stations
# IRIS_inv = IRIS.get_stations(
#     starttime=t1, endtime=t2, latitude=lat, longitude=lon, maxradius=r)
# SCEDC_inv = SCEDC.get_stations(
#     starttime=t1, endtime=t2, latitude=lat, longitude=lon, maxradius=r)

# # Print preview
# print(SCEDC_inv)

# # Quick and dirty network plot facilitated by ObsPy
# SCEDC_inv.plot(projection="local", resolution="f")

#%% Use the bulk waveform downloader for smaller datasets
# ntwks = []
# stns = []
# stns_coord = []

# station_meta = []
# bulk = []
# US = []

# # Get station information from inventory for waveform download
# networks = SCEDC_inv.networks

# for ntwk in networks:
#     stations = []
#     ntwk_code = ''
#     station_code = ''

#     if not (ntwk.code in ntwks):
#         ntwk_code = ntwk.code
#         ntwks.append(ntwk.code)

#     for stn in ntwk:
#         stn_code = stn.code
#         stn_coord = [stn.longitude, stn.latitude]

#         if not (stn.code in stns):
#             stns.append(stn_code)
#             stations.append(stn_code)
#             stns_coord.append([stn.longitude, stn.latitude])

#         bulk.append((ntwk_code, stn_code, "*", chan, t1, t2))

#         if ntwk_code == "US":
#             US.append((ntwk_code, stn_code, "*", chan, t1, t2))

#     station_meta.append((ntwk_code, stations))


# st_full = Stream()

# st_full = IRIS.get_waveforms_bulk(bulk)
# st_US = IRIS.get_waveforms_bulk(US)

#%% Use MassDownloader for continuous datasets
"""
    For continuous requests, using the MassDownloader may be required. This
    approach is more flexible and is useful for continuous data. The docs for
    it are here:
    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html?highlight=continuous
"""
domain = CircularDomain(latitude=lat, longitude=lon,
                        minradius=0.0, maxradius=r)

t1 = UTCDateTime(2021, 3, 2)
t2 = UTCDateTime(2021, 3, 5)

restrictions = Restrictions(
    # Get data for a whole year.
    starttime=t1,
    endtime=t2,

    # 1 day chunks
    chunklength_in_sec=86400,

    # If the location code is specified, the location priority list is not
    # used; the same is true for the channel argument and priority list.
    network=ntwk_query[2], station="USC", location="", channel="?N?",
    # The typical use case for such a data set are noise correlations where
    # gaps are dealt with at a later stage.
    reject_channels_with_gaps=False,
    # Same is true with the minimum length. All data might be useful.
    minimum_length=0.0,
    # Guard against the same station having different names.
    #minimum_interstation_distance_in_m=100.0
)

mdl = "tmp"
download = False

if download:
    mdl = MassDownloader(providers=["SCEDC"])
    mdl.download(domain, restrictions, mseed_storage=data_path + "waveforms/USC/",
                 stationxml_storage=data_path + "stations/")

#%% Read in a couple test waveforms
plots = False

HNx_st = Stream()
LNx_st = Stream()

# High Broad Band (H??)
HNx_st += read(data_path + "waveforms/USC/*HN*.mseed")
# Long Period (L??)
LNx_st += read(data_path + "waveforms/USC/*LN*.mseed")

if plots:
    HNx_st.plot()
    LNx_st.plot()


#%% Try to get Arias intensity ...
HNx_st = HNx_st.detrend()
LNx_st = LNx_st.detrend()

HNx_arias = get_arias(HNx_st)
LNx_arias = get_arias(LNx_st)

#%% Plot Arias
HNx_arias.plot()
LNx_arias.plot()

# ... Not what I expected ...
# Derp, needed to detrend at the very least

#%% Get a window around an earthquake to look at Arias inensity

# There is definitely an even late in the evening March 4th
t1 = UTCDateTime("2021-03-04T19:15:00")
# Using 24:00:00 throws error
t2 = UTCDateTime("2021-03-04T21:30:00")

HNx_st_eq = HNx_st.slice(starttime=t1, endtime=t2)
LNx_st_eq = LNx_st.slice(starttime=t1, endtime=t2)

HNx_arias_eq = HNx_arias.slice(starttime=t1, endtime=t2)
LNx_arias_eq = LNx_arias.slice(starttime=t1, endtime=t2)

HNx_st_eq = HNx_st_eq.plot()
LNx_st_eq = LNx_st_eq.plot()

HNx_arias_eq.plot()
LNx_arias_eq.plot()

#%% Try X second chunk times, discard if arias intensity rises?
yr = 2021
mnth = 3
day = 4

morning = [UTCDateTime(yr, mnth, day, 6), UTCDateTime(yr, mnth, day, 9)]

afternoon = [UTCDateTime(yr, mnth, day, 12), UTCDateTime(yr, mnth, day, 15)]

evening = [UTCDateTime(yr, mnth, day, 18), UTCDateTime(yr, mnth, day, 21)]

night = [UTCDateTime(yr, mnth, day, 22, 30),
         UTCDateTime(yr, mnth, day + 1, 1, 30)]

# tod = night
tods = [morning, afternoon, evening, night]
tod_nms = ["morning", "afternoon", "evening", "night"]

# chunklength_in_sec=46

for idx,tod in enumerate(tods):
    restrictions = Restrictions(
        starttime=tod[0],
        endtime=tod[1],
        chunklength_in_sec=30,
        network=ntwk_query[2], station="USC", location="", channel="?N?",
        reject_channels_with_gaps=True,
        minimum_length=0.0,
    )

    mdl = "tmp"
    download = False
    # tod = "night"

    if download:
        mdl = MassDownloader(providers=["SCEDC"])
        mdl.download(domain, restrictions, mseed_storage=data_path + "waveforms/USC_" + tod_nms[idx] + "/",
                     stationxml_storage=data_path + "stations/")


#%% Load
# tod_select = "afternoon"
tod_select = "night"

HNx_st = Stream()
LNx_st = Stream()

# High Broad Band (H??)
HNx_st += read(data_path + "waveforms/USC_" + tod_select + "/*HN*.mseed")
HNx_st.detrend()

# Long Period (L??)
LNx_st += read(data_path + "waveforms/USC_" + tod_select + "/*LN*.mseed")
LNx_st.detrend()

plots = False
arias = False

if plots:
    HNx_st.plot()
    LNx_st.plot()

    for tr in HNx_st:
        tr.plot()

    # Check Arias Intensity
    if arias:
        HNx_st = HNx_st.detrend()
        HNx_st_arias = get_arias(HNx_st)

        for tr in HNx_st_arias:
            tr.plot()




#%% Try the Lanczos interpolation to resample to 30 seconds

"""
If used for downsampling, ***make sure to apply an appropriate anti-aliasing lowpass filter first***

***Values of a >= 20 show good results even for data that has energy close to the Nyquist frequency. If your data is extremely oversampled you can get away with much smaller a‘s***

Also be aware of any boundary effects. All values outside the data range are assumed to be zero which matters when calculating interpolated values at the boundaries. At each side the area with potential boundary effects is a * old_dt. If you want to avoid any boundary effects you will have to remove these values.

Syntax:
Trace/Stream.interpolate(sampling_rate, method='weighted_average_slopes', starttime=None, npts=None, time_shift=0.0, *args, **kwargs)

lanczos_interpolation(data, old_start, old_dt, new_start, new_dt, new_npts, a, window='lanczos', *args, **kwargs)
"""
interp = False

if interp:
    # Choose parameters
    a = 1                           # Can test with plot_lanczos_windows()
    fs_synth = 39.07035985754947    # From Kyle
    npts_synth = 1798               # From Kyle
    windows = ["blackman",          # Just defaulting to Lanczos
               "hanning",
               "lancsoz"]
    new_t = 30                      # in s
    new_fs = 35                     # in Hz

    # Data features
    # fs_dat = 100.0
    fs_dat = check_stream_sampling_rate(HNx_st)     # 100Hz
    npts_dat = check_stream_npts(HNx_st)            # Variable
    t_dat = check_stream_t(HNx_st)                  # Variable

    # Calculated values based on data and targets
    new_npts = npts_dat

    """  Check out the different tapers that can be used for Lanczos interpolation
         for a particular value of 'a' plot_lanczos_windows(a=a)
    """
    plot_lanczos_windows(a=a)
    # plot_lanczos_windows(a=a, filename=fig_path+"lanczos_windows_a="+str(a)+".png")


    # Is this easier to do in a loop (check values for each trace?)
    HNx_resamp = HNx_st.copy()
    HNx_resamp = HNx_resamp.interpolate(fs_dat, "lanczos", a=a, window=windows[0])

#%% Plot this

    plots = False
    if plots:
        HNx_resamp.plot()

        for tr in HNx_resamp:
            tr.plot()

#%% Check stream histograms

    fs_dat = check_stream_sampling_rate(HNx_resamp)     # 100Hz
    npts_dat = check_stream_npts(HNx_resamp)            # Variable
    t_dat = check_stream_t(HNx_resamp)                  # Variable

#%% Get a few longer length noise time series to manually add noise to

st_long = Stream()

for tr in HNx_st:
# for tr in HNx_resamp:

    t_lim1  = 70
    t_lim2 = 75
    t_exact = 75

    t_tmp = tr.stats.delta * tr.stats.npts

    if t_lim2 > t_tmp > t_lim1:

        st_long += tr

    # if t_tmp == t_exact:

    #     st_long += tr

#%% Check again
npts_dat2 = check_stream_npts(st_long)
t_dat2 = check_stream_t(st_long)
st_long.plot()

# st_long_common_chan_trim = st_long.copy()
# st_long_common_chan_trim._trim_common_channels()
# st_long_common_chan_trim.plot()

#%% Write out this subset
# st_long.write(data_path + "by_hand_example/3comp_example_noise_75s_100Hz.mseed", formate="MSEED")

#%% Probably an easier way to do this
# [‘network’, ‘station’, ‘location’, ‘starttime’, ‘channel’, ‘endtime’]

# test_st = HNx_st.copy()

# test_st = test_st.sort(['starttime'])

# test_st = test_st._trim_common_channels()

#%% Load in 2 comp test noise
# from noise_generation import add_modeled_noise
# from noise_generation import add_random_noise
import noise_generation as nois_gen


# Load 2 comp horz noise
noise_st = read(data_path + "by_hand_example/horz-comp_example_noise_74s_100Hz.mseed")
noise_st.plot()

# Load synthetics (Note: this is a list of 2 streams, each having 2 components, X and Y)
synth_st = synth.read_synthetic_streams()

# Add noise (automatic, but not explicitly for this purpose)
# --> Need to adjust the underlying functions to handle variable noise windows with Lancsoz resamling <---
# synth_st_noisy = nois_gen.add_modeled_noise(synth_st[0], noise_st[0])
# synth_st_noisy.plot()


#%% Add noise (by hand) #

# Try the Lancsoz interp method
    # --> Need to work out the exact arguments, use simple resample for now

#%% Simple resample
#--> First resample all to 40Hz
target_fs = 40

synth_event1_compX = synth_st[0][0].copy()

t1_synth = synth_event1_compX.stats.starttime
synth_event1_compX.trim(t1_synth, t1_synth + 45)

# Isolate a single trace
noise_tr_compE = noise_st[0].copy()

noise_npts_old = len(noise_tr_compE)
noise_t_tot = noise_tr_compE.stats.delta * noise_npts_old
noise_npts_new = noise_t_tot / (1/target_fs)


# Trace resample method (uses nsamp as the arg)
synth_event1_compX_resamp_simple = synth_event1_compX.copy()
synth_event1_compX_resamp_simple.resample(target_fs)
synth_event1_compX_resamp_simple.plot()

# Trace resample method (uses nsamp as the arg)
noise_resamp_simple = noise_tr_compE.copy()
noise_resamp_simple.resample(target_fs)
noise_resamp_simple.plot()

# Crap, remember synthetics are in m/s, the noise is from an accelerometer
# --> Need to diff the synthetics or integrate the noise
noise_resamp_simple = noise_resamp_simple.copy()
noise_resamp_simple.detrend('linear')
noise_resamp_simple.plot()

# Well ... need to detrend I guess, check what diff'd signal looks like
synth_event1_compX_resamp_simple_acc = synth_event1_compX_resamp_simple.copy()
synth_event1_compX_resamp_simple_acc.differentiate()
synth_event1_compX_resamp_simple_acc.detrend('linear')
synth_event1_compX_resamp_simple_acc.plot()

#%% Add signal in the middle of the noise so there is pure noise before any signal

synth_T = synth_event1_compX_resamp_simple_acc.stats.endtime - synth_event1_compX_resamp_simple_acc.stats.starttime
noise_T = noise_resamp_simple.stats.endtime - noise_resamp_simple.stats.starttime

# Determine an offset to place the signal at
wiggle_room = noise_T - synth_T

# Need to match offset to nearest index/value in the longer noise time series
offset = 0.5 * wiggle_room      # We split the 'wiggle room' to have noise on either side

# Scaling factor
sf = 10

# Man these variable names are awful
synth_tr_tmp = synth_event1_compX_resamp_simple_acc.copy()
noise_tr_tmp = noise_resamp_simple.copy()

dat_s = synth_tr_tmp.data * sf
t_range_s = synth_tr_tmp.times() + offset

dat_n = noise_tr_tmp.data
t_range_n = noise_tr_tmp.times()

# New adjusted traces
synth_tr_adj = synth_tr_tmp.copy()
noise_tr_adj = noise_tr_tmp.copy()

synth_tr_adj.data = synth_tr_adj.data * sf
synth_tr_adj.stats.starttime = synth_tr_adj.stats.starttime + offset

synth_tr_adj.id = 'SY...X'
noise_tr_adj.id = 'SY...X'

# Pad data
# offset_idx = np.where(noise_tr_adj.times() > offset)[0][0]     # Annoying, returns NumPy array of arrays so need to index
offset_idx = next(i for i in range(len(noise_tr_adj.times()))  # Longer but uses a generator and no indexing
                  if noise_tr_adj.times()[i] > offset)
end_idx = offset_idx + len(synth_tr_adj)

# padded = np.concatenate(np.arange(0, offset_idx, synth_tr_adj.stats.delta) + synth_tr_adj.data + np.arange(end_idx, len(noise_tr_adj), synth_tr_adj.stats.delta))
# padded = np.concatenate([np.zeros(offset_idx), synth_tr_adj.data + np.zeros(len(noise_tr_adj) - end_idx)])

# Convert to lists to allow easy concatenation without worrying about axes and cast back to NumPy array
padded = np.array(list(np.zeros(offset_idx)) + list(synth_tr_adj.data) + list(np.zeros(len(noise_tr_adj) - end_idx)))
overlay = np.add(padded, noise_tr_adj)

# Cast data as an ObsPy stream or trace
noisy_sig_trace = noise_tr_adj.copy()
noisy_sig_trace.data = overlay

padded_sig_trace = noise_tr_adj.copy()
padded_sig_trace.data = padded

# Plot
fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)

ax1.plot(t_range_s, dat_s)
ax2.plot(t_range_n, dat_n)
ax3.plot(t_range_n, overlay)

# synth_tr_adj.plot()
# noise_tr_adj.plot()



#%% Now try to denoise the traces via wavelet method
from denoise import dwt
from denoise import utils

###################
### Save Plots? ###
###################
saveplot=False

# Let's get things cleaned up

# May need to convert to Stream
# noisy_sig_st = Stream(traces=[noisy_sig_tr])

# Do a simple denoising
# test_denoise =dwt.denoise(noisy_sig_st)

# There's an issue with removing pre-event noise, try denoise_trace instead
test_denoise_trace = noisy_sig_trace.copy()     # Make copy ,otherwise it operates on original data
test_denoise_trace = dwt.denoise_trace(test_denoise_trace, store_noise=True)

# --> Troubleshooting dwt and utils function arguments
#       - remove_pre_event_noise was missing the noiseCoeffs arg => fixed
#       - now soft_threshold takes 4 positional arguments, but 5 were given
#        => when automating things by abstracting into utils, I forgot to remove the
#           "channelLabel" arg from being passed to soft_threshold => fixed

#%% No time to do things properly and with articulation, try to plot some stuff quick!!
denoised_trace = test_denoise_trace['data']  # (it was returned as a dict)
removed_noise = test_denoise_trace['noise']

denoised_trace.plot()
removed_noise.plot()  # Ok well this looks effed up

# For plotting, get traces & names organized
synth_trace = synth_tr_adj.copy()
noise_trace = noise_tr_adj.copy()
# noisy_sig_trace from above
# denoise_trace from above

# Plot the traces
fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True)

ax1.plot(t_range_n, padded_sig_trace.data)
ax1.set_title("Padded Synthetic Signal", fontweight="bold")

ax2.plot(t_range_n, dat_n)
ax2.set_title("Real Noise Signal", fontweight="bold")

ax3.plot(t_range_n, noisy_sig_trace.data)
# ax3.plot(t_range_n, overlay)
ax3.set_title("Noisy Signal", fontweight="bold")

ax4.plot(t_range_n, denoised_trace.data)
ax4.set_title("Denoised Signal", fontweight="bold")


if saveplot:
    plt.savefig(fig_path + "trace_comparisons_update.png", dpi=500)


#%% FFTs
# fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True)

# ax1.plot(t_range_s, dat_s)
# ax2.plot(t_range_n, dat_n)
# ax3.plot(t_range_n, overlay)
# ax4.plot(t_range_n, denoised_trace.data)

#%% Spectrograms
fig, axs = plt.subplots(4, 1, figsize=(6, 8))

fig.suptitle("Spectrograms for Original, Noise, Noisy, and Denoised Time Series",
             fontsize=14, fontweight="bold")

padded_sig_trace.spectrogram(axes=axs[0])
axs[0].set_title("Padded Synthetic Signal", fontweight="bold")
noise_trace.spectrogram(axes=axs[1])
axs[1].set_title("Real Noise Signal", fontweight="bold")
noisy_sig_trace.spectrogram(axes=axs[2])
axs[2].set_title("Noisy Signal", fontweight="bold")
denoised_trace.spectrogram(axes=axs[3])
axs[3].set_title("Denoised Signal", fontweight="bold")

fig.subplots_adjust(hspace=0.5)

if saveplot:
    plt.savefig(fig_path + "spectrogram_comparisons_update.png", dpi=500)

#%% Combined figs
# from matplotlib import gridspec

fig, axs = plt.subplots(4, 2, gridspec_kw={
                           'width_ratios': [3, 1]}, figsize=(8, 6))
# fig, axs = plt.subplots(4, 2)   #, figsize=(6, 8))

fig.suptitle("Noise Introduction and Denoising Results",
              fontsize=14, fontweight="bold")

axs[0,0].plot(t_range_n, padded_sig_trace.data, label="Padded Synthetic Signal")
# axs[0,0].set_title("Padded Synthetic Signal", fontweight="bold")

axs[1,0].plot(t_range_n, dat_n, label="Real Noise Signal")
# axs[1,0].set_title("Real Noise Signal", fontweight="bold")

axs[2,0].plot(t_range_n, noisy_sig_trace.data, label="Noisy Signal")
# axs[2,0].set_title("Noisy Signal", fontweight="bold")

axs[3,0].plot(t_range_n, denoised_trace.data, label="Denoised Signal")
# axs[3,0].set_title("Denoised Signal", fontweight="bold")


padded_sig_trace.spectrogram(axes=axs[0,1])
# axs[0,1].set_title("Padded Synthetic Signal", fontweight="bold")
# axs[0,1].set_title("Corresponding Spectrograms", fontweight="bold")
noise_trace.spectrogram(axes=axs[1,1])
# axs[1,1].set_title("Real Noise Signal", fontweight="bold")
noisy_sig_trace.spectrogram(axes=axs[2,1])
# axs[2,1].set_title("Noisy Signal", fontweight="bold")
denoised_trace.spectrogram(axes=axs[3,1])
# axs[3,1].set_title("Denoised Signal", fontweight="bold")

# fig.subplots_adjust(hspace=0.75)
plt.tight_layout()
fig.subplots_adjust(top=0.92)

## This is so awesome
# plt.subplot_tool()

plt.legend()

if saveplot:
    plt.savefig(fig_path + "trace_and_spectrogram_comparisons.png", dpi=500)