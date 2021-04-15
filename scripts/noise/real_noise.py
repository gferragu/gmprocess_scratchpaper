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


import numpy as np
import matplotlib.pyplot as plt
import obspy
import cartopy
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

    # Check for fs that remain unset
    if any in npts == 0:
        print("\nWarning: Some NPTS are 0\n")
    # Check if all fs are equal
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
t2 = UTCDateTime(2021, 3, 3)

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
plots = True

HNx_st = Stream()
LNx_st = Stream()

if plots:
    # High Broad Band (H??)
    HNx_st += read(data_path + "waveforms/USC/*HN*.mseed")
    # Long Period (L??)
    LNx_st += read(data_path + "waveforms/USC/*LN*.mseed")

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

#%% Try 46 second chunk times, discard if arias intensity rises?
yr = 2021
mnth = 3
day = 4

morning = [UTCDateTime(yr, mnth, day, 6), UTCDateTime(yr, mnth, day, 9)]

afternoon = [UTCDateTime(yr, mnth, day, 12), UTCDateTime(yr, mnth, day, 15)]

evening = [UTCDateTime(yr, mnth, day, 18), UTCDateTime(yr, mnth, day, 21)]

night = [UTCDateTime(yr, mnth, day, 22, 30),
         UTCDateTime(yr, mnth, day + 1, 1, 30)]

tod = night

restrictions = Restrictions(
    starttime=tod[0],
    endtime=tod[1],
    chunklength_in_sec=46,
    network=ntwk_query[2], station="USC", location="", channel="?N?",
    reject_channels_with_gaps=False,
    minimum_length=0.0,
)

mdl = "tmp"
download = False
tod = "night"

if download:
    mdl = MassDownloader(providers=["SCEDC"])
    mdl.download(domain, restrictions, mseed_storage=data_path + "waveforms/USC_" + tod + "/",
                 stationxml_storage=data_path + "stations/")


#%% Plot
tod = "afternoon"

plots = True
arias = True

HNx_st = Stream()
LNx_st = Stream()

# High Broad Band (H??)
HNx_st += read(data_path + "waveforms/USC_" + tod + "/*HN*.mseed")
# Long Period (L??)
LNx_st += read(data_path + "waveforms/USC_" + tod + "/*LN*.mseed")

if plots:
    HNx_st.plot()
    LNx_st.plot()

    for tr in HNx_st:
        tr.plot()

    # Check Arias Intensity
    if arias:
        HNx_st = HNx_st.detrend()
        HNx_st_arias = get_arias(HNx_st)


#%% Try the Lanczos interpolation to resample to 30 seconds

"""
If used for downsampling, ***make sure to apply an appropriate anti-aliasing lowpass filter first***

***Values of a >= 20 show good results even for data that has energy close to the Nyquist frequency. If your data is extremely oversampled you can get away with much smaller a‘s***

Also be aware of any boundary effects. All values outside the data range are assumed to be zero which matters when calculating interpolated values at the boundaries. At each side the area with potential boundary effects is a * old_dt. If you want to avoid any boundary effects you will have to remove these values.

Syntax:
Trace/Stream.interpolate(sampling_rate, method='weighted_average_slopes', starttime=None, npts=None, time_shift=0.0, *args, **kwargs)

lanczos_interpolation(data, old_start, old_dt, new_start, new_dt, new_npts, a, window='lanczos', *args, **kwargs)
"""

# Choose parameters
a = 1
# fs_dat = 100.0
fs_synth = 39.07035985754947
windows = ["blackman", "hanning", "lancsoz"]

# Data features
fs_dat = check_stream_sampling_rate(HNx_st)
npts_dat = check_stream_npts(HNx_st)

# Check out the different tapers that can be used
# for Lanczos interpolation for a particular value of 'a'
plot_lanczos_windows(a=a)
# plot_lanczos_windows(a=a, filename=fig_path+"lanczos_windows_a="+str(a)+".png")


# Is this easier to do in a loop (check values for each trace?)
HNx_resamp = HNx_st.copy()
HNx_resamp = HNx_resamp.interpolate(fs_dat, "lanczos", a=a, window=windows[0])

HNx_resamp.plot()

