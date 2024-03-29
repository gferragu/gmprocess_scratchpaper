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
from obspy import read
from obspy.core import UTCDateTime
from obspy.core.stream import Stream
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader
from obspy.clients.fdsn.mass_downloader import CircularDomain
from obspy.clients.fdsn.mass_downloader import RectangularDomain
plt.rcParams['font.size'] = 6

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

t1 = UTCDateTime(2021, 3, 3)
t2 = UTCDateTime(2021, 3, 6)

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
# test_st = Stream()
plots = True

if plots:
    # High Broad Band (H??)
    HNx_st = read(data_path + "waveforms/USC/*HN*.mseed")
    # Long Period (L??)
    LNx_st = read(data_path + "waveforms/USC/*LN*.mseed")

    HNx_st.plot()
    LNx_st.plot()

#%% Try different plot types

# st.plot(type="section")

# plt_nm = "rel_plot-full_stream.png"
# st.plot(type="relative", outfile=fig_path+plt_nm)

# st.plot(type="dayplot")

# st.filter("bandpass", freqmin=0.1, freqmax=12)

#%% Plot traces in a stream iteratively

# for tr in st:
#     tr.stats.ev_coords = [lon, lat]
#     i = stns.index(tr.stats.station)
#     # tr.stats.longitude = stns_coord[i][0]
#     # tr.stats.latitude = stns_coord[i][1]
#     tr.stats.coordinates.longitude = stns_coord[i][0]
#     tr.stats.coordinates.latitude = stns_coord[i][1]

#     stats = tr.stats
#     # plt_nm = str(stats.network) + "_" + str(stats.station) + "-15min_pre_post-" + str(stats.channel) + ".png"
#     plt_nm = str(stats.network) + "_" + str(stats.station) + \
#         "_" + str(stats.channel) + ".png"

#     tr.detrend()
#     tr.filter("bandpass", freqmin=2, freqmax=10)
#     tr.plot(outfile=fig_path + "traces/" + plt_nm)

# st.plot(type="section", time_down=False, fillcolors=('black', 'None'),
#              linewidth=.25, grid_linewidth=.25, dist_degree=True, ev_coord=[lon, lat])
#              linewidth=.25, grid_linewidth=.25, dist_degree=True, ev_coord=[lon, lat])
