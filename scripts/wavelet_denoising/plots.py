# -*- coding: utf-8 -*-
#
# ======================================================================
#
#                           Brad T. Aagaard
#                        U.S. Geological Survey
#
#               Modified for gmprocess by Gabriel Ferragut
#              U.S. Geological Survey/ University of Oregon
#
# ======================================================================
#

# stdlib imports
import os
import logging

# local imports
import gmprocess.denoise as dn
from gmprocess import filtering
from gmprocess.smoothing import konno_omachi
from gmprocess.streamcollection import StreamCollection
from gmprocess.io.read import read_data
from gmprocess.processing import process_streams
from gmprocess.logging import setup_logger
from gmprocess.io.test_utils import read_data_dir

# third party imports
import pywt
import numpy
import matplotlib as plt

setup_logger()
#%% Script for plotting data / residuals and denoising

# Loma Prieta test station (nc216859)
data_files, origin = read_data_dir('geonet', 'us1000778i', '*.V1A')
streams = []
for f in data_files:
    streams += read_data(f)

fig1, fig = dn.utils.wavelet_bandpass_comparison(streams)
