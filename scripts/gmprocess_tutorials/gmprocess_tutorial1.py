#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:33:39 2020

@author: gabriel
"""

import glob
from gmprocess.io.read import read_data

# dir = "/Users/gabriel/packages/usgs/gmprocess-fork/"

gm_home = "/Users/gabriel/groundmotion-processing/"
data = "/gmprocess/data/testdata/knet/us2000cnnl/"
output = "/Users/gabriel/groundmotion-processing/output/"

# these sample files can be found in the repository
# under gmprocess/data/testdata/knet/us2000cnnl
# knet files are stored one channel per file.

datafiles = glob.glob(gm_home + data + 'AOM0011801241951.*')
streams = []
for datafile in datafiles:
    streams += read_data(datafile)
