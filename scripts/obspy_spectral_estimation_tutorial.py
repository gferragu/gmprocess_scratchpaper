#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 18:44:47 2020

@author: gabriel
"""
import matplotlib.pyplot as plt

from obspy import read
from obspy.signal import PPSD
from obspy.signal import spectral_estimation as spec

path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/synthetics/data/miniseed/clean/wasatch_synth1_horzXY.ms"


#%% Try the PPSD function

st = read(path)

tr = st.select(channel="EHZ")[0]

paz = {'gain': 60077000.0,

       'poles': [-0.037004+0.037016j, -0.037004-0.037016j,

                 -251.33+0j, -131.04-467.29j, -131.04+467.29j],

       'sensitivity': 2516778400.0,

       'zeros': [0j, 0j]}

ppsd = PPSD(tr.stats, paz)

print(ppsd.id)
print(ppsd.times_processed)

ppsd.add(st)
print(ppsd.times_processed)
ppsd.plot()

ppsd.save_npz("myfile.npz")
ppsd = PPSD.load_npz("myfile.npz")


#%% Try getting the NHNM and NLNM

nhnm = spec.get_nhnm()
plt.semilogx(nhnm[0],nhnm[1])

nlnm = spec.get_nlnm()
plt.semilogx(nlnm[0],nlnm[1])



