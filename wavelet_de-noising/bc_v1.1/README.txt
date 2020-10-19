README.txt

***********
Version 1.0
July 9, 2018

1. BCseis.m is the MatLab Graphical User Interface for interactively investigating a time series file in SAC format.

2. BCseis_process.m is an inline MatLab function for processing many seismograms. A sample driver MatLab program, BCseis_process_driver.m, is provided as an example of processing 4 seismograms contained in the /data directory.

3. The /Help directory contains a set of web pages that can be accessed with a web browser or from the BCseis GUI.  It contains detail and background on the functions of BCseis.

This software was developed with MatLab version R2017b. Anecdotally, a Memphis student using R2015b had problems that could be fixed with a small amount of reprogramming.

I recommend experimenting with the GUI to understand the signal processing involved before trying to use the inline function.

***********
Version 1.1
March 24, 2019

1. Added the ECDF method of finding the noise threshold to the GUI and inline functions.

2. Made small formating changes to the GUI.

Chuck Langston
clangstn@memphis.edu

******************************************************************************************
Disclaimer

This software is provided "as is" and can be freely used.  There are no claims that it is error free or will be useful for your particular application. Use at your own risk.
******************************************************************************************