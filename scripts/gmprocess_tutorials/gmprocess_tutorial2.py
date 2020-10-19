# -*- coding: utf-8 -*-

import os
import pkg_resources

from gmprocess.streamcollection import StreamCollection
from gmprocess.config import get_config
from gmprocess.processing import process_streams
from gmprocess.event import get_event_object

gm_home = "/Users/gabriel/groundmotion-processing/"
# data = "/gmprocess/data/testdata/knet/us2000cnnl/"
data = "/data/testdata/demo/ci38457511/raw/"
output = "/Users/gabriel/groundmotion-processing/output/"

# Path to example data
datapath = os.path.join('data', 'testdata', 'demo', 'ci38457511', 'raw')
datadir = pkg_resources.resource_filename('gmprocess', datapath)

sc = StreamCollection.from_directory(datadir)

# Includes 3 StationStreams, each with 3 StationTraces
sc.describe()

# Get the default config file
conf = get_config()

# Get event object
event = get_event_object('ci38457511')

# Process the streams
psc = process_streams(sc, event, conf)
psc.describe()

# Save plots of processed records
for st in psc:
    if os.path.exists(output + "test/"):
        st.plot(outfile='%stest/%s.png' % (output, st.get_id()))
    else:
        os.mkdir(os.path.join(output, "test/"))
        st.plot(outfile='%stest/%s.png' % (output, st.get_id()))
