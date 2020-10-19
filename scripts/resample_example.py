# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

x = np.linspace(0, 10, 20, endpoint=False)

y = np.cos(-x**2/6.0)

f = signal.resample(y, 100)

xnew = np.linspace(0, 10, 100, endpoint=False)

plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')

plt.legend(['data', 'resampled'], loc='best')

plt.show()
