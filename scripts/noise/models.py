# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import signal as sig
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# %% Eventually make some classes

# class noise():

#     def __init__(self):
#         self.npts = 1798
#         self.fs = 39.07035985754947
#         self.dt = 0.02559485
#         self.t = np.arange(0, (self.npts * self.dt), self.dt)


# class NHNM():

#     def __init__(self):
#         self.model = "NHNM"


# class NLNM():

#     def __init__(self):
#         self.model = "NLNM"


# %% Peterson (1993) - OFR 93-322 - New High/Low Noise Model (NHNM/NLNM)

def to_dB(signal):
    N = len(signal)
    dB_series = np.zeros(N)

    for i in range(N):
        dB_series[i] = 10*np.log10(signal[i])

    return dB_series


def to_log(series):
    log = np.log10(series)
    return log


def to_linear(series):
    linear = 10**series
    return linear


def to_Hz(period):
    N = len(period)
    Hz_series = np.zeros(N)
    for i in range(N):
        Hz_series[i] = 1 / period[i]

    return Hz_series


def to_Period(frequency):
    N = len(frequency)
    Hz_series = np.zeros(N)
    for i in range(N):
        if frequency[i] == 0.0:
            # frequency[i] == 0.00001
            # Hz_series[i] = 1 / frequency[i]
            Hz_series[i] = 1 / 0.00001

        else:
            Hz_series[i] = 1 / frequency[i]

    return Hz_series


def get_coeffs(model="high"):
    if model == "high":
        NHNM_coeffs = pd.read_csv('./noise_models/NHNM-coeffs.txt')
        P = np.array(NHNM_coeffs["P"])
        A = np.array(NHNM_coeffs["A"])
        B = np.array(NHNM_coeffs["B"])

        return [P, A, B]

    elif model == "low":
        NLNM_coeffs = pd.read_csv('./noise_models/NLNM-coeffs.txt')
        P = np.array(NLNM_coeffs["P"])
        A = np.array(NLNM_coeffs["A"])
        B = np.array(NLNM_coeffs["B"])

        return [P, A, B]

    else:
        print("Invalid model choice. Select: 'high' or 'low'")
        return None


def get_model_interp(interp_mode="log", model="high", quantity="acc", x_units="T", y_units="dB", npts=1800, delta=0.01):
    # change delta to df of dT
    if y_units == "dB" or y_units == "SI":
        if model == "high":
            x, acc = NHNM(quantity=quantity, units=y_units)

            # if x_units == "f":
            #     x = to_Hz(x)
            #     delta = 1 / delta

            if interp_mode == "log":
                log_x = np.log10(x)
                delta = (max((log_x)) - min((log_x))) / (npts / 2)
                interp = interp1d(log_x, acc, kind='linear')
                log_x = np.arange(min(log_x), max(log_x), delta)

                return log_x, interp

            elif interp_mode == "linear":
                interp = interp1d(x, acc, kind='linear')
                x = np.arange(min(x), max(x), delta)

                return x, interp

        elif model == "low":
            x, acc = NLNM(quantity=quantity, units=y_units)

            # if x_units == "f":
            #     x = to_Hz(x)
            #     delta = 1 / delta

            if interp_mode == "log":
                log_x = np.log10(x)
                delta = (max((log_x)) - min((log_x))) / (npts / 2)
                interp = interp1d(log_x, acc, kind='linear')
                log_x = np.arange(min(log_x), max(log_x), delta)

                return log_x, interp

            elif interp_mode == "linear":
                interp = interp1d(x, acc, kind='linear')
                x = np.arange(min(x), max(x), delta)

                return x, interp

        else:
            print("Invalid model choice. Select: 'high' or 'low'")
            return None

    else:
        print("Invalid units. Choose dB or SI")
        return None


def get_power(model="high", quantity="acc", units="dB", delta=0.01):
    if units == "dB" or units == "SI":
        if model == "high":
            log_T, NHNM = get_model_interp(model="high",
                                           quantity=quantity,
                                           units=units, delta=delta)
            P = np.zeros(len(log_T))
            for i in range(len(log_T)):

                P[i] = NHNM(log_T[i])[()]
            return [log_T, P]

        elif model == "low":
            log_T, NLNM = get_model_interp(model="low",
                                           quantity=quantity,
                                           units=units, delta=delta)
            P = np.zeros(len(log_T))
            for i in range(len(log_T)):

                P[i] = NLNM(log_T[i])[()]
            return [log_T, P]

        else:
            print("Invalid model choice. Select: 'high' or 'low'")
            return None
    else:
        print("Invalid units. Choose dB or SI")
        return None


def get_octaves(minfreq ,maxfreq):
    n = (1 / np.log2((maxfreq / minfreq)))
    return n


def get_center_freq(minfreq, octaves):
    f0 = minfreq * (2**(octaves / 2))
    return f0


def get_rbw(n, minfreq, maxfreq, f0, octaves=True, freq=False):
    if octaves:
        rbw = ((2**octaves) - 1) / (2**(octaves / 2))
        return rbw
    elif freq:
        if not None in [minfreq, maxfreq, f0]:
            rbw = (maxfreq - minfreq) / f0
            return rbw
    else:
        print("Invalid option")
        return None


def get_rand_phase(N):
    phases = np.random.rand(N) * 2 * np.pi
    return phases


def get_uniform_rand_phase(phase_min, phase_max, N):
    phases = np.random.uniform(phase_min, phase_max, N)
    rad = np.arange(0, N)

    plt.close()
    plt.title("Function Check: Phase via Numpy Random-Uniform")
    # plt.plot(rad, phases)
    plt.scatter(rad, phases)
    plt.xlabel("Phase (Random on 0 - 2pi)")
    plt.ylabel("N")
    plt.show()

    return phases


def get_spectral_amplitude1(psd):
    if any(val < 0 for val in psd):
        print("\nNegative values, units likely in dB, attempting to convert ...\n")
        psd_SI = np.zeros(len(psd))
        for i in range(len(psd)):
            psd_SI[i] = 10**(psd[i] / 10)
        psd = psd_SI

    amp = np.zeros_like(psd)

    for i in range(len(psd)):
        amp[i] = np.sqrt(2 * psd[i])

    plt.close()
    plt.title("Function Check: get_spectral_amplitude() output")
    plt.plot(amp)
    plt.xlabel("N")
    plt.ylabel("Spectral Amplitude")
    plt.show()

    return amp


def rand_phase_PSD_signal(freq, psd, phase):
    N = len(psd)
    Z = np.zeros(N, dtype="complex")
    A = get_spectral_amplitude1(psd)
    img = np.sqrt(-1 + 0j)
    if len(freq) == len(psd) == len(phase):
        for i in range(N):
            Z[i] = A[i] * np.exp(img * phase[i])
        return Z
    else:
        print("\nInput arrays must be of equal size\n")
        return None

    plt.close()
    plt.plot(A)
    plt.title("Spectral Amplitude from PSD")
    plt.xlabel("Amplitude")
    plt.ylabel("Period")
    plt.show()

    plt.close()
    plt.plot(Z)
    plt.title("Random Phase Signal in Freq. Domain")
    plt.show()

def NHNM(quantity="acc", units="dB", P=None):
    NHNM_coeffs = pd.read_csv('./noise_models/NHNM-coeffs.txt')
    NHNM_SI = pd.read_csv('./noise_models/NHNM.csv')

    if units == "dB":
        if P is None:
            P = np.array(NHNM_coeffs["P"])
        A = np.array(NHNM_coeffs["A"])
        B = np.array(NHNM_coeffs["B"])

        if quantity == "acc":
            acc = A + B*(np.log10(P))
            return  [P, acc]

        elif quantity == "vel":
            p, acc = NHNM(quantity="acc")
            vel = acc + 20.0*np.log10(P/(2*np.pi))
            return [P, vel]

        elif quantity == "disp":
            p, vel = NHNM(quantity="vel")
            disp = vel + 20.0*np.log10(P**2/(2*np.pi)**2)
            return [P, disp]
        else:
            print("Unacceptable argument for quantity")
    elif units == "SI":
        if P is None:
            P = np.array(NHNM_SI["T [s]"])
        if quantity == "acc":
            acc = np.array(NHNM_SI["Pa [m2s-4/Hz]"])
            return  [P, acc]

        elif quantity == "vel":
            vel = np.array(NHNM_SI["Pv [m2s-2/Hz]"])
            return [P, vel]

        elif quantity == "disp":
            disp = np.array(NHNM_SI["Pd [m2/Hz]"])
            return [P, disp]
        else:
            print("Unacceptable argument for quantity")
    else:
        print("Invalid units. Choose dB or SI")
        return None

def NLNM(quantity="acc", units="dB", P=None):
    NLNM_coeffs = pd.read_csv('./noise_models/NLNM-coeffs.txt')
    NLNM_SI = pd.read_csv('./noise_models/NLNM.csv')

    if units == "dB":
        if P is None:
            P = np.array(NLNM_coeffs["P"])
        A = np.array(NLNM_coeffs["A"])
        B = np.array(NLNM_coeffs["B"])

        if quantity == "acc":
            acc = A + B * (np.log10(P))
            return [P, acc]

        elif quantity == "vel":
            p, acc = NLNM(quantity="acc")
            vel = acc + 20.0 * np.log10(P / (2 * np.pi))
            return [P, vel]

        elif quantity == "disp":
            p, vel = NLNM(quantity="vel")
            disp = vel + 20.0 * np.log10(P**2 / (2 * np.pi)**2)
            return [P, disp]
        else:
            print("Unacceptable argument for quantity")
            return None
    elif units == "SI":
        if P is None:
            P = np.array(NLNM_SI["T [s]"])
        if quantity == "acc":
            acc = np.array(NLNM_SI["Pa [m2s-4/Hz]"])
            return [P, acc]

        elif quantity == "vel":
            vel = np.array(NLNM_SI["Pv [m2s-2/Hz]"])
            return [P, vel]

        elif quantity == "disp":
            disp = np.array(NLNM_SI["Pd [m2/Hz]"])
            return [P, disp]
        else:
            print("Unacceptable argument for quantity")
            return None
    else:
        print("Invalid units. Choose dB or SI")
        return None



# %% Implement finer, regular sampling


#%% Plotting things up

#%% Acceleration
def plot_acc_NHNM(log=True, save=False, path='./'):
    P, spectra = NHNM(quantity="acc")
    fig = plt.figure()
    plt.plot(P, spectra, label="NHNM")
    plt.title("NHNM Station PSD after Peterson (1993)")
    plt.xlabel("Period (s)")
    plt.ylabel("Power Spectral Density (m/s^2)^2/Hz")
    ax = plt.gca()
    if log:
        ax.set_xscale('log')
    plt.legend(loc=1)

    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/NHNM_power_spectrum.png',dpi=500)

    return fig


def plot_acc_NLNM(log=True, save=False, path='./'):
    P, spectra = NLNM(quantity="acc")
    fig = plt.figure()
    plt.plot(P, spectra, 'orange',label="NLNM")
    plt.title("NLNM Station PSD after Peterson (1993)")
    plt.xlabel("Period (s)")
    plt.ylabel("Power Spectral Density (m/s^2)^2/Hz")
    ax = plt.gca()
    if log:
        ax.set_xscale('log')
    plt.legend(loc=1)

    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/NLNM_power_spectrum.png',dpi=500)

    return fig

#%% Velocity
def plot_vel_NHNM(log=True, save=False, path='./'):
    P, spectra = NHNM(quantity="vel")
    fig = plt.figure()
    plt.plot(P, spectra, label="NHNM")
    plt.title("NHNM Station Velocity/Hz after Peterson (1993)")
    plt.xlabel("Period (s)")
    plt.ylabel("Spectral Density (m/s)^2/Hz")
    ax = plt.gca()
    if log:
        ax.set_xscale('log')
    plt.legend(loc=1)

    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/NHNM_velocity_spectra.png',dpi=500)

    return fig


def plot_vel_NLNM(log=True, save=False, path='./'):
    P, spectra = NLNM(quantity="vel")
    fig = plt.figure()
    plt.plot(P, spectra, 'orange', label="NLNM")
    plt.title("NLNM Station Velocity/Hz after Peterson (1993)")
    plt.xlabel("Period (s)")
    plt.ylabel("Spectral Density (m/s)^2/Hz")
    ax = plt.gca()
    if log:
        ax.set_xscale('log')
    plt.legend(loc=1)

    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/NLNM_velocity_spectra.png',dpi=500)

    return fig

#%% Displacement
def plot_disp_NHNM(log=True, save=False, path='./'):
    P, spectra = NHNM(quantity="disp")
    fig = plt.figure()
    plt.plot(P, spectra, label="NHNM")
    plt.title("NHNM Station Displacement/Hz after Peterson (1993)")
    plt.xlabel("Period (s)")
    plt.ylabel("Spectral Density m^2/Hz")
    ax = plt.gca()
    if log:
        ax.set_xscale('log')
    plt.legend(loc=1)

    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/NHNM_displacement_spectra.png',dpi=500)

    return fig


def plot_disp_NLNM(log=True, save=False, path='./'):
    P, spectra = NLNM(quantity="disp")
    fig = plt.figure()
    plt.plot(P, spectra, 'orange', label="NLNM")
    plt.title("NLNM Station Displacement/Hz after Peterson (1993)")
    plt.xlabel("Period (s)")
    plt.ylabel("Spectral Density m^2/Hz")
    ax = plt.gca()
    if log:
        ax.set_xscale('log')
    plt.legend(loc=1)

    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/NLNM_displacement_spectra.png',dpi=500)

    return fig

#%% Plotting both models
def plot_acc_NHNM_and_NLNM(log=True, save=False, path='./'):
    [P_H, spectra_H] = NHNM(quantity="acc")
    [P_L, spectra_L] = NLNM(quantity="acc")

    fig = plt.figure()
    plt.plot(P_H, spectra_H, label="NHNM")
    plt.plot(P_L, spectra_L, label="NLNM")
    plt.title("NHNM/NLNM PSD after Peterson (1993)")
    plt.xlabel("Period (s)")
    plt.ylabel("Power Spectral Density (m/s^2)^2/Hz")
    ax = plt.gca()
    if log:
        ax.set_xscale('log')
    plt.legend(loc=1)

    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/NHNM_and_NLNM_power_spectra.png',dpi=500)

    return fig


def plot_vel_NHNM_and_NLNM(log=True, save=False, path='./'):
    [P_H, spectra_H] = NHNM(quantity="vel")
    [P_L, spectra_L] = NLNM(quantity="vel")

    fig = plt.figure()
    plt.plot(P_H, spectra_H, label="NHNM")
    plt.plot(P_L, spectra_L, label="NLNM")
    plt.title("NHNM/NLNM Velocity/Hz after Peterson (1993)")
    plt.xlabel("Period (s)")
    plt.ylabel("Spectral Density (m/s)^2/Hz")
    ax = plt.gca()
    if log:
        ax.set_xscale('log')
    plt.legend(loc=1)

    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/NHNM_and_NLNM_velocity_spectra.png',dpi=500)

    return fig


def plot_disp_NHNM_and_NLNM(log=True, save=False, path='./'):
    P_H, spectra_H = NHNM(quantity="disp")
    P_L, spectra_L = NLNM(quantity="disp")

    fig = plt.figure()
    plt.plot(P_H, spectra_H, label="NHNM")
    plt.plot(P_L, spectra_L, label="NLNM")
    plt.title("NHNM/NLNM Displacement/Hz after Peterson (1993)")
    plt.xlabel("Period (s)")
    plt.ylabel("Spectral Density m^2/Hz")
    ax = plt.gca()
    if log:
        ax.set_xscale('log')
    plt.legend(loc=1)

    if save:
        plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/NHNM_and_NLNM_displacement_spectra.png',dpi=500)

    return fig

#%%

#%% Without interpolating, check new SI plots

# T1, P1 = NHNM("acc", "SI")
# T2, P2 = NLNM("acc", "SI")
T1, P1 = NHNM("vel", "dB")
T2, P2 = NLNM("vel", "dB")

# # plt.loglog(T1,P1)
# # plt.loglog(T2,P2)
# plt.semilogx(T1,P1)
# plt.semilogx(T2,P2)
# plt.show()

# ####RESAMPLE TEST####
# N = 100000
# T1_resample = sig.resample(T1, N)
# P1_resample = sig.resample(P1, N)
# # P2_resample = sig.resample(P2, 256)
# plt.semilogx(P1_resample)
# # plt.plot(P2_resample)
# plt.show()


# #####################

#%% More Functions

# def sample_model():
#     return


# def check_noise_PSD(t, signal, norm=False, frequency=True):
#     ### Manually calcuating a PSD to check
#     spectrum = []
#     if norm:
#         spectrum = np.abs(np.fft.rfft(signal, norm="ortho"))
#     else:
#         spectrum = np.abs(np.fft.rfft(signal))

#     power = (spectrum**2)
#     spec_r_dB = 10 * np.log10(spec_r)
#     power_dB = 10 * np.log10(power)

#     if frequency:
#         f_r = np.fft.rfftfreq(len(signal))

#         return [f_r, power_dB]
#     else:
#         t_r = to_Period(f_r)

#         return [t_r, power_dB]


def get_time_array(T, P, signal, npts, delta=0.001):
    max_freq = max(T)
    if max_freq > 10.0:
        print("\nLarge f values, input likely in seconds not Hz, using Hz ...\n")
        max_freq = 1 / min(T)

    M = len(T)
    N = (M - 1) * 2
    # N = (M) * 2
    dt = 1 / (2 * max_freq)

    print(T)
    print("\nMax frequency or period: " + str(max_freq))
    print("\nN: " + str(N))
    print("\nM: " + str(M))
    print("\nValue of dt: " + str(dt))
    print("\nValue of delta: " + str(delta))


    # t = np.arange(0, N * dt, dt)
    # t = np.arange(0, (N - 1) * dt, dt)
    t = np.arange(0, N * delta, delta)

    return t


def assemble_signal(interp_mode="log", model="high", quantity="acc", x_units="T", y_units="SI", npts=1798, delta=0.02559485):
    [T, P] = get_model_interp(interp_mode=interp_mode, model=model, quantity=quantity, x_units=x_units, y_units=y_units, delta=delta)

    print("\nValues of T/Freq: ")
    print(T)
    print("\nLength of T/Freq is: " + str(len(T)))

    M = len(T)
    N = (M - 1) * 2

    psd = np.zeros(M)
    for i in range(M):
        log_P = P(T[i])
        psd[i] = log_P

    phase = get_uniform_rand_phase(0, (2 * np.pi), M)
    Z = rand_phase_PSD_signal(T, psd, phase)

    # signal = np.fft.irfft(Z, n=npts, norm="ortho")
    # signal = np.fft.irfft(Z, n=(npts / 2), norm="ortho")
    # signal = np.fft.irfft(Z, norm="ortho")
    signal = np.fft.irfft(Z)

    t = get_time_array(T, psd, signal, npts, delta)

    plt.close()
    plt.plot(t, signal)
    plt.title("Function Check: Ouput of 'assemble_signal()'")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.show()

    return [t, signal, Z]


def save_noise_csv(t, signal, filename = "", path = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/noise/data/"):
    datdf = {'time': t, 'amplitude': signal}
    df = pd.DataFrame(datdf)
    df.to_csv(path + filename + '.csv')


#%% Test amplitude functions

"""
For the synthetics ...

1798 points
The time step is: 0.02559485 seconds.
This is a sampling rate of 39.07035985754947 Hz
Total time is 46.019540299999996 seconds
"""

# [t, signal, Z] = assemble_signal(interp_mode="log", model="high", quantity="vel", y_units="dB", delta=0.02559485)
# [t, signal, Z] = assemble_signal(interp_mode="log", model="low", quantity="vel", y_units="dB", delta=0.02559485)

# [t, signal, Z] = assemble_signal(interp_mode="log", model="high", quantity="acc", y_units="dB", delta=0.02559485)
[t, signal, Z] = assemble_signal(interp_mode="log", model="low", quantity="acc", y_units="dB", delta=0.02559485)

# plt.plot(t, signal)
# plt.title("Plotting the assembled signal")
# # plt.xlim(0,1798)
# plt.show()

dat = [t, signal]
datdf = {'time': t, 'amplitude': signal}

### Auto calculating a PSD to check (Need to be careful about windows)
# freqs_test, psd_test = sig.welch(signal, fs = 39.07035985754947)
# freqs_test, psd_test = sig.welch(signal, nfft=len(signal))
# freqs_test, psd_test = sig.welch(signal, nperseg=(len(signal) / 2))
freqs_test, psd_test = sig.welch(signal)


plt.title("signal.welch loglog plot test")
plt.loglog(freqs_test, psd_test)
plt.show()
plt.title("signal.welch semilogy plot test")
plt.semilogy(freqs_test, psd_test)
plt.show()
plt.plot(freqs_test, psd_test)
plt.show()

peri_test = to_Period(freqs_test)
plt.loglog(peri_test, psd_test)
plt.title("Frequency to Period Conversion Test (loglog)")
plt.show()

### Manually calcuating a PSD to check (square before or after transform?!)
# spec_r = np.abs(np.fft.rfft(signal, norm="ortho"))
spec_r = np.abs(np.fft.rfft(signal))
# spec_r = np.abs(np.fft.rfft(signal, n=1798, norm="ortho"))
# spec_r = np.abs(np.fft.rfft(signal, n=899, norm="ortho"))
spec2_r = (spec_r**2)
spec_r_dB = 10 * np.log10(spec_r)
spec2_r_dB = 10 * np.log10(spec2_r)

### These frequency values range from
f_r = np.fft.rfftfreq(len(signal))
# f_r = np.fft.fftfreq(len(spec_r))
t_r = to_Period(f_r)


###
plt.loglog(f_r, spec_r)
plt.show()

plt.loglog(f_r, spec2_r)
plt.show()
###

plt.loglog(t_r, spec2_r)
plt.show()


plt.semilogx(t_r, spec2_r_dB)
plt.show()


plt.plot(spec2_r_dB)
plt.title("Just plotting Abs(Spectra)^2 in dB --> Correct Shape")
plt.show()

T = np.linspace(min(T2), max(T2), len(spec_r_dB))
plt.semilogx(T, spec2_r_dB)
plt.title("Wrong T + Abs(Spectra)^2 in dB --> Correct Shape")
plt.show()

plt.plot(T, spec2_r_dB)
plt.title("Correct-ish T (spacing) + Abs(Spectra)^2 in dB --> Correct Shape")
plt.show()

# %%

### OK only dB units work for now ... ###
N_series = 5
model = "low"
quantity = "vel"
units = "dB"

for i in range(N_series):
    [t, signal, Z] = assemble_signal(model=model, quantity=quantity, y_units=units, delta=0.02559485)

    # N = 1798
    # resamp = sig.resample(signal, N)
    # resamp_scaled = np.zeros(len(NHNM_resamp))
    # plt.plot(resamp)

    # filename = "noise-" + model + "_" + quantity + "_" + units + "_" + str(i)
    # save_noise_csv(t, signal, filename=filename)


