# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import signal as sig
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
        Hz_series[i] = 1 / frequency[i]

    return Hz_series


def get_coeffs(model="high"):
    if model == "high":
        NHNM_coeffs = pd.read_csv('./noise_models/NHNM-coeffs.txt')
        P = np.array(NHNM_coeffs["P"])
        A = np.array(NHNM_coeffs["A"])
        B = np.array(NHNM_coeffs["B"])

        return [P,A,B]

    elif model == "low":
        NLNM_coeffs = pd.read_csv('./noise_models/NLNM-coeffs.txt')
        P = np.array(NLNM_coeffs["P"])
        A = np.array(NLNM_coeffs["A"])
        B = np.array(NLNM_coeffs["B"])

        return [P,A,B]

    else:
        print("Invalid model choice. Select: 'high' or 'low'")
        return None


def get_model_interp(model="high", quantity="acc", units="dB", delta=0.01):
    #Need to account for variation in SI and dB here
    # bandwidth = []
    if units == "dB" or units == "SI":
        if model == "high":
            T, acc = NHNM(quantity="acc", units=units)
            log_T = np.log10(T)
            interp_NHNM = interp1d(log_T, acc, kind='linear')
            log_T = np.arange(min(log_T), max(log_T), delta)

            return log_T, interp_NHNM

        elif model == "low":
            T, acc = NLNM(quantity="acc", units=units)
            log_T = np.log10(T)
            interp_NLNM = interp1d(log_T, acc, kind='linear')
            log_T = np.arange(min(log_T), max(log_T), delta)

            return log_T, interp_NLNM

        else:
            print("Invalid model choice. Select: 'high' or 'low'")
            return None
    else:
        print("Invalid units. Choose dB or SI")
        return None


def get_power(model="high", quantity="acc", units="dB", delta=0.01):
    if units == "dB" or units == "SI":
        if model == "high":
            log_T, NHNM = get_model_interp(model="high", quantity=quantity, units=units, delta=delta)
            P = np.zeros(len(log_T))
            for i in range(len(log_T)):
                # print(log_T[i])
                # print(NHNM(log_T[i]))
                P[i] = NHNM(log_T[i])[()]
            return [log_T, P]

        elif model == "low":
            log_T, NLNM = get_model_interp(model="low", quantity=quantity, units=units, delta=delta)
            P = np.zeros(len(log_T))
            for i in range(len(log_T)):
                # print(log_T[i])
                # print(NLNM(log_T[i]))
                P[i] = NLNM(log_T[i])[()]
            return [log_T, P]

        else:
            print("Invalid model choice. Select: 'high' or 'low'")
            return None
    else:
        print("Invalid units. Choose dB or SI")
        return None


def get_octaves(minfreq ,maxfreq):
    n = (1/np.log2((maxfreq/minfreq)))
    return n


def get_center_freq(minfreq, octaves):
    f0 = minfreq * (2**(octaves/2))
    return f0


def get_rbw(n, minfreq, maxfreq, f0, octaves=True, freq=False):
    if octaves:
        rbw = ((2**octaves)-1)/(2**(octaves/2))
        return rbw
    elif freq:
        if not None in [minfreq, maxfreq, f0]:
            rbw = (maxfreq-minfreq)/f0
            return rbw
    else:
        print("Invalid option")
        return None


def get_rand_phase(N):
    phases = np.random.rand(N) * 2 * np.pi
    return phases


def get_uniform_rand_phase(N):
    phases = np.random.uniform(0,2*np.pi,N)
    return phases


def get_spectral_amplitude1(psd):
    if any(val < 0 for val in psd):
        print("Negative values, units likely in dB, attempting to convert ...")
        psd_SI = np.zeros(len(psd))
        for i in range(len(psd)):
            psd_SI[i] = 10**(psd[i]/10)
        psd = psd_SI
    amp = np.zeros_like(psd)
    for i in range(len(psd)):
        amp[i] = np.sqrt(2*psd[i])
    return amp

    ####
    # NEED to return frequencies here and determine freq interval, etc.
    ####


# def get_spectral_amplitude2(period):
#     return


# def get_record_amplitude(period):
#     return

def rand_phase_PSD_signal(freq, psd, phase):
    N = len(psd)
    Z = np.zeros(N, dtype="complex")
    A = get_spectral_amplitude1(psd)
    img = np.sqrt(-1+0j)
    if len(freq) == len(psd) == len(phase):
        for i in range(N):
            Z[i] = A[i] * np.exp(img * phase[i])
        return Z
    else:
        print("Input arrays must be of equal size")
        return None

    ####
    # NEED to return frequencies here and determine freq interval, etc.
    ####


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
            acc = A + B*(np.log10(P))
            return [P, acc]

        elif quantity == "vel":
            p, acc = NLNM(quantity="acc")
            vel = acc + 20.0*np.log10(P/(2*np.pi))
            return [P, vel]

        elif quantity == "disp":
            p, vel = NLNM(quantity="vel")
            disp = vel + 20.0*np.log10(P**2/(2*np.pi)**2)
            return [P, disp]
        else:
            print("Unacceptable argument for quantity")
            return None
    elif units == "SI":
        if P is None:
            P = np.array(NLNM_SI["T [s]"])
        if quantity == "acc":
            acc = np.array(NLNM_SI["Pa [m2s-4/Hz]"])
            return  [P, acc]

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

#%% Manual stuff

# plot_acc_NHNM(save=False)
# plot_acc_NLNM(save=False)
# plot_acc_NHNM_and_NLNM(save=False)

# # plot_vel_NHNM(save=True)
# # plot_vel_NLNM(save=True)
# plot_vel_NHNM_and_NLNM(save=False)

# # plot_disp_NHNM(save=True)
# # plot_disp_NLNM(save=True)
# plot_disp_NHNM_and_NLNM(save=False)

# # spectra_H = acc_NHNM(P_NHNM, A_NHNM, B_NHNM)
# # spectra_L = acc_NLNM(P_NLNM, A_NLNM, B_NLNM)


#%% Translating models to the time domain

# Start with acceleration
# p_H, acc_H = NHNM(quantity="acc")
# p_L, acc_L = NLNM(quantity="acc")

# Test finer scaled frequency sampling
# freq_H = np.arange(min(p_H),100000,0.1)
# p_H, acc_H = NHNM(quantity="acc", P=freq_H)

# Interpolate to a finer, even, spacing
# label="1d Linear Interp.Linear Period, not Log()"
# fig_name = "NHNM-1D-linear-Interpolation-NonLog(Period)"
# interp_NHNM = interp1d(p_H, acc_H, kind='linear')
# interp_NLNM = interp1d(p_L, acc_L, kind='linear')

# label="1d Nearest Interp."
# fig_name = "NHNM-1D-nearest-Interpolation"
# interp_NHNM = interp1d(p_H, acc_H, kind='nearest')
# interp_NLNM = interp1d(p_L, acc_L, kind='nearest')

# label="1d Zero Interp."
# fig_name = "NHNM-1D-nearest-Interpolation"
# interp_NHNM = interp1d(p_H, acc_H, kind='zero')
# interp_NLNM = interp1d(p_L, acc_L, kind='zero')

# label="1d Linear Spline Interp."
# fig_name = "NHNM-1D-slinear-Interpolation"
# interp_NHNM = interp1d(p_H, acc_H, kind='slinear')
# interp_NLNM = interp1d(p_L, acc_L, kind='slinear')

# label="1d Next Interp."
# fig_name = "NHNM-1D-next-Interpolation"
# interp_NHNM = interp1d(p_H, acc_H, kind='next')
# interp_NLNM = interp1d(p_L, acc_L, kind='next')

# label="Cubic Spline Interp."
# fig_name = "NHNM-cubic-Interpolation"
# interp_NHNM = interp1d(p_H, acc_H, kind='cubic')
# interp_NLNM = interp1d(p_L, acc_L, kind='cubic')

# freq_H = np.arange(min(p_H),100000,0.1)
# power = interp_NHNM(freq_H)


# plot_acc_NHNM(save=False)
# plt.semilogx(freq_H, power, label=label)
# # plt.plot(freq_H, power, label=label)
# plt.title("NHNM PSD and 1D Interpolation")
# plt.legend( prop={'size': 8})
# plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/'+ fig_name + '.png',dpi=500)

#%%

# Doing stuff in log space

# p_H, acc_H = NHNM(quantity="acc")
# p_L, acc_L = NLNM(quantity="acc")

# logp_H = np.log10(p_H)
# logp_L = np.log10(p_L)

# # Interpolate to a finer, even, spacing
# label="1d Linear Interp."
# fig_name = "NHNM-1D-linear-Interpolation-Log(Period)"
# interp_NHNM = interp1d(logp_H, acc_H, kind='linear')
# interp_NLNM = interp1d(logp_L, acc_L, kind='linear')

# logfreq_H = np.arange(min(logp_H),max(logp_H),0.1)
# logfreq_L = np.arange(min(logp_L),max(logp_L),0.1)

# logpow_H = interp_NHNM(logfreq_H)
# logpow_L = interp_NLNM(logfreq_L)


# # plot_acc_NHNM(save=False)
# # plt.semilogx(freq_H, power, label=label)
# plt.plot(logfreq_H, logpow_H, label=label)
# plt.plot(logp_H, acc_H, "--", label="Linear Interp done on Log10(Period)")
# plt.xlabel("Log(Period (s))")
# plt.ylabel("Power Spectral Density (m/s^2)^2/Hz")
# plt.title("NHNM PSD and 1D Interpolation")
# plt.legend()
# # plt.savefig(fname='/Users/gabriel/Documents/Research/USGS_Work/gmprocess/figs/models/'+ fig_name + '.png',dpi=500)


# p_H = 10**(logfreq_H)
# p_L = 10**(logfreq_L)



#%% Trying the interpolation functions


## Old Implementation ##
# log_P, interp_NHNM = get_model_interp(model="high")
# log_P, interp_NHNM = get_model_interp(model="high")

# pow_H = get_power(1, model="high")
# pow_L = get_power(1, model="low")

# test = get_power(10)

# P = np.arange(0.01,10000,0.1)

# pow_H = np.zeros(len(P))
# pow_L = np.zeros(len(P))

# for i in range(len(P)):
#     pow_H[i] = get_power(i, model = "high")
#     pow_L[i] = get_power(i, model = "low")


# Models in SI units

# [T, P] = get_power(model = "high")
# plt.plot(T,P)
# plt.show()

# [T, P] = get_power(model = "low")
# plt.plot(T,P)
# plt.show()

# [T, P] = get_power(model = "high", units="SI")
# plt.plot(T,P)
# plt.show()

# [T, P] = get_power(model = "low", units="SI")
# plt.plot(T,P)
# plt.show()

# [T, P] = get_power(model = "high", units="SI")
# plt.semilogx(T,P)
# plt.show()

# [T, P] = get_power(model = "low", units="SI")
# plt.semilogx(T,P)
# plt.show()

# [T, P] = get_power(model = "high", units="SI")
# plt.semilogy(T,P)
# plt.show()

# [T, P] = get_power(model = "low", units="SI")
# plt.semilogy(T,P)
# plt.show()


#%% Without interpolating, check new SI plots

T1, P1 = NHNM("acc", "SI")
T2, P2 = NLNM("acc", "SI")
# T1, P1 = NHNM("disp", "dB")
# T2, P2 = NLNM("disp", "dB")

plt.loglog(T1,P1)
plt.loglog(T2,P2)
# plt.semilogx(T1,P1)
# plt.semilogx(T2,P2)
plt.show()


#%% Test PSD recalc function
def sample_model(T, P, samp):
    return

def check_noise_PSD():
    return


def get_time():
    return

#%% Test amplitude functions

# [T, P] = get_power(model = "high")
##############################################################################
##############################################################################
# [T, P] = get_model_interp(model="high", quantity="acc", units="SI")
[T, P] = get_model_interp(model="high", quantity="vel", units="SI")
# [T, P] = get_model_interp(model="high", quantity="disp", units="SI")

# [T, P] = get_model_interp(model="high", quantity="acc", units="dB")
# [T, P] = get_model_interp(model="high", quantity="vel", units="dB")
# [T, P] = get_model_interp(model="high", quantity="disp", units="dB")
#----------------------------------------------------------------------------#
# [T, P] = get_model_interp(model="low", quantity="acc", units="SI")
# [T, P] = get_model_interp(model="low", quantity="vel", units="SI")
# [T, P] = get_model_interp(model="low", quantity="disp", units="SI")

# [T, P] = get_model_interp(model="low", quantity="acc", units="dB")
# [T, P] = get_model_interp(model="low", quantity="vel", units="dB")
# [T, P] = get_model_interp(model="low", quantity="disp", units="dB")
##############################################################################
##############################################################################


# print(min(10**(T)), max(10**(T)))
# print(min(T), max(T))

delta = (max(10**(T)) - min(10**(T)))/len(T)
# T_test = np.arange(min(T),max(T),delta)
T_test = np.arange(min(10**T),max(10**T),delta)
# T_test_ffreq = np.fft.fftfreq(len(T), delta) # no good

### Checking actual delta, spacing may be log as well?
# for i in range(len(T)):
#     if i == 0:
#         print(T[i])
#     else:
#         print(T[i] - T[i-1])
###
### Okay all the spacings are 0.01, but thats log space -1 to 6
# for i in range(len(T)):
#     if i == 0:
#         print(10**T[i])
#     else:
#         print(10**T[i] - 10**T[i-1])
###
### Now the samplings are constantly increasing ###

N = len(T)
psd = np.zeros(N)

## This only samples the interp model linearly in log space
#  This could be causing problems
for i in range(len(T)):
    log_P = P(T[i])
    psd[i] = log_P


phase = get_uniform_rand_phase(N)

Z = rand_phase_PSD_signal(T, psd, phase)

### T values are in log10 space already, amp is in SI, log shows it like NHNM
plt.semilogy(T,Z)
plt.show()

T_linear = to_linear(T)

plt.loglog(T_linear,Z)
plt.show()


### Getting the signal by using an iFFT (but which one?!)
# signal = np.fft.ifft(Z, len(Z))
# signal = np.fft.irfft(Z, len(Z))
# signal = np.fft.ifft(Z)


#----------------------#
signal = np.fft.irfft(Z)
#----------------------#

t = np.arange(0, len(signal))
plt.plot(t,signal)
plt.show()

dat = [t, signal]
datdf = {'time': t,'amplitude': signal}

### Auto calculating a PSD to check (Need to be careful about windows)
freqs_test, psd_test = sig.welch(signal.real)
plt.loglog(freqs_test,psd_test)
plt.show()
peri_test = np.zeros(len(freqs_test))
for i in range(len(freqs_test)):
    peri_test[i] = 1/freqs_test[i]
plt.loglog(peri_test,psd_test)
plt.show()

### Manually calcuating a PSD to check (square before or after transform?!)
signal_2 = np.zeros(len(signal))
for i in range(len(signal)):
    signal_2[i] = signal[i]*signal[i]

plt.plot(t,signal_2)
plt.show()

spec = np.abs(np.fft.fft(signal))
spec_r = np.abs(np.fft.rfft(signal))

spec2 = np.abs(np.fft.fft(signal_2))
spec2_r = np.abs(np.fft.rfft(signal_2))

## NOTES
# Using np.conj() and simply squaring yield the same, but there is still
# an imaginary component ...

for i in range(len(spec)):
    spec[i] = spec[i]*spec[i]

### These frequency values range from
f = np.fft.fftfreq(len(spec))
f_r = np.fft.rfftfreq(len(spec))
# f_r = np.fft.rfftfreq(len(spec_r))

f2 = np.fft.fftfreq(len(spec2))
f2_r = np.fft.rfftfreq(len(spec2_r))

## Converting from frequency to period ...?
t_r = np.zeros(len(f_r))
for i in range(len(t_r)):
    t_r[i] = (1 / f_r[i])

t_r = to_Period(f_r)


###
plt.plot(f_r, spec_r)
plt.show()

plt.semilogx(f_r, spec_r)
plt.show()

plt.loglog(f_r, spec_r)
plt.show()

plt.loglog(f_r, spec2_r)
plt.show()
###


## To Do

# Check real vs imaginary casting
#
# Add dB vs. SI checks to ensure signal construction correct
#
# Clean up code, make function for transforming back to time domain that
# formats time consistently and correctly
#
# Make these methods applicable to acc, vel, and disp
#
# Ensure consistent conventionw i.e Power = P, Period = T

plt.plot(t_r, spec_r)
plt.show()

plt.semilogx(t_r, spec_r)
plt.show()

# plt.loglog(t_r, spec2_r)
# plt.show()

plt.loglog(t_r, spec_r)
plt.show()

plt.loglog(t_r, spec2_r)
plt.show()


### Problem ###

# freq and period arrays are just the values incremented by 1, so max is
# the length of the array. Need to get actual period content (up to 10^5)

##########################################
# Arranging 10**T values provides the closest looking psd to NHNM/NLNM

# But ... it's backwards

##########################################

## Lengths are incorrect ##

# plt.loglog(T_test, spec_r)
# plt.show()

# plt.loglog(T_test, spec2_r)
# plt.show()

# plt.loglog(T_test, np.flip(spec_r))
# plt.show()

# plt.loglog(T_test, np.flip(spec2_r))
# plt.show()


# %%

# save_dir = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/noise/data/"

# filename = "rand_NHNM_1"
# filename = "rand_NLNM_1"

# path = save_dir + filename

df = pd.DataFrame(datdf)

df.to_csv('/Users/gabriel/Documents/Research/USGS_Work/gmprocess/scripts/noise/data/rand_NHNM_vel3.csv')
# df.to_csv('/data/rand_NHNM_1.csv')

