# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal as sig
from scipy.interpolate import interp1d

from obspy import read
from obspy import Trace
from obspy.core.stream import Stream
from obspy.signal import PPSD

# plt.style.use('ggplot')
plt.style.use('seaborn')

# %% Peterson (1993) - OFR 93-322 - New High/Low Noise Model (NHNM/NLNM)

def to_dB(signal):
    N = len(signal)
    dB_series = np.zeros(N)

    for i in range(N):
        dB_series[i] = 10 * np.log10(signal[i])

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


def get_model_interp(interp_mode="log", model="high", quantity="acc", x_units="T", y_units="dB", npts=1798, delta=0.01):
    # change delta to df of dT
    if y_units == "dB" or y_units == "SI":
        if model == "high":
            x, acc = NHNM(quantity=quantity, units=y_units)

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

# def get_power(model="high", quantity="acc", units="dB", delta=0.01):
#     if units == "dB" or units == "SI":
#         if model == "high":
#             log_T, NHNM = get_model_interp(model="high",
#                                            quantity=quantity,
#                                            units=units, delta=delta)
#             P = np.zeros(len(log_T))
#             for i in range(len(log_T)):
#                 P[i] = NHNM(log_T[i])[()]
#             return [log_T, P]

#         elif model == "low":
#             log_T, NLNM = get_model_interp(model="low",
#                                            quantity=quantity,
#                                            units=units, delta=delta)
#             P = np.zeros(len(log_T))
#             for i in range(len(log_T)):
#                 P[i] = NLNM(log_T[i])[()]
#             return [log_T, P]

#         else:
#             print("Invalid model choice. Select: 'high' or 'low'")
#             return None
#     else:
#         print("Invalid units. Choose dB or SI")
#         return None


def get_uniform_rand_phase(phase_min, phase_max, N, plot_checks=False):
    phases = np.random.uniform(phase_min, phase_max, N)
    rad = np.arange(0, N)

    if plot_checks:
        plt.close()
        plt.title("Function Check: Phase via Numpy Random-Uniform")
        plt.scatter(rad, phases)
        plt.xlabel("N")
        plt.ylabel("Phase (Random on 0 - 2pi)")
        plt.show()

        plt.close()
        plt.title("Function Check: Phase via Numpy Random-Uniform")
        plt.hist(phases, 20, label="Uniformly sampled mostly")
        plt.ylabel("Counts")
        plt.xlabel("Phase (Random on 0 - 2pi)")
        plt.legend()
        plt.show()

    return phases


def get_spectral_amplitude(psd, interp_mode):
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
    if interp_mode == 'log':
        plt.semilogy(amp)
    else:
        plt.loglog(amp)

    plt.title("Function Check: get_spectral_amplitude() output")
    plt.xlabel("Sample N from PSD (corresponds to Period)")
    plt.ylabel("Spectral Amplitude")
    plt.show()

    return amp


def rand_phase_PSD_signal(freq, psd, phase, interp_mode):
    N = len(psd)
    Z = np.zeros(N, dtype="complex")
    A = get_spectral_amplitude(psd, interp_mode)
    img = np.sqrt(-1 + 0j)
    if len(freq) == len(psd) == len(phase):
        for i in range(N):
            Z[i] = A[i] * np.exp(img * phase[i])
        return Z
    else:
        print("\nInput arrays must be of equal size\n")
        return None

def NHNM(quantity="acc", units="dB", P=None):
    NHNM_coeffs = pd.read_csv('./noise_models/NHNM-coeffs.txt')
    NHNM_SI = pd.read_csv('./noise_models/NHNM.csv')

    if units == "dB":
        if P is None:
            P = np.array(NHNM_coeffs["P"])
        A = np.array(NHNM_coeffs["A"])
        B = np.array(NHNM_coeffs["B"])

        if quantity == "acc":
            acc = A + (B * (np.log10(P)))
            return [P, acc]

        elif quantity == "vel":
            p, acc = NHNM(quantity="acc")
            vel = acc + (20.0 * np.log10(P / (2 * np.pi)))
            return [P, vel]

        elif quantity == "disp":
            p, vel = NHNM(quantity="vel")
            disp = acc + (20.0 * np.log10(P**2 / (2 * np.pi)**2))
            return [P, disp]
        else:
            print("Unacceptable argument for quantity")
    elif units == "SI":
        if P is None:
            P = np.array(NHNM_SI["T [s]"])
        if quantity == "acc":
            acc = np.array(NHNM_SI["Pa [m2s-4/Hz]"])
            return [P, acc]

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
            disp = acc + 20.0 * np.log10(P**2 / (2 * np.pi)**2)
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

#%% More Functions

def assemble_signal(interp_mode="log", model="high",
                    quantity="acc", x_units="T", y_units="dB",
                    npts=1798, delta=0.02559485, plot_checks=False):

    M = 2 * npts

    [T, P] = get_model_interp(interp_mode=interp_mode,
                              model=model, quantity=quantity,
                              x_units=x_units, y_units=y_units,
                              npts=M, delta=delta)

    amplitude_spectrum = P(T)
    amplitude_spectrum = 10**(amplitude_spectrum / 10)

    phase = get_uniform_rand_phase(0, (2 * np.pi), int(M / 2))

    amplitude_r = amplitude_spectrum * np.cos(phase)
    amplitude_i = amplitude_spectrum * np.sin(phase)
    ifft_complex2 = amplitude_r + amplitude_i * 1j

    signal = np.fft.ifft(ifft_complex2)
    signal_r = np.real(signal)
    signal_i = np.imag(signal)

    # Build time array
    tmax = (npts * delta)
    t = np.arange(0, tmax, delta)

    if plot_checks:
        if model == "high":
            label = "NHNM"

        elif model == "low":
            label = "NLNM"

        plt.plot(t, signal_r, label=quantity)
        plt.title(label + ": Reconstructed Time Series (Real)")
        plt.xticks(np.arange(0, max(t), 5))
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

        plt.scatter(signal_r, signal_i, label="Discrete points in Complex Signal")
        plt.title("Polar Plot of Reconstructed Time Series in Complex Plane")
        plt.xlabel("Real Signal")
        plt.ylabel("Imag. Signal")
        plt.legend()
        plt.show()

        # Informative, but takes a little while to plot
        # plt.figure()
        # for p in signal:
        #     plt.polar([0, np.angle(p)], [0, np.abs(p)], marker='o')
        # plt.title("Phase of Reconstructed Time Series in Complex Plane")
        # plt.xlabel("Real", labelpad=10)
        # plt.ylabel("Imaginary", labelpad=35)
        # plt.tight_layout()
        # plt.show()

        # plt.title("Histogram of Signal")
        # plt.hist(signal, bins=20, label=model)
        # plt.legend()
        # plt.show()

    return [t, signal_r]

def generate_noise_boore(model='NHNM', npts=1798, dt = 0.02559485):
    # Get appropriate model to use in the Boore (2003) method
    model_coeffs = pd.DataFrame()

    if model == 'NHNM':
        print("\nGrabbing NHNM model coeffiecients ... \n")
        model_coeffs = pd.read_csv('./noise_models/NHNM-coeffs.txt')
    elif model == 'NLNM':
        print("\nGrabbing NLNM model coeffiecients ... \n")
        model_coeffs = pd.read_csv('./noise_models/NLNM-coeffs.txt')
    else:
        print("Invalid model selection ... Exiting ...")

    A = np.array(model_coeffs["A"])
    B = np.array(model_coeffs["B"])

    # Calculate the model values from coefficients
    model_period = np.array(model_coeffs["P"])
    model_acc = A + B * (np.log10(model_period))

    # Create function for interpolation of these values
    interp = interp1d(model_period, model_acc,
                      kind="linear", fill_value='extrapolate')

    ##################################
    ##### Temporary Plot Check  ######
    ##################################
    plt.figure()
    plt.semilogx(model_period, model_acc, label=model)
    plt.title("Check Intermediate Step: Pre Interpolation Noise Model ")
    plt.ylabel('PSD (vel) 10log_10([m/s]^2/[Hz])', fontweight="bold")
    plt.xlabel('Period (S)', fontweight="bold")
    plt.legend()
    plt.show()
    ##################################

    # Determine which points we want from the FT
    FT_npts = npts
    stop = (FT_npts - 1) * dt

    # Create a stochastic time series (can mod std. dev. of this)
    x = np.random.normal(0, 1, FT_npts)
    t = np.linspace(0, stop, FT_npts)

    # Construct array of +freq indeces we need
    fft_idx = np.arange(int(FT_npts / 2), FT_npts)

    # Take the FFT and shift zero freq term
    sig_fft = (np.fft.fft(x) * dt)
    freq = (np.fft.fftfreq(len(sig_fft), dt))

    sig_fft = np.fft.fftshift(sig_fft)
    freq = np.fft.fftshift(freq)

    ##################################
    ##### Temporary Plot Check  ######
    ##################################
    plt.figure()
    plt.loglog(freq, np.abs(sig_fft), label="Stochastic Signal")
    plt.title("Check Intermediate Step Signal: FFT ")
    plt.ylabel('Spectral Amplitude', fontweight="bold")
    plt.xlabel('Frequency (Hz)', fontweight="bold")
    plt.legend()
    plt.show()
    ##################################

    # Set zero freq term nonzero to avoid discontinuity
    ind_zero = np.where(freq == 0.0)
    freq_temp = freq.copy()
    freq_temp[ind_zero] = 0.01   # changed from 1

    # Take only positive freq terms and convert to period
    freq_abs = np.abs(freq_temp)
    period_freq_abs = (1 / freq_abs)
    period_freq_abs[ind_zero] = 0.01  # changed from 1

    # Interpolate the model values and get it out of dB ()
    ''' The defining equation for decibels is

        A = 10*log10(P2/P1)     (dB)

        where P1 is the power being measured, and P1 is
        the reference to which P2 is being compared.

        To convert from decibel measure back to power ratio:

        P2/P1 = 10^(A/10) '''

    # Get the noise model and convert from dB to (presumably) (m/s^2) / Hz
    NM_acc_i_dB = interp(period_freq_abs)

    ##################################
    ##### Temporary Plot Check  ######
    ##################################
    plt.figure()
    plt.semilogx(period_freq_abs, NM_acc_i_dB, label=model)
    plt.title("Check Intermediate Step Noise Model: Interpolated but *Before* Conversion from dB")
    plt.ylabel('PSD (vel) 10log_10([m/s]^2/[Hz])', fontweight="bold")
    plt.xlabel('Frequency (Hz)', fontweight="bold")
    plt.legend()
    plt.show()
    ##################################

    NM_acc_i = 10**(NM_acc_i_dB / 10)               # Scale wrong?
    # NM_acc_i = 10**(np.sqrt(NM_acc_i_dB) / 10)    # Try sqrt?

    ##################################
    ##### Temporary Plot Check  ######
    ##################################
    plt.figure()
    plt.semilogx(period_freq_abs, NM_acc_i, label=model)
    plt.title("Check Intermediate Step Noise Model: *After* Conversion from dB")
    plt.ylabel('Spectral Amplitude', fontweight="bold")
    plt.xlabel('Frequency (Hz)', fontweight="bold")
    plt.legend()
    plt.show()
    ##################################

    # Get the mean square average
    msa = np.sqrt(np.mean(np.abs(sig_fft)**2))

    # Normalize the FFT of the signal by this
    sig_fft_norm = sig_fft / msa

    ##################################
    ##### Temporary Plot Check  ######
    ##################################
    plt.figure()
    plt.loglog(period_freq_abs, np.abs(sig_fft_norm), label="Stochastic Signal")
    plt.title("Check Intermediate Step Signal: MSA Normalized")
    plt.ylabel('Spectral Amplitude', fontweight="bold")
    plt.xlabel('Frequency (Hz)', fontweight="bold")
    plt.legend()
    plt.show()
    ##################################

    # Multiply noise model and normalized FT
    mod_fft = sig_fft_norm * NM_acc_i

    # Transfer back to the time domain
    sim_ifft = np.fft.ifft(mod_fft)

    # Take the real component only
    sim = np.real(sim_ifft)
    sim_im = np.imag(sim_ifft)

    # Check the FFT and phase of the signal
    sim_fft = np.fft.fft(sim)
    sim_fft_abs = np.abs(sim_fft)
    sim_fft_phase = np.angle(sim_fft)

    ##################################
    ##### Temporary Plot Check  ######
    ##################################
    # plt.figure()
    # plt.loglog(period_freq_abs, sim_fft_abs, label=model + " Modulated Stoch. Signal: FFT")
    # plt.title("Check Intermediate Step Signal: FFT of Noise Modulated Stoch. Signal")
    # plt.ylabel('Spectral Amplitude', fontweight="bold")
    # plt.xlabel('Frequency (Hz)', fontweight="bold")
    # plt.legend()
    # plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].set_title('FFT of Noise Modulated Stoch. Signal')
    axs[0].loglog(period_freq_abs, sim_fft_abs, label=model + " Noise Modulated Stoch. Signal: FFT")
    # axs[1].scatter(sim, sim_fft_phase, label=model +" Noise Modulated Signal Phase")
    axs[1].scatter(sim, sim_im, label=model +" Noise Modulated Signal Phase")
    axs[1].set_title('Checking Phase As Well')
    axs[0].legend() ; axs[1].legend()
    axs[0].set_ylabel('Spectral Amplitude', fontweight="bold")
    axs[0].set_xlabel('Frequency (Hz)', fontweight="bold")
    axs[1].set_ylabel('Imag. Signal', fontweight="bold")
    axs[1].set_xlabel('Real Signal', fontweight="bold")
    plt.show()
    ##################################

    # Return +freq portions of the arrays
    return [t, sim, freq[fft_idx], NM_acc_i[fft_idx],
            sim_fft_abs[fft_idx], sim_fft_phase[fft_idx]]



def save_noise_csv(t, signal, filename="",
                   path="/Users/gabriel/Documents/Research/USGS_Work/"
                   "gmprocess_scratchpaper/scripts/noise/data/csv/"):

    datdf = {'time': t, 'amplitude': signal}
    df = pd.DataFrame(datdf)
    df.to_csv(path + filename + '.csv')


#%%################################################
################ Scratch work below! ##############
###################################################

#%% Write noise to csv and then to miniseed

### OK only dB units work for now ... ###
"""
For the synthetics ...

1798 points
The time step is: 0.02559485 seconds.
This is a sampling rate of 39.07035985754947 Hz
Total time is 46.019540299999996 seconds
"""

N_series = 1
# model = "low"
# model = "high"

quantity = "vel"

units = "dB"
# units = "SI"  # No idea what's up with this

models = ["high", "low"]

save_csv = True
# Creating noise by constructing a random phase signal in freq. domain
for mod in models:

    for i in range(N_series):

        [t, signal] = assemble_signal(model=mod, quantity=quantity,
                                      y_units=units, npts=1798,
                                      delta=0.02559485, plot_checks=True)

        if mod == "high":
            filename = "NHNM/noise-test-" + mod + "_" + quantity + \
                                                "_" + units + "_ID-" + str(i)
            if save_csv:
                save_noise_csv(t, signal, filename=filename)

        if mod == "low":
            filename = "NLNM/noise-test-" + mod + "_" + quantity +  \
                                                "_" + units + "_ID-" + str(i)
            if save_csv:
                save_noise_csv(t, signal, filename=filename)

# Creating noise with the NNM modulated stochastic noise like Boore (2003)

""" Using "NHNM or NLNM might be an invalid model choice, double check " """
for mod in models:

    for i in range(N_series):

        if mod == "high":
            [t, x, freq1, NHNM,
             sim_abs, sim_phase] = generate_noise_boore(model='NHNM')

            filename = "NHNM/boore-noise-test-" + mod + "_" + quantity + \
                                                "_" + units + "_ID-" + str(i)
            plt.plot(NHNM)
            plt.title("Plot NHNM from generate_noise_boore() output")
            plt.show()

            if save_csv:
                save_noise_csv(t, x, filename=filename)

        if mod == "low":
            [t, x, freq1, NLNM,
             sim_abs, sim_phase] = generate_noise_boore(model='NLNM')

            filename = "NLNM/boore-noise-test-" + mod + "_" + quantity +  \
                                                "_" + units + "_ID-" + str(i)

            plt.plot(NLNM)
            plt.title("Plot NLNM from generate_noise_boore() output")
            plt.show()

            if save_csv:
                save_noise_csv(t, x, filename=filename)

#%% Read noise in as ObsPy trace

san_check = "/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/figs/sanity_checks/noise_generation/"
boore_noise = '/Users/gabriel/Documents/Research/USGS_Work/gmprocess_scratchpaper/scripts/noise/data/miniseed/boore/'

from noise_generation import read_noise, write_noise
# Comparing low and high noise models

# Read in noise using ObsPy, save as miniseed files
[NHNM_st, NLNM_st] = read_noise(quantity="boore") #quantity is really just a keyword search

#%% Plot Check
NHNM_st.plot()
NLNM_st.plot() # NLNM gets messed up prior to this, read_noise()?

#%% Write Noise
write_noise(NHNM_st, "NHNM/boore-noise-test-NHNM-vel", path=boore_noise)
write_noise(NLNM_st, "NLNM/boore-noise-test-NLNM-vel", path=boore_noise)


#%% Add some noise to the synthetics
from noise_generation import add_modeled_noise
from synthetics import read_synthetic_streams

synths = read_synthetic_streams()

# Just get Trace object
NHNM_noise = NHNM_st[0]
NLNM_noise = NLNM_st[0]

# Scale noise by 10
NHNM_noise_x10 = NHNM_noise.copy()
NHNM_noise_x10.data = NHNM_noise_x10.data * 10

NLNM_noise_x10 = NLNM_noise.copy()
NLNM_noise_x10.data = NLNM_noise_x10.data * 10

# Scale noise by 100
NHNM_noise_x100 = NHNM_noise.copy()
NHNM_noise_x100.data = NHNM_noise_x100.data * 100

NLNM_noise_x100 = NLNM_noise.copy()
NLNM_noise_x100.data = NLNM_noise_x100.data * 100

# Scale noise by 1000
NHNM_noise_x1k = NHNM_noise.copy()
NHNM_noise_x1k.data = NHNM_noise_x1k.data * 1000

NLNM_noise_x1k = NLNM_noise.copy()
NLNM_noise_x1k.data = NLNM_noise_x1k.data * 1000

# Scale noise by 10K
NHNM_noise_x10k = NHNM_noise.copy()
NHNM_noise_x10k.data = NHNM_noise_x10k.data * 10000

NLNM_noise_x10k = NLNM_noise.copy()
NLNM_noise_x10k.data = NLNM_noise_x10k.data * 10000

# Add the noise to the synthetics
## 'signal' may be stream of traces, 'noise' should be single trace
NHNM_st_noisy = add_modeled_noise(synths[0], NHNM_noise)
NLNM_st_noisy = add_modeled_noise(synths[0], NLNM_noise)

NHNM_st_noisy_x10 = add_modeled_noise(synths[0], NHNM_noise_x10)
NLNM_st_noisy_x10 = add_modeled_noise(synths[0], NLNM_noise_x10)

NHNM_st_noisy_x100 = add_modeled_noise(synths[0], NHNM_noise_x100)
NLNM_st_noisy_x100 = add_modeled_noise(synths[0], NLNM_noise_x100)

NHNM_st_noisy_x1k = add_modeled_noise(synths[0], NHNM_noise_x1k)
NLNM_st_noisy_x1k = add_modeled_noise(synths[0], NLNM_noise_x1k)

NHNM_st_noisy_x10k = add_modeled_noise(synths[0], NHNM_noise_x10k)
NLNM_st_noisy_x10k = add_modeled_noise(synths[0], NLNM_noise_x10k)

#%% Quick plots with ObsPy
NHNM_noise.plot()
NLNM_noise.plot() #This is fucked up for some reason

# The above should be the same as this ...
# NHNM_st[0].plot()
# NLNM_st[0].plot()   #Confirmed

NHNM_st_noisy_x10[0].plot()
NLNM_st_noisy_x10[0].plot()

NHNM_st_noisy_x100[0].plot()
NLNM_st_noisy_x100[0].plot()

NHNM_st_noisy_x10k[0].plot()
NLNM_st_noisy_x10k[0].plot()


#%% Double check NHNM and NLNM are different

plt.figure()
plt.title("Check that NHNM and NLNM are different")
plt.plot(NHNM_noise.times(), NHNM_noise, label="NHNM Noise")
plt.plot(NLNM_noise.times(), NLNM_noise, '--', label="NLNM Noise")
plt.legend()
# plt.show()

# plt.savefig(san_check + "Boore Noise  - Overlay.png",dpi=600)


fig, axs = plt.subplots(2, 1, figsize=(12, 6))
# # fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
# ax1.plot(NHNM_noise.times(), NHNM_noise, label="NHNM Noise")
# ax2.plot(NLNM_noise.times(), NLNM_noise, '--', label="NLNM Noise")
axs[0].plot(NHNM_noise.times(), NHNM_noise, label="NHNM Noise")
axs[1].plot(NLNM_noise.times(), NLNM_noise, label="NLNM Noise")
axs[0].legend() ; axs[1].legend()
# plt.show()

# plt.savefig(san_check + "Boore Noise  - Side by Side - Nope NLNM is Just Wrong.png",dpi=600)


#%% Can't see differences with ObsPy plot, maybe a problem? Try manual plots

# Double check the scaling is working appropriately

plot_log = False

if not plot_log:
    #NHNM
    plt.figure()
    plt.title("Noisy Synthetic Signal:"
              "Scaling NHNM", fontsize=13, fontweight="bold")

    plt.plot(NHNM_st_noisy_x10k[0].times(), NHNM_st_noisy_x10k[0], label="Noisy signal - NHNM_x10K")
    # plt.plot(NHNM_st_noisy_x1k[0].times(), NHNM_st_noisy_x1k[0], label="Noisy signal - NHNM_x1K")
    # plt.plot(NHNM_st_noisy_x100[0].times(), NHNM_st_noisy_x100[0], label="Noisy signal - NHNM_x100")
    # plt.plot(NHNM_st_noisy_x10[0].times(), NHNM_st_noisy_x10[0], label="Noisy signal - NHNM_x10")
    # plt.plot(NHNM_st[0].times(), NHNM_st[0], label="Noisy signal - No Scaling")
    plt.plot(synths[0][0].times(), synths[0][0], label="Original")

    plt.xticks(np.arange(0, max(NHNM_st[0].times()), 5))
    plt.xlabel("Time (s)", fontweight="bold")
    plt.ylabel("Velocity (m/s)", fontweight="bold")
    # plt.xlim(12, 14)
    plt.legend()
    # plt.show()

    # plt.savefig(san_check + "Synthetics  - Original and 10K Scaled NHNM Noise Added.png",dpi=600)
    # plt.savefig(san_check + "Synthetics  - Noisy Signal, No Scaling - NHNM.png",dpi=600)

    #NLNM
    plt.figure()
    plt.title("Noisy Synthetic Signal:"
              "Scaling NLNM", fontsize=13, fontweight="bold")

    plt.plot(NLNM_st_noisy_x10k[0].times(), NLNM_st_noisy_x10k[0], label="Noisy signal - NLNM_x10K")
    # plt.plot(NLNM_st_noisy_x1k[0].times(), NLNM_st_noisy_x1k[0], label="Noisy signal - NLNM_x1K")
    # plt.plot(NLNM_st_noisy_x100[0].times(), NLNM_st_noisy_x100[0], label="Noisy signal - NLNM_x100")
    # plt.plot(NLNM_st_noisy_x10[0].times(), NLNM_st_noisy_x10[0], label="Noisy signal - NLNM_x10")
    # plt.plot(NLNM_st[0].times(), NLNM_st[0], label="Noisy signal - No Scaling")
    plt.plot(synths[0][0].times(), synths[0][0], label="Original")

    plt.xticks(np.arange(0, max(NLNM_st[0].times()), 5))
    plt.xlabel("Time (s)", fontweight="bold")
    plt.ylabel("Velocity (m/s)", fontweight="bold")
    # plt.xlim(12, 14)
    plt.legend()
    # plt.show()

    # plt.savefig(san_check + "Synthetics  - Original and 10K Scaled NLNM Noise Added.png",dpi=600)
    # plt.savefig(san_check + "Synthetics  - Noisy Signal, No Scaling - NLNM.png",dpi=600)

else:
    #NHNM
    plt.figure()
    plt.title("Noisy Synthetic Signal:"
              "Scaling NHNM", fontsize=13, fontweight="bold")

    plt.semilogy(NHNM_st_noisy_x10k[0].times(), abs(NHNM_st_noisy_x10k[0].data), label="Noisy signal - NHNM_x10K")
    plt.semilogy(NHNM_st_noisy_x1k[0].times(), abs(NHNM_st_noisy_x1k[0].data), label="Noisy signal - NHNM_x1K")
    plt.semilogy(NHNM_st_noisy_x100[0].times(), abs(NHNM_st_noisy_x100[0].data), label="Noisy signal - NHNM_x100")
    plt.semilogy(NHNM_st_noisy_x10[0].times(),abs( NHNM_st_noisy_x10[0].data), label="Noisy signal - NHNM_x10")
    plt.semilogy(NHNM_st[0].times(), abs(NHNM_st[0].data), label="Noisy signal - No Scaling")
    plt.semilogy(synths[0][0].times(), abs(synths[0][0].data), label="Original")

    plt.xticks(np.arange(0, max(NHNM_st[0].times()), 5))
    plt.xlabel("Time (s)", fontweight="bold")
    plt.ylabel("Velocity (m/s)", fontweight="bold")
    # plt.xlim(12, 14)
    plt.legend()
    # plt.show()

    # plt.savefig(san_check + "Synthetics  - Log Plotting to Check Scaling - NHNM.png",dpi=600)

    #NLNM
    plt.figure()
    plt.title("Noisy Synthetic Signal:"
              "Scaling NLNM", fontsize=13, fontweight="bold")

    plt.semilogy(NLNM_st_noisy_x10k[0].times(), abs(NLNM_st_noisy_x10k[0].data), label="Noisy signal - NLNM_x10K")
    plt.semilogy(NLNM_st_noisy_x1k[0].times(), abs(NLNM_st_noisy_x1k[0].data), label="Noisy signal - NLNM_x1K")
    plt.semilogy(NLNM_st_noisy_x100[0].times(), abs(NLNM_st_noisy_x100[0].data), label="Noisy signal - NLNM_x100")
    plt.semilogy(NLNM_st_noisy_x10[0].times(), abs(NLNM_st_noisy_x10[0].data), label="Noisy signal - NLNM_x10")
    plt.semilogy(NLNM_st[0].times(), abs(NLNM_st[0].data), label="Noisy signal - No Scaling")
    plt.semilogy(synths[0][0].times(), abs(synths[0][0].data), label="Original")

    plt.xticks(np.arange(0, max(NLNM_st[0].times()), 5))
    plt.xlabel("Time (s)", fontweight="bold")
    plt.ylabel("Velocity (m/s)", fontweight="bold")
    # plt.xlim(12, 14)
    plt.legend()
    # plt.show()

    # plt.savefig(san_check + "Synthetics  - Log Plotting to Check Scaling - NLNM.png",dpi=600)

#%% Make some plots to check things out

plot_a_palooza=False
if plot_a_palooza:
    def plot_waveform_overlay(NHNM_st, NLNM_st, reverse_zorder=False):

        print("\nPlotting Waveform Overlay ...\n")

        plt.figure()
        plt.title("Noise Series Constructed From NHNM and NLNM:"
                  "Anomalous Amplitudes", fontsize=13, fontweight="bold")

        if reverse_zorder:
            plt.plot(NLNM_st[0].times(), NLNM_st[0], label="NLNM")
            plt.plot(NHNM_st[0].times(), NHNM_st[0], label="NHNM")
        else:
            plt.plot(NHNM_st[0].times(), NHNM_st[0], label="NHNM")
            plt.plot(NLNM_st[0].times(), NLNM_st[0], label="NLNM")

        plt.xticks(np.arange(0, max(NHNM_st[0].times()), 5))
        plt.xlabel("Time (s)", fontweight="bold")
        plt.ylabel("Velocity (m/s)", fontweight="bold")
        plt.legend()
        plt.show()


    plot_waveform_overlay(NHNM_st, NLNM_st, reverse_zorder=True)


    def plot_fft_overlay(NHNM_st, NLNM_st, reverse_zorder=False):
        print("\nPlotting Periodogram Overlay ...\n")
        delta_h = NHNM_st[0].stats.delta
        delta_l = NLNM_st[0].stats.delta

        nfft_h = len(NHNM_st[0]) * 2
        nfft_l = len(NLNM_st[0]) * 2

        fft_h = np.abs(np.fft.fftshift(np.fft.fft(NHNM_st[0], nfft_h) * (delta_h)))
        freq_h = np.fft.fftfreq(nfft_h, delta_h)
        freq_h = np.fft.fftshift(freq_h)

        fft_l = np.abs(np.fft.fftshift(np.fft.fft(NLNM_st[0], nfft_l) * (delta_l)))
        freq_l = np.fft.fftfreq(nfft_l, delta_l)
        freq_l = np.fft.fftshift(freq_l)

        if reverse_zorder:
            plt.loglog(freq_l, fft_l, label="NLNM")
            plt.loglog(freq_h, fft_h, label="NHNM")

        else:
            plt.loglog(freq_h, fft_h, label="NHNM")
            plt.loglog(freq_l, fft_l, label="NLNM")

        plt.title("FFT of Noise Time Series Generated from NHNM/NLNM", fontsize=16, fontweight="bold")
        plt.ylabel('Spectral Amplitude', fontweight="bold")
        plt.xlabel('Frequency (Hz)', fontweight="bold")
        # plt.xlim(0, 20)
        plt.legend()

        plt.savefig(san_check + "NHNM-NLNM FFTs.png")
        # plt.show()


    plot_fft_overlay(NHNM_st, NLNM_st, reverse_zorder=True)


    def plot_spectrograms(NHNM_st, NLNM_st):
        print("\nPlotting Spectrograms ...\n")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        fig.suptitle("Spectrograms for NHNM and NLNM Time Series",
                     fontsize=16, fontweight="bold")

        NHNM_st[0].spectrogram(axes=axs[0])
        # NHNM_st[0].spectrogram(axes=axs[0], dbscale=True)
        # NHNM_st[0].spectrogram(axes=axs[0], log=True)
        axs[0].set_title("Time Series Constructed from NHNM", fontweight="bold")
        axs[0].set_xlabel("Time (s)", fontweight="bold")
        axs[0].set_ylabel("Frequency (Hz)", fontweight="bold")
        axs[0].set_xticks(np.arange(0, max(NHNM_st[0].times()), 5))

        NLNM_st[0].spectrogram(axes=axs[1])
        # NLNM_st[0].spectrogram(axes=axs[1], dbscale=True)
        # NLNM_st[0].spectrogram(axes=axs[1], log=True)
        axs[1].set_title("Time Series Constructed from NLNM", fontweight="bold")
        axs[1].set_xlabel("Time (s)", fontweight="bold")
        axs[1].set_ylabel("Frequency (Hz)", fontweight="bold")
        axs[1].set_xticks(np.arange(0, max(NHNM_st[0].times()), 5))

        # plt.savefig(san_check + "NHNM-NLNM Spectrograms.png")
        plt.show()


    plot_spectrograms(NHNM_st, NLNM_st)


    def plot_ppsd_welch(NHNM_st, NLNM_st):

        plt.figure()
        # segmt_lens = [32, 64, 128, 256, 512]
        segmt_lens = [32, 256]
        segmt_lens.reverse()

        for nperseg in segmt_lens:
            fs_h = NHNM_st[0].stats.sampling_rate
            fs_l = NLNM_st[0].stats.sampling_rate

            freq_wh, Pxx_wh = sig.welch(NHNM_st[0], fs_h, nperseg=nperseg)
            freq_wl, Pxx_wl = sig.welch(NLNM_st[0], fs_l, nperseg=nperseg)
            label_h = "NHNM, nperseg: " + str(nperseg)
            label_l = "NLNM, nperseg: " + str(nperseg)
            plt.semilogy(freq_wh, Pxx_wh, label=label_h)
            plt.semilogy(freq_wl, Pxx_wl, label=label_l)

        # plt.ylim([0.5e-3, 1])
        plt.title("Estimated PSD for NHNM/NLNM Time Series with Welch's Method",
                  fontsize=13, fontweight="bold")
        plt.xlabel('frequency [Hz]', fontweight="bold")
        plt.ylabel('PSD [V**2/Hz]', fontweight="bold")
        plt.legend()
        plt.savefig(san_check + "PSD via Welch's Method - NHNM-NLNM - 32 and 256 nperseg.png")


    plot_ppsd_welch(NHNM_st, NLNM_st)



