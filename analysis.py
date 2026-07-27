"""
analysis.py

Provides analysis and plotting utilities for the OFC earthquake
simulation.

The functions in this module visualise quantities recorded during
the simulation, including

    • earthquake size time series,
    • synchronisation histories,
    • earthquake size distributions.

The earthquake size distribution is plotted on logarithmic axes
to investigate whether the OFC model exhibits power-law behaviour,
a characteristic feature of self-organised critical systems.

Synchronisation histories are plotted to study the evolution of
local and global order parameters during the simulation.

These functions are intended to be called after the simulation has
completed and do not modify the simulation state.

Author: Anne-Marie
Date: July 2026
"""
# -------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

# -------------------------------------------------
# EXPONENTIAL DECAY MODEL
# -------------------------------------------------
def exponential(t, A, tau, C):
    """
    Exponential decay model.

    R(t) = C + A exp(-t/tau)
    """
    return C + A * np.exp(-t / tau)

# -------------------------------------------------
# CALCULATE EARTHQUAKE SIZE DISTRIBUTION
# -------------------------------------------------
def earthquake_distribution(sizes):
    """
    Calculates the probability distribution of earthquake sizes.

    Parameters
    ----------
    sizes : array-like
        Recorded earthquake sizes.

    Returns
    -------
    unique_sizes : np.ndarray
        Unique earthquake sizes.

    probability : np.ndarray
        Probability of each earthquake size.

    poisson_errors : np.ndarray
        Poisson statistical uncertainty in calculates probabilities.
    """

    unique_sizes, counts = np.unique(sizes, return_counts=True)
    probability = counts / len(sizes)
    poisson_errors = np.sqrt(counts / len(sizes))

    return unique_sizes, probability, poisson_errors

# -------------------------------------------------
# FIT LINEAR REGRESSION TO EXTRACT POWER LAW EXPONENT
# -------------------------------------------------
def fit_power_law(sizes):
    """
    Fits a power law to the earthquake size distribution.

    Assumes

        P(s) ∝ s^(-tau)

    and performs linear regression in log-log space.

    Parameters
    ----------
    sizes : array-like
        Recorded earthquake sizes.

    Returns
    -------
    gradient : float
        Fitted gradient = -tau where tau is the power law exponent.

    intercept : float
        Fitted y-intercept.

    r_squared : float
        Goodness of fit.
    """

    unique_sizes, probability, poisson_errors = earthquake_distribution(sizes)

    # Remove zero values before taking logs
    mask = probability > 0

    # Convert to log space
    log_sizes = np.log(unique_sizes[mask])
    log_probability = np.log(probability[mask])

    fit = linregress(log_sizes, log_probability)

    gradient = -fit.slope
    intercept = fit.intercept
    r_squared = fit.rvalue**2

    return gradient, intercept, r_squared

# -------------------------------------------------
# NORMALISED AVERAGE SYNCHRONICITY TIME SERIES
# -------------------------------------------------
def average_domain_synchronicity(domains, params):
    SYNCHRONISATION_HISTORY_LENGTH = params.synchronisation_history_length
    total_synchronisation_history = np.zeros(SYNCHRONISATION_HISTORY_LENGTH)

    # INTERPOLATION
    total_load_arr = []
    for domain in domains:
        total_load = domain.load_history[-1] - domain.load_history[0]
        total_load_arr.append(total_load)
    max_load = np.max(total_load_arr)
    common_load = np.linspace(0, max_load, 500)

    interpolated_domains = []

    for domain in domains:
        history = domain.history
        relative_load = (domain.load_history - domain.load_history[0])
        R_interp = np.interp(
            common_load,
            relative_load,
            domain.history,
            left=np.nan,
            right=np.nan
        )
        interpolated_domains.append(R_interp)

    all_R = np.vstack(interpolated_domains)
    mean_R = np.nanmean(all_R, axis=0)

    # CUT OFF LAST 50 UNDERSAMPLED DATA POINTS
    mean_R = mean_R[:-50]
    common_load = common_load[:-50]

    return mean_R, common_load

# -------------------------------------------------
# EXPONENTIAL FIT TO DESYNCHRONICITY CURVE
# -------------------------------------------------
def fit_exponential(mean_R, time):
    # Exclude initial peak for fitting
    peak = np.argmax(mean_R)
    time_fit = time[peak:]
    mean_R_fit = mean_R[peak:]

    # Initial guesses
    A0 = np.max(mean_R_fit) - np.min(mean_R_fit)
    tau0 = len(time_fit) / 3
    C0 = np.min(mean_R_fit)

    popt, pcov = curve_fit(
        exponential,
        time_fit,
        mean_R_fit,
        p0=[A0, tau0, C0]
    )

    A, tau, C = popt
    print(f'A = {A}, τ = {tau}, C = {C}')

    fitted_R = exponential(time_fit, A, tau, C)

    return time_fit, fitted_R, popt