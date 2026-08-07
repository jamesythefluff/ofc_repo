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
def exponential(load, A, tau, C):
    """
    Exponential relaxation model

        R(load) = C + A exp(-load / tau)

    where

    A
        Initial synchronisation amplitude.

    tau
        Characteristic accumulated load over which 
        desynchronisation occurs.

    C
        Long-time residual synchronisation.
    """
    return C + A * np.exp(-load / tau)

# -------------------------------------------------
# CALCULATE EARTHQUAKE SIZE DISTRIBUTION
# -------------------------------------------------
def earthquake_distribution(sizes):
    """
    Calculates the earthquake size probability distribution.

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
        Poisson uncertainty on the estimated probabilities.

    Notes
    -----
    Earthquake probabilities are estimated by normalising the
    histogram of observed earthquake sizes. Poisson counting
    statistics are used to estimate the uncertainty.
    """
    # SIZE HISTOGRAM

    unique_sizes, counts = np.unique(sizes, return_counts=True)

    # NORMALISE COUNTS

    probability = counts / len(sizes)

    # POISSON UNCERTAINTY

    poisson_errors = np.sqrt(counts / len(sizes))

    return unique_sizes, probability, poisson_errors

# -------------------------------------------------
# FIT LINEAR REGRESSION TO EXTRACT POWER LAW EXPONENT
# -------------------------------------------------
def fit_power_law(sizes):
    """
    Fits a power law to the earthquake size distribution.

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
    
    Notes
    -----
    Assumes the earthquake size distribution follows

        P(s) ∝ s^(-tau)

    and estimates tau using ordinary least-squares linear
    regression in logarithmic coordinates.
    """
    # UNPACK DATA

    unique_sizes, probability, poisson_errors = earthquake_distribution(sizes)

    # TAKE LOGS

    # Remove zero values before taking logs
    mask = probability > 0

    # Convert to log space
    log_sizes = np.log(unique_sizes[mask])
    log_probability = np.log(probability[mask])

    # LINEAR REGRESSION

    fit = linregress(log_sizes, log_probability)

    gradient = -fit.slope
    intercept = fit.intercept
    r_squared = fit.rvalue**2

    return gradient, intercept, r_squared

# -------------------------------------------------
# AVERAGE DOMAIN SYNCHRONISATION
# -------------------------------------------------
def average_domain_synchronisation(domains, params):
    """
    Computes the average synchronisation history of tracked domains.

    Individual domains evolve over different total loads, so each
    synchronisation history is interpolated onto a common load axis
    before averaging.

    Parameters
    ----------
    domains : list[Domain]
        Tracked domains.

    params : Parameters
        Simulation parameters.

    Returns
    -------
    mean_synchronisation : np.ndarray
        Average synchronisation as a function of load.

    common_load : np.ndarray
        Common interpolated load axis.
    """
    # PARAMETERS

    NUM_INTERPOLATION_POINTS = params.num_interpolation_points
    TAIL_CUTOFF = params.tail_cutoff

    # INTERPOLATE

    total_loads = [
        domain.load_history[-1] - domain.load_history[0] 
        for domain in domains
    ]
    max_load = np.max(total_loads)
    common_load = np.linspace(0, max_load, NUM_INTERPOLATION_POINTS)

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

    # AVERAGE

    all_synchronisations = np.vstack(interpolated_domains)
    mean_synchronisation = np.nanmean(all_synchronisations, axis=0)

    # DISCARD UNDERSAMPLED TAIL

    # Near the maximum load only a small number of domains remain, so the average becomes noisy. 
    mean_synchronisation = mean_synchronisation[:-TAIL_CUTOFF]
    common_load = common_load[:-TAIL_CUTOFF]

    return common_load, mean_synchronisation

# -------------------------------------------------
# EXPONENTIAL FIT TO SYNCHRONISATION CURVE
# -------------------------------------------------
def fit_exponential(mean_synchronisation, accumulated_load):
    """
    Fits an exponential decay to the average synchronisation curve.

    The initial transient increase in synchronisation is excluded by
    locating the maximum of the curve. The remaining data are fitted
    with the model

        R(load) = C + A exp(-load / tau)

    where tau is the characteristic accumulated load over which
    desynchronisation occurs.

    Initial parameter estimates are obtained from the data using the
    amplitude, asymptotic value, and the point at which the decay
    reaches approximately 1/e of its initial amplitude.

    Parameters
    ----------
    mean_synchronisation : np.ndarray
        Average synchronisation order parameter.

    accumulated_load : np.ndarray
        Accumulated load corresponding to ``mean_synchronisation``.

    Returns
    -------
    accumulated_load_fit : np.ndarray
        Portion of the accumulated load axis used for the fit.

    fitted_synchronisation : np.ndarray
        Best-fit exponential evaluated on ``accumulated_load_fit``.

    popt : np.ndarray
        Optimal fit parameters ``(A, tau, C)``.

    tau_err : float
        One-standard-deviation uncertainty on the fitted
        desynchronisation load tau.
    """
    # REMOVE INITIAL TRANSIENT

    # Ignore the initial transient increase in synchronisation and fit
    # only the subsequent desynchronisation regime.
    peak = np.argmax(mean_synchronisation)
    accumulated_load = accumulated_load[peak:]
    mean_synchronisation_fit = mean_synchronisation[peak:]

    # INITIAL PARAMETER ESTIMATES

    A0 = np.max(mean_synchronisation_fit) - np.min(mean_synchronisation_fit)
    C0 = np.min(mean_synchronisation_fit)

    # Estimate tau as the load where the decay reaches 1/e of its initial amplitude. 
    target = C0 + A0/np.e
    idx = np.argmin(np.abs(mean_synchronisation_fit - target))
    tau0 = accumulated_load[idx]

    # FIT EXPONENTIAL

    popt, pcov = curve_fit(
        exponential, 
        accumulated_load, 
        mean_synchronisation_fit, 
        p0=[A0, tau0, C0]
    )

    A, tau, C = popt
    tau_err = np.sqrt(pcov[1,1])
    
    print(f'τ = {tau:.3f} +/- {tau_err:.3f}')

    fitted_synchronisation = exponential(accumulated_load, A, tau, C)

    return accumulated_load, fitted_synchronisation, popt, tau_err