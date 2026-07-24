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