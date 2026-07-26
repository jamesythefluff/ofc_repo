"""
plotting.py

Provides plotting utilities for the OFC earthquake
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

Author: Anne-Marie
Date: July 2026
"""
# -------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# IMPORT MODULES
# -------------------------------------------------
from analysis import *

# Useful plotting function
def finish_plot(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", alpha=0.3)
    plt.show(block=True)

# -------------------------------------------------
# EARTHQUAKE SIZE ANALYSIS
# -------------------------------------------------
def plot_time_series(sizes):
    """
    Plots the earthquake size as a function of simulation time.

    Earthquake sizes are normalised by the maximum recorded
    earthquake size to aid visualisation.

    Parameters
    ----------
    sizes : array-like
        Time series of earthquake sizes recorded during the
        simulation.
    """
    time = np.arange(len(sizes))
    scaled_sizes = sizes / np.max(sizes)

    plt.figure()
    plt.bar(time, scaled_sizes)

    finish_plot(
    "Earthquake Size Time Series",
    "Time",
    "Scaled Earthquake Size"
    )

def plot_sizes(sizes):
    """
    Plots the earthquake size probability distribution on
    logarithmic axes with Poisson uncertainty.

    The probability of each earthquake size is calculated
    from the recorded simulation data, and the uncertainty
    is estimated assuming Poisson counting statistics.

    Parameters
    ----------
    sizes : array-like
        Time series of earthquake sizes.
    """
    unique_sizes, probability, poisson_errors = earthquake_distribution(sizes)

    # Earthquake probability vs size plot log-log plot with Poisson errorbars
    plt.figure()
    plt.errorbar(unique_sizes, probability, yerr = poisson_errors, fmt = ".")
    plt.xscale("log")
    plt.yscale("log")

    finish_plot(
    "Earthquake Size Distribution",
    "Earthquake Size",
    "Probability"
    )

def plot_power_law_fit(sizes):
    """
    Plots the earthquake size probability distribution on log-log axes
    with a fitted power law.

    Assumes a relationship of the form

        P(s) ∝ s^(-tau)

    where tau is extracted from a linear regression in log-log space.

    Parameters
    ----------
    sizes : array-like
        Time series of earthquake sizes recorded during the simulation.
    """

    # Calculate probability distribution
    unique_sizes, probability, poisson_errors = earthquake_distribution(sizes)

    # Remove any zero probabilities
    mask = probability > 0

    sizes_fit = unique_sizes[mask]
    probability_fit = probability[mask]

    # Linear regression
    gradient, intercept, rvalue = fit_power_law(sizes)

    tau = gradient

    # Generate fitted line
    fitted_probability = np.exp(intercept) * sizes_fit**(-tau)

    # Plot
    plt.figure()

    plt.scatter(
        sizes_fit,
        probability_fit,
        label="Simulation data"
    )

    plt.plot(
        sizes_fit,
        fitted_probability,
        label=fr"Power law fit: $\tau={tau:.3f}$"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.legend()

    finish_plot(
    "Earthquake Size Distribution",
    "Earthquake Size",
    "Probability"
    )

    print(f"Power law exponent τ = {tau:.3f}")
    print(f"R² = {rvalue**2:.3f}")

# -------------------------------------------------
# SYNCHRONISATION ANALYSIS
# -------------------------------------------------

# -------------------------------------------------
# Plot lattice synchronisation
# -------------------------------------------------
def plot_lattice_R(lattice):
    """
    Plots the synchronisation history of the entire lattice.

    Parameters
    ----------
    lattice : TrackedDomain
        Object representing the whole lattice. Its recorded
        synchronisation history is plotted.
    """
    # Only plot a subset of values to reduce processing demand
    sample = 1000
    R_lattice = lattice.history[::sample]
    time = np.arange(len(R_lattice))

    plt.figure()
    plt.scatter(time, R_lattice)

    finish_plot(
    "Synchronisation of entire lattice over time",
    "Time",
    "Normalised Magnitude of R"
    )

# -------------------------------------------------
# Plot domain synchronisation
# -------------------------------------------------
def plot_domain_R(domains):
    """
    Plots the synchronisation histories of all tracked domains.

    Parameters
    ----------
    domains : list[TrackedDomain]
        Collection of tracked domains.
    """
    for domain in domains:
        # Only plot a subset of values to reduce processing demand
        R_domain = domain.history[::1000]
        time = np.arange(len(R_domain))
        
        plt.figure()
        plt.scatter(time, R_domain)
        
        finish_plot(
        f"Synchronisation of domain size {domain.size} over time",
        "Time",
        "Normalised Magnitude of R"
        )

def plot_desynchronisation(domains, params):
    
    mean_R, time = average_domain_synchronicity(domains, params)

    time_fit, mean_R_fit, popt = fit_exponential(mean_R, time)
    A, tau, C = popt

    a = params.a

    plt.figure()
    plt.scatter(time, mean_R, label = 'Data')
    plt.plot(time_fit, mean_R_fit, label = f'Exponential fit τ = {tau}')
    plt.legend()

    finish_plot(
            f"Average Synchronisation Time Series α = {a}",
            "Time",
            f"Synchronisation Averaged over {len(domains)} Domains"
            )