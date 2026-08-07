"""
plotting.py

Provides plotting utilities for analysing and visualising
results from the OFC earthquake simulation.

The functions in this module generate plots of

    • earthquake size series,
    • earthquake size distributions,
    • power-law fits to earthquake size distributions,
    • lattice and domain synchronisation histories,
    • averaged synchronisation and exponential desynchronisation fits,
    • desynchronisation load as a function of the energy-transfer
      parameter alpha.

The plotting functions use analysis routines defined in
``analysis.py`` and do not modify the simulation state.

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
from analysis import (
    earthquake_distribution,
    fit_power_law,
    average_domain_synchronisation,
    fit_exponential
)

#------------------------------------------------
# FINISH PLOT FUNCTION
#------------------------------------------------
def finish_plot(title, xlabel, ylabel, filename=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", alpha=0.3)

    if filename is not None:
        plt.savefig(filename, dpi=300)

    plt.show(block=True)

# -------------------------------------------------
# PLOT EARTHQUAKE SIZE TIME SERIES
# -------------------------------------------------
def plot_earthquake_size_series(sizes):
    """
    Plots the earthquake size progression.

    Earthquake sizes are normalised by the maximum recorded
    earthquake size to aid visualisation.

    Parameters
    ----------
    sizes : array-like
        Series of earthquake sizes recorded during the
        simulation.
    """
    earthquake_number = np.arange(len(sizes))
    scaled_size = sizes / np.max(sizes)

    # PLOT

    plt.figure()
    plt.bar(earthquake_number, scaled_size)

    finish_plot(
    "Earthquake size series",
    "Earthquake number",
    "Scaled earthquake size"
    )

# -------------------------------------------------
# PLOT EARTHQUAKE SIZE DISTRIBUTION
# -------------------------------------------------
def plot_sizes(sizes):
    """
    Plots the earthquake size probability distribution on
    logarithmic axes with Poisson uncertainty.

    The probability of each earthquake size is calculated from
    the recorded simulation data.

    Parameters
    ----------
    sizes : array-like
        Recorded earthquake sizes.
    """
    # CALCULATE EARTHQUAKE SIZE DISTRIBUTION

    unique_sizes, probability, poisson_errors = earthquake_distribution(sizes)

    # PLOT

    # Plot earthquake probability against size with Poisson uncertainty on logarithmic axes.
    plt.figure()
    plt.errorbar(unique_sizes, probability, yerr = poisson_errors, fmt = ".")
    plt.xscale("log")
    plt.yscale("log")

    finish_plot(
    "Earthquake size probability distribution",
    "Earthquake size",
    "Probability"
    )

# -------------------------------------------------
# PLOT POWER LAW FIT TO EARTHQUAKE SIZE DISTRIBUTION
# -------------------------------------------------
def plot_power_law_fit(sizes):
    """
    Plots the earthquake size probability distribution together
    with a fitted power law on logarithmic axes.

    The power-law exponent is obtained by linear regression in
    log-log space using ``fit_power_law``.

    Parameters
    ----------
    sizes : array-like
        Recorded earthquake sizes.
    """
    # CALCULATE EARTHQUAKE SIZE DISTRIBUTION

    unique_sizes, probability, _ = earthquake_distribution(sizes)

    # REMOVE ZERO PROBABILITY VALUES FOR FITTING

    mask = probability > 0
    sizes_fit = unique_sizes[mask]
    probability_fit = probability[mask]

    # LINEAR REGRESSION

    gradient, intercept, r_squared = fit_power_law(sizes)
    tau = gradient

    # Generate fitted line
    fitted_probability = np.exp(intercept) * sizes_fit**(-tau)

    # PLOT

    plt.figure()

    # DATA
    plt.scatter(
        sizes_fit,
        probability_fit,
        label="Simulation data"
    )

    # FIT
    plt.plot(
        sizes_fit,
        fitted_probability,
        label=fr"Power law fit: $\tau={tau:.3f}$"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.legend()

    finish_plot(
    "Earthquake size probability distribution",
    "Earthquake size",
    "Probability"
    )

    print(f"Power law exponent τ = {tau:.3f}")
    print(f"R² = {r_squared:.3f}")

# -------------------------------------------------
# PLOT LATTICE SYNCHRONISATION
# -------------------------------------------------
def plot_lattice_synchronisation(lattice, params):
    """
    Plots the synchronisation history of the entire lattice
    as a function of cumulative load.

    Only every 1000th recorded point is plotted to reduce the
    number of points displayed.

    Parameters
    ----------
    lattice : TrackedDomain
        TrackedDomain object representing the entire lattice.
    """
    # TAKE SUBSET OF DATA

    # Only plot a subset of values to reduce processing demand
    PLOT_SAMPLE = params.plot_sample
    synchronisation = lattice.synchronisation_history[::PLOT_SAMPLE]
    cumulative_load = lattice.load_history[::PLOT_SAMPLE]

    # PLOT

    plt.figure()
    plt.scatter(cumulative_load, synchronisation)

    finish_plot(
    "Synchronisation of entire lattice vs cumulative load",
    "Cumulative load",
    "Synchronisation order parameter R"
    )

# -------------------------------------------------
# PLOT DOMAIN SYNCHRONISATION
# -------------------------------------------------
def plot_domain_synchronisation(domains, params):
    """
    Plots the synchronisation history of each tracked domain
    as a function of cumulative load.

    Only every 1000th recorded point is plotted to reduce the
    number of points displayed.

    Parameters
    ----------
    domains : list[TrackedDomain]
        Collection of tracked domains to plot.
    """
    PLOT_SAMPLE = params.plot_sample

    for domain in domains:

        # TAKE SUBSET OF DATA

        # Only plot a subset of values to reduce processing demand
        synchronisation = domain.synchronisation_history[::PLOT_SAMPLE]
        cumulative_load = domain.load_history[::PLOT_SAMPLE]

        # PLOT

        plt.figure()
        plt.scatter(cumulative_load, synchronisation)

        finish_plot(
        f"Synchronisation vs cumulative load, domain size = {domain.size}",
        "Cumulative load",
        "Synchronisation order parameter R"
        )

# -------------------------------------------------
# PLOT AVERAGE SYNCHRONISATION
# -------------------------------------------------
def plot_average_synchronisation(domains, params):
    """
    Plots the average synchronisation of the tracked domains as a
    function of cumulative load, together with an exponential
    desynchronisation fit.

    The synchronisation histories of the individual domains are
    interpolated onto a common cumulative-load axis before being
    averaged. The initial transient increase in synchronisation is
    excluded when fitting the subsequent desynchronisation regime.

    Parameters
    ----------
    domains : list[TrackedDomain]
        Collection of tracked domains whose synchronisation histories
        are averaged.

    params : Parameters
        Simulation parameters used to determine the interpolation
        settings and energy-transfer parameter alpha.
    """
    # AVERAGE SYNCHRONISATION OVER DOMAINS

    cumulative_load, mean_synchronisation = average_domain_synchronisation(domains, params)

    # FIT EXPONENTIAL

    load_fit, fitted_synchronisation, popt, tau_err = fit_exponential(mean_synchronisation, cumulative_load)
    tau = popt[1]

    # PLOT DATA AND FIT

    plt.figure()
    plt.scatter(cumulative_load, mean_synchronisation, label = 'Data')
    plt.plot(load_fit, fitted_synchronisation, label = f'Exponential fit τ = {tau:.3f} +/- {tau_err:.3f}')
    plt.legend()

    finish_plot(
            f"Average synchronisation vs cumulative load, α = {params.alpha:.3f}",
            "Cumulative load",
            f"Synchronisation order parameter R, averaged over {len(domains)} domains",
            )

# -------------------------------------------------
# PLOT DESYNCHRONISATION LOAD
# -------------------------------------------------
def plot_desynchronisation_load(alphas, taus, tau_errs, params):
    """
    Plots the characteristic desynchronisation load as a function
    of the energy-transfer parameter alpha.

    Parameters
    ----------
    alphas : array-like
        Values of the energy-transfer parameter alpha.

    taus : array-like
        Mean fitted desynchronisation loads for each alpha.

    tau_errs : array-like
        Uncertainties in the mean desynchronisation loads.

    params : Parameters
        Simulation parameters.
    """
    # PLOT DATA WITH ERROR BARS

    plt.figure()
    plt.errorbar(alphas, taus, yerr = tau_errs, fmt = '.', label = 'Data')
    plt.legend()

    filename = f"tau_vs_alpha_{params.alpha_min}_{params.alpha_max}_{params.num_alphas}_{params.runs_per_alpha}.png"

    finish_plot(
            f"Characteristic desynchronisation load τ vs energy transfer constant α",
            "Energy transfer constant α",
            f"Characteristic desynchronisation load τ",
            filename
            )