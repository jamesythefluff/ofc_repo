"""
experiments.py

Contains high-level numerical experiments performed using the
OFC model.

Each experiment coordinates one or more simulations, analyses
the resulting data, and produces figures suitable for comparison
and interpretation. Internal helper functions provide common
functionality shared between experiments, such as estimating
desynchronisation loads and saving results.

Author: Anne-Marie
Date: July 2026
"""

# -------------------------------------
# Import libraries
# -------------------------------------
import numpy as np

# -------------------------------------
# Import modules
# -------------------------------------
from simulation import run_simulation
from analysis import average_domain_synchronisation, fit_exponential
from plotting import plot_average_synchronisation, plot_desynchronisation_load

# -------------------------------------
# Internal helper functions
# -------------------------------------

def _estimate_desynchronisation_load(params):
    """
    Estimates the characteristic desynchronisation load for a
    single parameter set.

    The simulation is repeated several times using independent
    random initial conditions. For each run, the average domain
    synchronisation is calculated and fitted with an exponential
    decay to extract the characteristic desynchronisation load.
    The mean desynchronisation load and its standard error are
    returned.

    Parameters
    ----------
    params : Parameters
        Object containing the model, simulation and experiment
        parameters.

    Returns
    -------
    tau_mean : float
        Mean desynchronisation load obtained from all simulation
        runs.

    tau_sem : float
        Standard error of the mean desynchronisation load.
    """
    sweep_type = params.sweep_type

    tau_values = []

    # -------------------------------------
    # Loop over independent simulation runs
    # -------------------------------------
    
    for run in range(getattr(params, f"runs_per_{sweep_type}")):

        _, domains, _ = run_simulation(params)

        cumulative_load, mean_synchronisation = average_domain_synchronisation(domains, params)
        
        _, _, popt, _ = fit_exponential(cumulative_load, mean_synchronisation)
        tau = popt[1]

        # Only store reasonable fits
        if 0 < tau < params.tau_upper_bound:
            tau_values.append(tau)  

    # Plot final run for each parameter value to check fit quality
    plot_average_synchronisation(domains, params)

    # -------------------------------------
    # Average over all runs to obtain mean and standard error of the mean
    # -------------------------------------
    if len(tau_values) > 0:
        tau_mean = np.mean(tau_values)
        tau_sem = np.std(tau_values, ddof=1) / np.sqrt(len(tau_values))
    else:
        print(f"Warning: No valid desynchronisation loads found for {sweep_type} = {getattr(params, sweep_type)}.")

    return tau_mean, tau_sem


def _save_results(tau_means, tau_sems, params):
    """
    Saves the results of any parameter sweep to a NumPy archive.

    The filename is generated automatically from the lower and
    upper bounds of the parameter sweep.

    Parameters
    ----------
    tau_means : array_like
        Mean desynchronisation load corresponding to each parameter value.

    tau_sems : array_like
        Standard error of the mean desynchronisation load for
        each parameter value.

    params : Parameters
        Parameter object storing sweep_values and used to generate the filename.
    """
    sweep_type = params.sweep_type
    sweep_values = params.sweep_values

    directory = f"data/{sweep_type}_sweep"
    filename = (
        f"{sweep_type}_sweep_"
        f"{np.min(sweep_values):.3f}_"
        f"{np.max(sweep_values):.3f}.npz"
    )
    filepath = f"{directory}/{filename}"

    np.savez(filepath,
        sweep_values=sweep_values,
        tau_means=tau_means,
        tau_sems=tau_sems,
    )

# -------------------------------------
# Public experiments
# -------------------------------------

def alpha_sweep(params):
    """
    Investigates how the characteristic desynchronisation load
    depends on the stress transfer parameter α.

    The neighbour coupling constant is varied to produce a range
    of α values. For each α, the characteristic desynchronisation
    load is estimated from multiple independent simulation runs.
    The resulting mean desynchronisation load and its standard
    error are saved and plotted as a function of α.

    Parameters
    ----------
    params : Parameters
        Object containing the model, simulation and sweep
        parameters.

    Returns
    -------
    alphas : ndarray
        Stress transfer parameter values.

    tau_means : list[float]
        Mean desynchronisation load for each α.

    tau_sems : list[float]
        Standard error of the mean desynchronisation load.
    """
    # -------------------------------------
    # Initialise sweep
    # -------------------------------------

    params.sweep_type = "alpha"

    alphas = params.sweep_values

    tau_means = []
    tau_sems = []

    # -------------------------------------
    # Loop over alpha values
    # -------------------------------------
    for alpha in alphas:

        # Set the simulation alpha
        params.alpha = alpha

        tau_mean, tau_sem = _estimate_desynchronisation_load(params)

        tau_means.append(tau_mean)
        tau_sems.append(tau_sem)

        print(
            f"Completed {len(tau_means)}/{len(alphas)} "
            f"α values: τ = {tau_mean:.2f} ± {tau_sem:.2f}"
        )

    # -------------------------------------
    # Save and plot results
    # -------------------------------------
    _save_results(tau_means, tau_sems, params)

    plot_desynchronisation_load(tau_means, tau_sems, params)

    return alphas, tau_means, tau_sems

def L_sweep(params):
    """
    Investigates how the characteristic desynchronisation load
    depends on the lattice size L.

    For each L, the characteristic desynchronisation
    load is estimated from multiple independent simulation runs.
    The resulting mean desynchronisation load and its standard
    error are saved and plotted as a function of L.

    Parameters
    ----------
    params : Parameters
        Object containing the model, simulation and sweep
        parameters.

    Returns
    -------
    lattice_sizes : ndarray
        Lattice sizes.

    tau_means : list[float]
        Mean desynchronisation load for each L.

    tau_sems : list[float]
        Standard error of the mean desynchronisation load.
    """
    # -------------------------------------
    # Initialise sweep
    # -------------------------------------

    params.sweep_type = "L"

    lattice_sizes = params.sweep_values

    tau_means = []
    tau_sems = []

    # -------------------------------------
    # Loop over lattice sizes
    # -------------------------------------
    for lattice_size in lattice_sizes:

        # Set the simulation lattice size
        params.L = lattice_size

        tau_mean, tau_sem = _estimate_desynchronisation_load(params)

        tau_means.append(tau_mean)
        tau_sems.append(tau_sem)

        print(
            f"Completed {len(tau_means)}/{len(lattice_sizes)} "
            f"lattice sizes: τ = {tau_mean:.2f} ± {tau_sem:.2f}"
        )

    # -------------------------------------
    # Save and plot results
    # -------------------------------------
    _save_results(tau_means, tau_sems, params)

    plot_desynchronisation_load(tau_means, tau_sems, params)

    return lattice_sizes, tau_means, tau_sems