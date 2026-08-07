"""
experiments.py

Contains high-level numerical experiments performed using the
OFC model.

Each experiment coordinates one or more simulations, analyses
the resulting data, and produces figures suitable for comparison
and interpretation. Internal helper functions provide common
functionality shared between experiments, such as estimating
desynchronisation times and saving results.

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
from analysis import average_domain_synchronicity, fit_exponential
from plotting import plot_desynchronisation_load

# -------------------------------------
# Internal helper functions
# -------------------------------------

def _estimate_desynchronisation_time(params):
    """
    Estimates the characteristic desynchronisation time for a
    single parameter set.

    The simulation is repeated several times using independent
    random initial conditions. For each run, the average domain
    synchronisation is calculated and fitted with an exponential
    decay to extract the characteristic desynchronisation time.
    The mean desynchronisation time and its standard error are
    returned.

    Parameters
    ----------
    params : Parameters
        Object containing the model, simulation and experiment
        parameters. The current value of ``params.alpha`` is used
        for every simulation.

    Returns
    -------
    tau_mean : float
        Mean desynchronisation time obtained from all simulation
        runs.

    tau_sem : float
        Standard error of the mean desynchronisation time.
    """
    tau_values = []

    # -------------------------------------
    # Loop over independent simulation runs
    # -------------------------------------
    
    for run in range(params.runs_per_alpha):

        sizes, domains, lattice = run_simulation(params)

        mean_synchronicity, time = average_domain_synchronicity(domains, params)
        
        time_fit, mean_synchronicity_fit, popt, tau_err = fit_exponential(mean_synchronicity, time)
        tau = popt[1]

        # Only store reasonable fits (finite and positive desynchronisation times)
        if np.isfinite(tau) and tau > 0:
            tau_values.append(tau)  

    # -------------------------------------
    # Average over all runs to obtain mean and standard error of the mean
    # -------------------------------------
    if len(tau_values) > 0:
        tau_mean = np.mean(tau_values)
        tau_sem = np.std(tau_values, ddof=1) / np.sqrt(len(tau_values))
    else:
        print(f"Warning: No valid desynchronisation times found for alpha = {params.alpha}.")

    return tau_mean, tau_sem


def _save_results(alphas, tau_means, tau_sems, params):
    """
    Saves the results of an alpha sweep to a NumPy archive.

    The filename is generated automatically from the lower and
    upper bounds of the alpha sweep.

    Parameters
    ----------
    alphas : array_like
        Alpha values used in the sweep.

    tau_means : array_like
        Mean desynchronisation time corresponding to each alpha.

    tau_sems : array_like
        Standard error of the mean desynchronisation time for
        each alpha.

    params : Parameters
        Parameter object used to generate the filename.
    """
    filename = (
        f"alpha_sweep_"
        f"{params.alpha_min:.3f}_"
        f"{params.alpha_max:.3f}.npz"
    )

    np.savez(filename,
        alphas=alphas,
        tau_means=tau_means,
        tau_sems=tau_sems,
    )

# -------------------------------------
# Public experiments
# -------------------------------------

def alpha_sweep(params):
    """
    Investigates how the characteristic desynchronisation time
    depends on the stress transfer parameter α.

    The neighbour coupling constant is varied to produce a range
    of α values. For each α, the characteristic desynchronisation
    time is estimated from multiple independent simulation runs.
    The resulting mean desynchronisation time and its standard
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
        Mean desynchronisation time for each α.

    tau_sems : list[float]
        Standard error of the mean desynchronisation time.
    """
    # -------------------------------------
    # Initialise sweep
    # -------------------------------------

    # Generate array of alpha values from the input parameters
    alphas = np.linspace(params.alpha_min, params.alpha_max, params.num_alpha_values)

    tau_means = []
    tau_sems = []

    # -------------------------------------
    # Loop over alpha values
    # -------------------------------------
    for alpha in alphas:

        # Set the simulation alpha
        params.alpha = alpha

        tau_mean, tau_sem = _estimate_desynchronisation_time(params)

        tau_means.append(tau_mean)
        tau_sems.append(tau_sem)

        print(
            f"Completed {len(tau_means)}/{len(alphas)} "
            f"α values: τ = {tau_mean:.2f} ± {tau_sem:.2f}"
        )

        #plot_desynchronisation(domains, params)

    # -------------------------------------
    # Save and plot results
    # -------------------------------------
    _save_results(alphas, tau_means, tau_sems, params)

    plot_desynchronisation_load(alphas, tau_means, tau_sems, params)

    return alphas, tau_means, tau_sems