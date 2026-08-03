"""
main.py

Entry point for the OFC earthquake simulation.

This script orchestrates the entire simulation workflow by

    1. Creating the simulation parameters.
    2. Initialising any live visualisations.
    3. Running the OFC simulation.
    4. Performing post-simulation analysis and plotting.

The simulation itself is implemented in `simulation.py`, while the
underlying OFC dynamics are defined in `model.py`. This file is
primarily responsible for configuring and connecting the different
components of the project.

Author: Anne-Marie
Date: July 2026
"""
# -------------------------------------------------
# IMPORT MODULES
# -------------------------------------------------
from model import Parameters
from simulation import run_simulation 
from plotting import (
    plot_average_synchronisation,
    plot_lattice_synchronisation,
    plot_power_law_fit,
)
from visualiser import ( 
    create_force_visualiser, 
    create_phase_visualiser
)
from experiments import alpha_sweep


def main():
    """
    Configures and executes the OFC simulation workflow.

    Depending on the configuration flags, either performs an
    alpha parameter sweep or runs a single simulation with live
    visualisation and post-processing analysis.
    """
    # -------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------
    ALPHA = 0.245

    ENABLE_PHASE_VISUALISATION = False
    ENABLE_FORCE_VISUALISATION = False
    RUN_ALPHA_SWEEP = True

    params = Parameters()
    params.alpha = ALPHA

    # -------------------------------------------------
    # RUN EXPERIMENT OR SIMULATION
    # -------------------------------------------------
    if RUN_ALPHA_SWEEP:
        alpha_sweep(params)
        return

    # -------------------------------------------------
    # INITIALISE VISUALISERS
    # -------------------------------------------------
    if ENABLE_FORCE_VISUALISATION:
        vis_force = create_force_visualiser(params)
    else:
        vis_force = None  # Pass None to run_simulation().

    if ENABLE_PHASE_VISUALISATION:
        vis_phase = create_phase_visualiser()
    else:
        vis_phase = None  # Pass None to run_simulation().

    # -------------------------------------------------
    # RUN SIMULATION
    # -------------------------------------------------
    sizes, domains, lattice = run_simulation(params, vis_force=vis_force, vis_phase=vis_phase)

    # -------------------------------------------------
    # POST-SIMULATION PLOTS
    # -------------------------------------------------
    plot_lattice_synchronisation(lattice)
    plot_power_law_fit(sizes)
    plot_average_synchronisation(domains, params)


if __name__ == "__main__":
    main()