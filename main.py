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
# IMPORT LIBRARIES
# -------------------------------------------------
import matplotlib.pyplot as plt

# -------------------------------------------------
# IMPORT MODULES
# -------------------------------------------------
from model import Parameters
from simulation import initialise, run_simulation, parameter_sweep
from plotting import *
from visualiser import ForceVisualiser, PhaseVisualiser

# -------------------------------------------------
# SIMULATION PARAMETERS
# -------------------------------------------------
params = Parameters()

# -------------------------------------------------
# FORCE LATTICE VISUALISATION
# -------------------------------------------------
"""# Visulatisation
F = initialise(params)
plt.ion()  # interactive mode
fig, ax = plt.subplots()
im = ax.imshow(F, cmap='gray', origin='lower', vmin=0, vmax=1)
plt.colorbar(im, label='Force')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()"""

vis_force = None # ForceVisualiser(im)

# -------------------------------------------------
# PHASE-CIRCLE SYNCHRONISATION VISUALISATION
# -------------------------------------------------
"""plt.ion()
fig, ax = plt.subplots(figsize=(6,6))
# Draw unit circle
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), 'k--')
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
fig.canvas.draw()
fig.canvas.flush_events()
plt.pause(0.001)"""

vis_phase = None # PhaseVisualiser(ax)

# -------------------------------------------------
# RUN SIMULATION
# -------------------------------------------------
# Returns:
#   sizes   : avalanche size time series
#   domains : tracked synchronisation domains
#   lattice : synchronisation history of the entire lattice

#sizes, domains, lattice = run_simulation(params, vis_force=vis_force, vis_phase=vis_phase)
Ks, taus = parameter_sweep(params, vis_force=vis_force, vis_phase=vis_phase)

# -------------------------------------------------
# POST-SIMULATION ANALYSIS
# -------------------------------------------------
#plot_lattice_R(lattice)
#plot_domain_R(domains)
#plot_power_law_fit(sizes)
#plot_desynchronisation(domains, params)
plot_desynchronisation_time(Ks, taus)