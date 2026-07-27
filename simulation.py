"""
simulation.py

Runs the OFC simulation by repeatedly driving the lattice,
generating avalanches, updating tracked domains, and recording
synchronisation statistics.

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
from model import drive, avalanche
from analysis import average_domain_synchronicity, fit_exponential


class TrackedDomain:
    """
    Represents a collection of lattice sites whose synchronisation
    is tracked throughout the simulation.

    Attributes
    ----------
    coords : list[(int, int)]
        Coordinates belonging to the domain.
    size : int
        Number of sites in the domain.
    R : complex
        Complex Kuramoto order parameter.
    history : list[float]
        Time series of |R|/N.

    Methods
    -------
    compute_R(phase_vectors)
        Computes the Kuramoto order parameter for the domain.
    update(phase_vectors)
        Updates the Kuramoto order parameter and appends its normalised magnitude to the history.
    """
    def __init__(self, coords, phase_vectors):
        self.coords = coords
        self.size = len(coords)
        self.compute_R(phase_vectors)
        self.history = []
        self.load_history = []

    def compute_R(self, phase_vectors):
        phase_vectors = np.array(phase_vectors)
        self.R = 0j
        for i,j in self.coords:
            self.R += phase_vectors[i, j]

    def update(self, phase_vectors, cumulative_loading_force):
        self.compute_R(phase_vectors)
        self.history.append(np.abs(self.R) / self.size)
        self.load_history.append(cumulative_loading_force)
        

def initialise(params):
    """
    Initialises an L by L lattice with padded boundaries.

    Returns
    -------
    forces : np.ndarray
        Randomly initialised force lattice with padded boundaries.
    phase_vectors : np.ndarray
        Complex phase vectors for each lattice site.
    """
    L = params.L
    forces = np.zeros((L+2,L+2)) # Padded grid to avoid boundary checks
    forces[1:-1, 1:-1] = np.random.rand(L, L)
    phase_vectors = np.exp(2j*np.pi*forces)
    return forces, phase_vectors


def run_simulation(params, vis_force=None, vis_phase=None):
    """
    Runs the OFC simulation.

    Returns
    -------
    cumulative_load : list
        Cumulative loading force applied to each site.
    sizes : list
        Time series of avalanche sizes.
    domains : list[TrackedDomain]
        Tracked domains.
    lattice : TrackedDomain
        Entire lattice domain.
    """
    # -------------------------------------------------
    # INITIALISE
    # -------------------------------------------------

    # Model parameters
    L = params.L
    ITERATIONS = params.iterations
    BEGIN_TRACKING = params.begin_data
    TRACKED_DOMAINS = params.tracked_domain_no
    MIN_DOMAIN_SIZE = params.min_domain_size
    MAX_DOMAIN_SIZE = params.max_domain_size
    SYNCHRONISATION_HISTORY_LENGTH = params.synchronisation_history_length

    # Initialize random force lattice
    forces, phase_vectors = initialise(params)

    # Initialise entire lattice domain for total synchronisation tracking
    lattice_coords = [(i, j) for i in range(1, L+1) for j in range(1, L+1)]
    lattice = TrackedDomain(lattice_coords, phase_vectors)

    # Store quake size time series and tracked domains
    sizes = []
    domains = []
    
    # Domain for unit circle order parameter visualiser
    displayed_domain = None

    cumulative_loading_force = 0

    # -------------------------------------------------
    # RUN SIMULATION
    # -------------------------------------------------
    for n in range(ITERATIONS):

        # -------------------------------------------------
        # Drive the lattice and generate an avalanche
        # -------------------------------------------------
        cumulative_loading_force, forces, seed_site = drive(cumulative_loading_force, forces, params)
        forces, phase_vectors, toppled_sites, topplings = avalanche(forces, seed_site, params, vis_force, vis_phase)

        # Update time series
        sizes.append(topplings)

        # Update unit circle visualisation
        if vis_phase is not None and displayed_domain is not None:
            vis_phase.update(forces, displayed_domain)

        # If the domain is suitable store it for tracking
        should_track = (len(domains) < TRACKED_DOMAINS and 
                           MIN_DOMAIN_SIZE <= len(toppled_sites) < MAX_DOMAIN_SIZE and 
                           n > BEGIN_TRACKING)
        if should_track:
            domain = TrackedDomain(toppled_sites, phase_vectors)
            domains.append(domain)
            # Only visualise the first tracked domain
            if displayed_domain is None:
                displayed_domain = toppled_sites

        # ------------------------------------------------
        # Update tracked domains and entire lattice domain
        # ------------------------------------------------
        for tracked_domain in domains:
            # Only track domain synchronisation over a certain number of avalanches (saves time)
            if len(tracked_domain.history) < SYNCHRONISATION_HISTORY_LENGTH:
                tracked_domain.update(phase_vectors, cumulative_loading_force)

        lattice.update(phase_vectors, cumulative_loading_force)

        # ------------------------------------------------
        # Progress
        # ------------------------------------------------
        if n % 1000 == 0:
            print(f"progress: {n * 100 / ITERATIONS:.1f}%")

    # ------------------------------------------------
    # Print summary
    # ------------------------------------------------
    print("Mean force =", np.mean(forces[1:-1, 1:-1]))
    print("Max earthquake size =", np.max(sizes))
    print("Mean earthquake size =", np.mean(sizes))

    return sizes, domains, lattice


def parameter_sweep(params, vis_force=None, vis_phase=None):

    K_min = params.K_min
    K_max = params.K_max
    sweeps = params.sweeps

    K_arr = np.linspace(K_min, K_max, sweeps)

    Ks = []
    taus = []

    for K in K_arr:

        params.K = K

        sizes, domains, lattice = run_simulation(params, vis_force=None, vis_phase=None)

        mean_R, time = average_domain_synchronicity(domains, params)
        
        time_fit, mean_R_fit, popt = fit_exponential(mean_R, time)
        A, tau, C = popt

        Ks.append(K)
        taus.append(tau)

    return Ks, taus

