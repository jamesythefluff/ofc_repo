"""
simulation.py

Provides the high-level routines required to run a single
OFC simulation. The module initialises the lattice,
tracks the synchronisation of selected earthquake domains,
and records simulation data for subsequent analysis.

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


class TrackedDomain:
    """
    Represents a domain of lattice sites whose synchronisation
    is monitored after an earthquake.

    Each tracked domain stores its constituent lattice sites,
    its normalised synchronisation, and the evolution of its
    synchronisation as the lattice is subsequently
    driven.

    Attributes
    ----------
    coords : list[(int, int)]
        Coordinates belonging to the domain.

    size : int
        Number of sites in the domain.

    synchronisation : float
        Normalised synchronisation parameter.

    history : list[float]
        Time series of |R|/N.

    load_history : list[float]
        Cumulative loading force corresponding to each recorded
        synchronisation value.

    Methods
    -------
    compute_synchronisation(phase_vectors)
        Computes the synchronisation for the domain.

    update(phase_vectors)
        Updates the synchronisation and appends it to the history.
    """
    def __init__(self, coords, phase_vectors):
        self.coords = coords
        self.size = len(coords)
        self.compute_synchronisation(phase_vectors)
        self.synchronisation_history = []
        self.load_history = []

    def compute_synchronisation(self, phase_vectors):
        """
        Computes the synchronisation for the domain.

        Parameters
        ----------
        phase_vectors : np.ndarray
            Complex phase vector associated with every lattice
            site.

        Notes
        -----
        The synchronisation is the absolute value of the
        vector sum of the phase vectors belonging to the tracked 
        domain, normalised by the domain size.
        """
        vector_sum = 0j

        for i,j in self.coords:
            vector_sum += phase_vectors[i, j]

        # Take absolute value and normalise by domain size
        self.synchronisation = abs(vector_sum) / self.size

    def update(self, phase_vectors, cumulative_loading_force):
        """
        Updates the synchronisation history of the domain.

        Parameters
        ----------
        phase_vectors : np.ndarray
            Complex phase vectors for the current lattice state.

        cumulative_loading_force : float
            Total loading force applied to the lattice since the
            start of the simulation.

        Notes
        -----
        Computes and stores the current synchronisation together 
        with the corresponding cumulative loading force.
        """
        self.compute_synchronisation(phase_vectors)
        self.synchronisation_history.append(self.synchronisation)
        self.load_history.append(cumulative_loading_force)
        

def initialise(params):
    """
    Initialises an L by L lattice with padded boundaries.

    The lattice is padded with a one-site border of zeros.
    This simplifies neighbour indexing by avoiding explicit
    boundary checks when constructing neighbour coordinates.

    Parameters
    ----------
    params : Parameters
        Simulation parameters containing the lattice size.

    Returns
    -------
    forces : np.ndarray
        Randomly initialised force lattice with padded boundaries.

    phase_vectors : np.ndarray
        Complex phase vectors for each lattice site.
    """
    L = params.L
    forces = np.zeros((L+2,L+2)) # Padded grid to avoid boundary checks

    # Populate the lattice with random forces in the range [0,1)
    # Exclude the padded boundaries which remain at zero
    forces[1:-1, 1:-1] = np.random.rand(L, L) 

    phase_vectors = np.exp(2j*np.pi*forces)

    return forces, phase_vectors


def run_simulation(params, vis_force=None, vis_phase=None):
    """
    Runs a single OFC simulation by repeatedly driving the
    lattice, propagating avalanches, recording earthquake
    sizes, and tracking the synchronisation of selected
    domains.

    Parameters
    ----------
    params : Parameters
        Model and simulation parameters.

    vis_force : ForceVisualiser, optional
        Interactive force lattice visualiser.

    vis_phase : PhaseVisualiser, optional
        Interactive phase vector visualiser.

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
            domain.update(phase_vectors, cumulative_loading_force)
            domains.append(domain)
            # Visualise only the first tracked domain
            displayed_domain = displayed_domain or toppled_sites

        # ------------------------------------------------
        # Update tracked domains and entire lattice domain
        # ------------------------------------------------
        for tracked_domain in domains:
            # Limit the recorded synchronisation history to reduce
            # memory usage and computation time.
            if len(tracked_domain.synchronisation_history) < SYNCHRONISATION_HISTORY_LENGTH:
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