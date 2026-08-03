"""
model.py

Contains model and simulation parameters, as well as
functions to drive the lattice and generate avalanches.

Author: Anne-Marie
Date: July 2026
"""

# -------------------------------------
# Import libraries
# -------------------------------------
import numpy as np
from collections import deque


class Parameters:
    """
    Stores all parameters controlling the OFC model, simulation,
    and numerical experiments.

    The class also provides properties for computing the stress
    transfer coefficient α from the spring constants, and allows
    α to be set directly by updating the neighbour coupling K.

    Attributes
    ----------

    MODEL
    ----
    L : int
        Side length of the L by L square lattice.

    Fth : float
        Threshold force for site topple.

    K : float
        Coupling strength between sites.

    KL : float
        Leaf-spring coupling strength (between tectonic plates).

    history : list[float]
        Time series of |R|/N.

    SIMULATION
    ---------
    iterations : int
        Number of simulation iterations.

    begin_data : int
        Iteration number to start data collection.

    tracked_domain_no : int
        Number of domains to be tracked.

    min_domain_size : int
        Minimum suitable size (no. of sites) for a domain to 
        be tracked.

    max_domain_size : int
        Maximum suitable size (no. of sites) for a domain 
        to be tracked.

    synchronisation_history_length : int
        Length of domain synchronisation histories to be recorded.

    EXPERIMENT
    ----------
    alpha_min : float
        Lower bound for alpha sweep experiment.

    alpha_max : float
        Upper bound for alpha sweep experiment.

    num_alpha_values : int
        Number of different alpha values used in the alpha sweep 
        experiment.

    runs_per_alpha : int
        Number of runs per alpha value for tau mean and sem calculation 
        in alpha sweep experiment.

    Properties
    -------
    alpha : float
        Stress transfer coefficient. Reading this property computes α
        from K and KL. Assigning to it updates K while keeping KL fixed.

    alpha_edge : float
        Stress transfer coefficient for edge sites. (Currently not used in the model)

    alpha_corner : float
        Stress transfer coefficient for corner sites. (Currently not used in the model)
    """
    def __init__(self):

        # Model parameters
        self.L = 32
        self.Fth = 1.0
        self.K = 1.0
        self.KL = 1.0

        # Simulation parameters
        self.iterations = 100000
        self.begin_data = 40000
        self.tracked_domain_no = 1000
        self.min_domain_size = 10
        self.max_domain_size = 20
        self.synchronisation_history_length = 10000

        # Experiment parameters
        self.alpha_min = 0.15
        self.alpha_max = 0.24
        self.num_alpha_values = 20
        self.runs_per_alpha = 10

    @property
    def alpha(self):
        """
        float : Stress transfer parameter α.

        Computed from the current neighbour coupling constant K
        and leaf spring constant KL as

            α = K / (4K + KL).
        """
        return self.K/(4*self.K + self.KL)
    
    @alpha.setter
    def alpha(self, value):
        """
        Sets the stress transfer parameter α by updating the
        neighbour coupling constant K.

        Parameters
        ----------
        value : float
            Desired value of the stress transfer parameter α.
            The corresponding neighbour coupling constant K is
            calculated while keeping KL fixed.

        Raises
        ------
        ValueError
            If alpha is outside the interval 0 ≤ α < 0.25.
        """
        self.K = value * self.KL / (1 - 4*value)

        if not (0 <= value < 0.25):
            raise ValueError("Unphysical stress transfer parameter")

    @property
    def alpha_edge(self):
        return self.K/(3*self.K + self.KL)

    @property
    def alpha_corner(self):
        return self.K/(2*self.K + self.KL)


def drive(cumulative_loading_force, forces, params):
    """
    Drives the lattice by globally increasing the force on 
    all sites such that the most stressed site reaches threshold.

    Parameters
    ----------
    cumulative_loading_force : float
        Total loading applied since the start of the simulation.

    forces : np.ndarray
        Current force lattice.

    params : Parameters
        Model parameters.

    Returns
    -------
    forces : np.ndarray
        Updated force lattice.

    initial_site : tuple(int, int)
        Coordinates of the site that reached threshold.
    """
    # Obtain Fth and Fmax
    Fth = params.Fth
    Fmax = np.max(forces)
    initial_site = np.argwhere(forces == Fmax)[0]

    # Calculate uniform drive amount
    loading_increment = Fth - Fmax

    # Drive all sites except the padded boundaries
    forces[1:-1, 1:-1] += loading_increment

    # Update cumulative loading force
    cumulative_loading_force += loading_increment

    return cumulative_loading_force, forces, initial_site


def avalanche(forces, initial_site, params, vis_force=None, vis_phase=None):
    """
    Propagates an avalanche from an initial unstable site until all
    sites in the lattice are below the threshold force.

    During propagation, each unstable site topples by resetting its
    force to zero and redistributing a fraction of its force to its
    four nearest neighbours. Any neighbours that subsequently exceed
    the threshold are added to a queue and processed in turn.

    Parameters
    ----------
    forces : np.ndarray
        Current force lattice, including the padded boundary.
        This array is modified in place during the avalanche.

    initial_site : tuple[int, int]
        Coordinates of the site that first reached the threshold
        and triggered the avalanche.

    params : Parameters
        Object containing the model and simulation parameters,
        including the threshold force, stress transfer coefficient,
        and lattice size.

    vis_force : ForceVisualiser, optional
        Interactive force lattice visualiser. If provided, the
        display is updated after each site topples. Default is None.

    vis_phase : PhaseVisualiser, optional
        Reserved for future phase-vector visualisation during
        avalanche propagation. Currently unused. Default is None.

    Returns
    -------
    forces : np.ndarray
        Updated force lattice after the avalanche has finished.

    phase_vectors : np.ndarray
        Complex phase vector corresponding to the final force at
        each lattice site.

    toppled_sites : list[tuple[int, int]]
        Coordinates of all unique sites that toppled during the
        avalanche.

    topplings : int
        Total number of toppling events during the avalanche.
        This counts repeated topplings of the same site separately.
    """
    # -------------------------------------------------
    # INITIALISE
    # -------------------------------------------------

    # Initialize parameters
    Fth = params.Fth
    ALPHA = params.alpha
    L = params.L

    topplings = 0 
    toppled_sites = set()

    # Double-ended queue used to process unstable sites in
    # first-in-first-out order until no sites remain above threshold.
    q = deque([initial_site]) 

    # -------------------------------------------------
    # PROPAGATE AVALANCHE
    # -------------------------------------------------
    while q:
        #-------------------------------------------------
        # Process the next site in the queue
        #-------------------------------------------------
        i, j = q.popleft()
        force = forces[i, j]
        
        if force < Fth:
            continue  # already relaxed (important safety check)

        # ------------------------------------------------
        # Topple site
        # ------------------------------------------------
        # Reset force on toppled site
        forces[i, j] = 0 

        # Update avalanche size and domain
        topplings += 1
        toppled_sites.add((i,j))

        # ------------------------------------------------
        # Redistribute force to neighbours
        # ------------------------------------------------
        # neighbours (padded grid → no boundary checks needed)
        neighbours = [
            (i+1, j),
            (i-1, j),
            (i, j+1),
            (i, j-1)
        ]

        for ni, nj in neighbours:

            if 1 <= ni <= L and 1 <= nj <= L: # Ensure padding is untouched
                forces[ni, nj] += ALPHA * force # Update force on site
                
                # Add site to queue of active sites if force exceeds threshold 
                if forces[ni, nj] >= Fth:
                    q.append((ni, nj)) 
        
        # ------------------------------------------------
        # UPDATE FORCE VISUALISER
        # ------------------------------------------------
        if vis_force is not None:
            vis_force.update(forces)

    # Convert quake domain set to list
    toppled_sites = list(toppled_sites)

    # Calculate new phase vector array
    phase_vectors = np.exp(2j * np.pi * forces)

    return forces, phase_vectors, toppled_sites, topplings