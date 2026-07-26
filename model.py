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
    Contains model and simulation parameters.

    Attributes
    ----------
    L : int
        Side length of the L by L square lattice.
    Fth : float
        Threshold force for site topple.
    K : float
        Coupling strength between sites.
    KL : float
        Leaf-spring coupling strenght (between tectonic plates).
    history : list[float]
        Time series of |R|/N.
    iterations : int
        Number of simulation iterations.
    begin_data : int
        Iteration number to start data collection.
    tracked_domain_no : int
        Number of domains to be tracked.
    min_domain_size : int
        Minimum suitable size (no. of sites) for a domain to be tracked.
    max_domain_size : int
        Maximum suitable size (no. of sites) for a domain to be tracked.
    synchronisation_history_length : int
        Length of domain synchronisation histories to be recorded.

    Properties
    -------
    a : float
        Fraction of force transferred to each neighbour during a topple.
    ae : float
        Fraction of force transferred to each neighbour during a topple for edge sites. (Currently not used in the model)
    ac : float
        Fraction of force transferred to each neighbour during a topple for corner sites. (Currently not used in the model)
    """
    def __init__(self):
        # Model parameters
        self.L = 32
        self.Fth = 1.0
        self.K = 2.0
        self.KL = 1.0
        # Simulation parameters
        self.iterations = 200000
        self.begin_data = 40000
        self.tracked_domain_no = 1000
        self.min_domain_size = 10
        self.max_domain_size = 20
        self.synchronisation_history_length = 3000


    @property
    def a(self):
        return self.K/(4*self.K + self.KL)

    @property
    def ae(self):
        return self.K/(3*self.K + self.KL)

    @property
    def ac(self):
        return self.K/(2*self.K + self.KL)
            


def drive(forces, params):
    """
    Drives the lattice by globally increasing the force on all sites such that the most stressed site reached threshold.

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
    Fadd = Fth - Fmax

    # Drive all sites except the padded boundaries
    forces[1:-1, 1:-1] += Fadd 

    return forces, initial_site



def avalanche(forces, initial_site, params, vis_force=None, vis_phase=None):
    """
    Propagates an avalanche starting from the intial site until all sites are below threshold.

    Returns
    -------
    forces : np.ndarray
        Updated force lattice.
    phase_vectors : np.ndarray
        Complex phase vectors for each lattice site.
    toppled_sites : list[tuple(int, int)]
        Coordinates of all unique sites that toppled.
    topplings : int
        Total number of topplings.
    """
    # -------------------------------------------------
    # INITIALISE
    # -------------------------------------------------

    # Initialize parameters
    Fth = params.Fth
    ALPHA = params.a
    L = params.L

    topplings = 0 
    toppled_sites = set()

    # Double-ended queue for efficient processing of active sites
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