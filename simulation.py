from cmath import phase

import numpy as np
import matplotlib.pyplot as plt
from model import drive
from model import avalanche
from visualiser import ForceVisualiser, PhaseVisualiser


class Domain:

    def __init__(self, coords, phase):
        self.coords = coords
        self.size = len(coords)

        self.R = 0j
        for coord in coords:
            i, j = coord
            self.R += phase[i, j]

        self.history = []

    def update(self, phase):
        R = 0j
        for coord in self.coords:
            i, j = coord
            R += phase[i, j]
        return R

    def record(self):
        self.history.append(np.abs(self.R) / self.size)


def initialise(params):
    L = params.L
    F = np.zeros((L+2,L+2)) # Padded grid to avoid boundary checks
    F[1:-1, 1:-1] = np.random.rand(L, L)
    return F


def run_simulation(params, vis_force=None, vis_phase=None):

    L, Fth, iterations, begin_data = params.unpack()

    F = initialise(params)
    phase = np.exp(2j*np.pi*F)

    domains = []
    sizes = []
    
    lattice_coords = [(i, j) for i in range(1, L+1) for j in range(1, L+1)]
    lattice = Domain(lattice_coords, phase)
    
    iteration_counter = 0
    loop_10_counter = 0

    start_circle_vis = False

    for n in range(iterations):

        F, initial_site = drive(F, params)
        F, toppled_sites, topplings = avalanche(F, initial_site, params, vis_force, vis_phase)

        phase = np.exp(2j * np.pi * F)

        if start_circle_vis == True:
            vis_phase.update(F, circle_vis_domain)

        sizes.append(topplings)
        domain_coords = list(toppled_sites)

        domain = Domain(domain_coords, phase)

        if len(domains) < 1 and len(domain_coords) > 150 and n > begin_data:
            domains.append(domain)
            circle_vis_domain = np.array(domain_coords)
            start_circle_vis = True

        for domain in domains:
            domain.R = domain.update(phase)
            domain.record()

        lattice.R = lattice.update(phase)
        lattice.record()

        print(f"progress: {n * 100 / iterations:.1f}%")

    print("Threshold force =", Fth)
    print("Mean force =", np.mean(F))
    print("Max earthquake size =", np.max(sizes))
    print("Mean earthquake size =", np.mean(sizes))

    return sizes, domains, lattice
