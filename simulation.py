from cmath import phase

import numpy as np
import matplotlib.pyplot as plt
from model import drive
from model import avalanche


class Domain:

    def __init__(self, coords, phase):
        self.coords = coords
        self.size = len(coords)

        self.R = 0j
        for coord in coords:
            i, j = coord
            self.R += phase[i, j]

        self.history = []

    def update(self, delta_phase):
        self.R += delta_phase

    def record(self):
        self.history.append(np.abs(self.R) / self.size)

    def recompute(self, phase):
        R = 0j
        for coord in self.coords:
            i, j = coord
            R += phase[i, j]
        return R


def initialise(params):
    L = params.L
    F = np.zeros((L+2,L+2)) # Padded grid to avoid boundary checks
    F[1:-1, 1:-1] = np.random.rand(L, L)
    return F


def run_simulation(params, vis=None):

    L, Fth, iterations, begin_data = params.unpack()

    F = initialise(params)

    phase = np.exp(2j*np.pi*F)

    domains = []
    membership = {}
    sizes = []
    
    lattice_coords = [(i, j) for i in range(1, L+1) for j in range(1, L+1)]
    
    iteration_counter = 0
    loop_10_counter = 0

    for n in range(iterations):

        F, Fadd, initial_site = drive(F, params)

        phase_factor = np.exp(2j * np.pi * Fadd)
        phase[1:-1, 1:-1] *= phase_factor

        lattice = Domain(lattice_coords, phase)

        F, changed_sites = avalanche(F, initial_site, params, vis)

        coords = []

        for coord, info in changed_sites.items():
            coords.append(coord) 
        
        size = 0
        delta_phase_tot = 0j

        for coord, info in changed_sites.items():
            
            new_force = info["new_force"]
            size += info["topplings"]
            i, j = coord
            delta_phase = np.exp(2j * np.pi * new_force) - phase[i, j]
            phase[i, j] += delta_phase
            delta_phase_tot += delta_phase

            if coord in membership:
                for domain_id in membership[coord]:
                    domains[domain_id].update(delta_phase)

            else:
                membership[coord] = []

            domain_id = len(domains)

            if len(domains) < 1 and len(coords) > 150 and n > begin_data:
                membership[coord].append(domain_id)

        lattice.update(delta_phase_tot)
        sizes.append(size)

        domain_coords = [coord for coord in coords]

        domain = Domain(domain_coords, phase)

        if len(domains) < 1 and len(coords) > 150 and n > begin_data:
            domains.append(domain)

        for domain in domains:
            domain.R = domain.recompute(phase)
            domain.record()

        print(f"progress: {n * 100 / iterations:.1f}%")

        loop_10_counter += 1
        iteration_counter += 1

    print("Threshold force =", Fth)
    print("Mean force =", np.mean(F))
    print("Max earthquake size =", np.max(sizes))
    print("Mean earthquake size =", np.mean(sizes))

    return sizes, domains, lattice
