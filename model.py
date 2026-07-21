import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class Parameters:

    def __init__(self):
        self.L = 32
        self.Fth = 1.0
        self.K = 1.0
        self.KL = 1.0

        self.iterations = 100000
        self.begin_data = 40000

    def unpack(self):
        return self.L, self.Fth, self.iterations, self.begin_data
    
    @property
    def a(self):
        return self.K/(4*self.K + self.KL)

    @property
    def ae(self):
        return self.K/(3*self.K + self.KL)

    @property
    def ac(self):
        return self.K/(2*self.K + self.KL)
            


def drive(F, params):
    Fth = params.Fth
    Fmax = np.max(F)
    initial_site = np.argwhere(F == Fmax)[0]
    Fadd = Fth - Fmax
    F[1:-1, 1:-1] += Fadd
    return F, initial_site



def avalanche(F, initial_site, params, vis=None):

    Fth = params.Fth
    a = params.a
    L = params.L

    topplings = 0
    toppled_sites = set()  # To keep track of unique toppled sites

    q = deque([initial_site]) 

    while q:

        i, j = q.popleft()
        coord = (i, j)

        if F[i, j] < Fth:
            continue  # already relaxed (important safety check)

        topplings += 1
        toppled_sites.add(coord)

        Fij = F[i, j]
        F[i, j] = 0

        # neighbours (padded grid → no boundary checks needed)
        neighbours = [
            (i+1, j),
            (i-1, j),
            (i, j+1),
            (i, j-1)
        ]

        for ni, nj in neighbours:

            if 1 <= ni <= L and 1 <= nj <= L: # Ensure padding is untouched

                coord = (ni, nj)

                F[ni, nj] += a * Fij

                if F[ni, nj] >= Fth:
                    q.append((ni, nj))

        if vis is not None:
            vis.update(F)

    phase = np.exp(2j * np.pi * F)

    return F, phase, toppled_sites, topplings

