import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class Parameters:

    def __init__(self):
        self.L = 50
        self.Fth = 1.0
        self.K = 1.0
        self.KL = 1.0

        self.iterations = 1000000
        self.begin_data = 400000

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
    return F, Fadd, initial_site



def avalanche(F, initial_site, params, vis=None):

    Fth = params.Fth
    a = params.a
    L = params.L

    changed_sites = {}

    q = deque([initial_site]) 

    while q:

        i, j = q.popleft()
        coord = (i, j)

        if F[i, j] < Fth:
            continue  # already relaxed (important safety check)

        if coord not in changed_sites:
            changed_sites[coord] = {
            "old_force": F[i, j],
            "new_force": None,
            "topplings" : 0
            }

        Fij = F[i, j]
        F[i, j] = 0

        changed_sites[coord]["new_force"] = F[i, j]
        changed_sites[coord]["topplings"] += 1

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

                if coord not in changed_sites:
                    changed_sites[coord] = {
                    "old_force": F[ni, nj],
                    "new_force": None,
                    "topplings" : 0
                    }

                F[ni, nj] += a * Fij
                
                changed_sites[coord]["new_force"] = F[ni, nj]
                changed_sites[coord]["topplings"] += 1

                if F[ni, nj] >= Fth:
                    q.append((ni, nj))

        if vis is not None:
            vis.update(F)

    return F, changed_sites

