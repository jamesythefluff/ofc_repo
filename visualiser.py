import matplotlib.pyplot as plt
import numpy as np

class ForceVisualiser:
    def __init__(self, im):
        self.im = im

    def update(self, F):
        self.im.set_data(F)
        plt.pause(0.0001)

class PhaseVisualiser:
    def __init__(self, ax):
        self.ax = ax
        self.scatter = ax.scatter([], [])

    def update(self, F, domain_vis):
        u = []
        v = []

        for i, j in domain_vis:
            theta = 2*np.pi*F[i, j]
            u.append(np.cos(theta))
            v.append(np.sin(theta))

        u = np.array(u)
        v = np.array(v)

        points = np.column_stack((u, v))
        self.scatter.set_offsets(points)

        plt.pause(0.0001)

