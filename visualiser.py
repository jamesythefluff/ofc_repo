"""
visualiser.py

Provides live visualisation tools for the OFC earthquake simulation.

Two visualisers are currently implemented:

    • ForceVisualiser
        Displays the force stored at each lattice site as a
        grayscale image during avalanche propagation.

    • PhaseVisualiser
        Displays the phase vectors of the sites in a tracked
        domain on the unit circle, providing a visual
        representation of synchronisation.

These visualisers are optional and can be disabled by passing
None to run_simulation().

Author: Anne-Marie
Date: July 2026
"""

# -------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# IMPORT MODULES
# -------------------------------------------------
from simulation import initialise


def create_phase_visualiser():

    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--')
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

    vis_phase = PhaseVisualiser(ax)
    
    return vis_phase


def create_force_visualiser(params):

    forces, phase_vetors = initialise(params)
    plt.ion()  # interactive mode
    fig, ax = plt.subplots()
    im = ax.imshow(forces, cmap='gray', origin='lower', vmin=0, vmax=1)
    plt.colorbar(im, label='Force')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

    vis_force = ForceVisualiser(im)

    return vis_force

# -------------------------------------------------
# FORCE LATTICE VISUALISER
# -------------------------------------------------
class ForceVisualiser:
    """
    Live visualisation of the force lattice.

    Attributes
    ----------
    im : matplotlib.image.AxesImage
        Image object returned by matplotlib.pyplot.imshow().
    """
    def __init__(self, im):
        self.im = im

    def update(self, forces):
        """
        Updates the displayed force lattice.

        Parameters
        ----------
        forces : np.ndarray
            Current force lattice.
        """
        self.im.set_data(forces)
        plt.pause(0.0001)

# -------------------------------------------------
# PHASE-CIRCLE VISUALISER
# -------------------------------------------------
class PhaseVisualiser:
    """
    Live visualisation of domain synchronisation.

    Each lattice site is represented by a unit vector

        r = exp(2πiforces),

    whose tip is plotted on the unit circle. Clustered phase
    vectors indicate a synchronised domain.

    Attributes
    ----------
    ax : matplotlib.axes.Axes
        The axes on which the unit-circle visualisation is drawn.

    scatter : matplotlib.collections.PathCollection
        Scatter plot artist representing the phase vectors.
        This object is updated each frame rather than recreated,
        allowing efficient real-time visualisation.

    Methods
    -------
    update(forces, displayed_domain)
        Updates the positions of the phase vectors for the
        specified domain.
    """
    def __init__(self, ax):
        self.ax = ax
        self.scatter = ax.scatter([], [])

    def update(self, forces, displayed_domain):
        """
        Updates the unit-circle visualisation.

        Parameters
        ----------
        forces : np.ndarray
            Current force lattice.

        displayed_domain : iterable[(int, int)]
            Coordinates of the lattice sites whose phase
            vectors are displayed.
        """
        # Store real and imaginary parts of complex phase vectors
        u = []
        v = []

        # Calculate and store real and imaginary parts of complex phase vectors
        for i, j in displayed_domain:
            theta = 2 * np.pi * forces[i, j]
            u.append(np.cos(theta))
            v.append(np.sin(theta))

        u = np.array(u)
        v = np.array(v)

        # Update plot
        self.scatter.set_offsets(np.column_stack((u, v)))

        plt.pause(0.0001)

