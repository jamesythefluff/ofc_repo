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

# Delay between interactive figure updates (seconds).
_REFRESH_INTERVAL = 1e-4


# -------------------------------------------------
# FORCE LATTICE VISUALISER
# -------------------------------------------------
class ForceVisualiser:
    """
    Live visualisation of the force stored at each lattice site
    during avalanche propagation.

    Attributes
    ----------
    im : matplotlib.image.AxesImage
        Image artist used to display the force lattice.

    figure : matplotlib.figure.Figure
        Figure containing the displayed force lattice.

    Methods
    -------
    create(forces)
        Creates and initialises a live force lattice
        visualiser.

    update(forces)
        Updates the displayed force lattice using the
        current force values.
    """
    def __init__(self, im):
        self.im = im
        self.figure = im.figure

    @classmethod
    def create(cls, forces):
        """
        Creates and initialises a live force lattice visualiser.

        Parameters
        ----------
        forces : np.ndarray
            Initial force lattice.

        Returns
        -------
        ForceVisualiser
            An initialised instance of the force lattice visualiser.
        """
        # Interactive mode
        plt.ion()

        # Create Figure and Axes objects
        fig, ax = plt.subplots()

        # Ceate an AxesImage object to display the force lattice as a grayscale image.
        im = ax.imshow(
            forces,
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=1,
        )

        # Add a colour bar to indicate the force values
        plt.colorbar(im, label="Force")

        # Format axes
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Force lattice")

        return cls(im)

    def update(self, forces):
        """
        Updates the displayed force lattice and refreshes 
        the interactive figure.

        Parameters
        ----------
        forces : np.ndarray
            Current force lattice.
        """
        # Update the data stored in the AxesImage object and refresh the figure.
        self.im.set_data(forces)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

        plt.pause(_REFRESH_INTERVAL)


# -------------------------------------------------
# PHASE-CIRCLE VISUALISER
# -------------------------------------------------
class PhaseVisualiser:
    """
    Live visualisation of the synchronisation of a tracked domain.

    Each lattice site is represented by the complex phase vector

    exp(2πiF),

    where F ∈ [0, 1] is the normalised force stored at a
    lattice site. The tips of these vectors are plotted on 
    the complex unit circle, allowing the degree of synchronisation 
    within a tracked domain to be visualised.

    Attributes
    ----------
    ax : matplotlib.axes.Axes
        The axes on which the unit-circle visualisation is drawn.

    scatter : matplotlib.collections.PathCollection
        Scatter plot artist representing the phase vectors.
        This object is updated each frame rather than recreated,
        allowing efficient real-time visualisation.

    figure : matplotlib.figure.Figure
        Figure containing the unit-circle visualisation.

    Methods
    -------
    update(forces, displayed_domain)
        Updates the positions of the phase vectors for the
        specified domain.
    """
    def __init__(self, ax):
        self.scatter = ax.scatter([], [])
        self.figure = self.scatter.figure
        

    @classmethod
    def create(cls):
        """
        Creates and initialises the phase-circle visualiser.

        A Matplotlib figure containing the complex unit circle is
        created and configured for interactive updating during the
        simulation.

        Returns
        -------
        PhaseVisualiser
            Initialised phase-circle visualiser.
        """
        # Interactive mode
        plt.ion()

        # Create Figure and Axes objects
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot unit circle by creating a Scatter object
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), "k--")

        # Format axes
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title("Domain phase vectors")

        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.pause(_REFRESH_INTERVAL)

        return cls(ax)

    def update(self, forces, displayed_domain):
        """
        Updates the unit-circle visualisation and refreshes 
        the interactive figure

        Parameters
        ----------
        forces : np.ndarray
            Current force lattice.

        displayed_domain : iterable[(int, int)]
            Coordinates of the tracked domain whose phase
            vectors are displayed on the unit circle.
        """
        points = np.empty((len(displayed_domain), 2))

        # Convert each site's phase vector into Cartesian coordinates.
        for k, (i, j) in enumerate(displayed_domain):
            theta = 2*np.pi*forces[i, j]
            points[k] = np.cos(theta), np.sin(theta)

        # Update Scatter object data and refresh the figure.
        self.scatter.set_offsets(points)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

        plt.pause(_REFRESH_INTERVAL)