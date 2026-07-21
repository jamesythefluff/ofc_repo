from simulation import initialise, run_simulation
from analysis import *
from visualiser import ForceVisualiser, PhaseVisualiser

params = Parameters()

F = initialise(params)

"""# Visulatisation
plt.ion()  # interactive mode
fig, ax = plt.subplots()
im = ax.imshow(F, cmap='gray', origin='lower', vmin=0, vmax=1)
plt.colorbar(im, label='Force')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()"""

vis_force = None #ForceVisualiser(im)


# Domain synchronisation unit circle visualisation

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


sizes, domains, lattice = run_simulation(params, vis_force=vis_force, vis_phase=vis_phase)

#plot_time_series(sizes)
plot_lattice_R(lattice)
plot_domain_R(domains)
plot_sizes_poisson(sizes, params)