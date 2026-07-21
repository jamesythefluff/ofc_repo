from simulation import initialise, run_simulation
from analysis import *
from visualiser import Visualiser

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

vis = None #Visualiser(im)

sizes, domains, lattice = run_simulation(params, vis=vis)

#plot_time_series(sizes)
plot_lattice_R(lattice)
plot_domain_R(domains)
plot_sizes_poisson(sizes, params)