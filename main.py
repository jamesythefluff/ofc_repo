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

sizes, domains, R_arr, r_arr, r_ref_arr = run_simulation(params, vis=vis)

plot_time_series(sizes)
plot_sizes(sizes, params)
plot_R(R_arr)
plot_domain_r(domains, r_arr, r_ref_arr)