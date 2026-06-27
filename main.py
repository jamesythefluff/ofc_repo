from simulation import run_simulation
from analysis import *

sizes, domains, R_arr, r_arr, r_ref_arr = run_simulation()

plot_time_series(sizes)
plot_sizes(sizes)
plot_R(R_arr)
plot_domain_r(domains, r_arr, r_ref_arr)