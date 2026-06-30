import numpy as np
import matplotlib.pyplot as plt
from model import Parameters
from model import drive
from model import avalanche


def initialise(params):
    L = params.L
    F = np.zeros((L+2,L+2)) # Padded grid to avoid boundary checks
    F[1:-1, 1:-1] = np.random.rand(L, L)
    return F


def run_simulation(params, vis=None):

    iterations = params.iterations
    begin_data = params.begin_data
    L = params.L
    Fth = params.Fth

    F = np.zeros((L+2,L+2)) # Padded grid to avoid boundary checks
    F[1:-1, 1:-1] = np.random.rand(L, L)

    sizes = [] 
    domains = [] 
    r_arr = [[] for _ in range(iterations - begin_data)] 
    R_arr = [] 
    r_ref_arr = [[] for _ in range(iterations - begin_data)]

    iteration_counter = 0
    loop_10_counter = 0

    for n in range(iterations):

        F, initial_site = drive(F, params)
        F, avalanche_size, domain = avalanche(F, initial_site, params, vis)
            
        # Calculation of R value for entire lattice - expect plateau when steady state reached
        if loop_10_counter == 1000: # Only calculate R value every 1000 iterations to save time

            loop_10_counter = 0

            interior = F[1:-1, 1:-1]
 
            R_x = np.sum(np.cos(2*np.pi*interior))
            R_y = np.sum(np.sin(2*np.pi*interior))

            R_mod = np.sqrt(R_x**2 + R_y**2)
            R_norm = R_mod / (L*L)
            R_arr.append(R_norm)

            print(f"progress {iteration_counter/iterations*100:.3f} %") # Progress update
        
        if iteration_counter > begin_data: # Only take data once steady-state is reached
            
            sizes.append(avalanche_size)

            if len(domains) < 1:
                if len(domain) > 100: # Only look at reasonably sized domains
                    domains.append(domain)

            if loop_10_counter == 0: # Only calculate domain r values every 1000 iterations to save time

                # Calculation of r values for each domain
                for domain_idx in range(len(domains)):
                    r = [0,0]
                    for coord in domains[domain_idx]:
                        i = coord[0]
                        j = coord[1]
                        F_ij = F[i,j]
                        r_ij = [np.cos(2*np.pi*F_ij), np.sin(2*np.pi*F_ij)] # Vector in complex plane
                        r[0] += r_ij[0]
                        r[1] += r_ij[1]
                        
                    r_ref = [0,0]
                    rand_i_arr = np.random.randint(1, L+1, size=len(domains[domain_idx]))
                    rand_j_arr = np.random.randint(1, L+1, size=len(domains[domain_idx]))

                    # Calculate reference total r value from random sample with size equal to the domain size
                    for index in range(len(domains[domain_idx])):
                        rand_i = rand_i_arr[index]
                        rand_j = rand_j_arr[index]
                        F_ij_rand = F[rand_i,rand_j]
                        r_ij_rand = [np.cos(2*np.pi*F_ij_rand), np.sin(2*np.pi*F_ij_rand)]
                        r_ref[0] += r_ij_rand[0]
                        r_ref[1] += r_ij_rand[1]
                    
                    r_mod_arr = np.sqrt(r[0]**2 + r[1]**2)
                    r_ref_mod_arr = np.sqrt(r_ref[0]**2 + r_ref[1]**2)

                    r_ref_arr[domain_idx].append(r_ref_mod_arr)
                    r_arr[domain_idx].append(r_mod_arr)

        loop_10_counter += 1
        iteration_counter += 1

    print("Threshold force =", Fth)
    print("Mean force =", np.mean(F))
    print("Max earthquake size =", np.max(sizes))
    print("Mean earthquake size =", np.mean(sizes))

    return sizes, domains, R_arr, r_arr, r_ref_arr
