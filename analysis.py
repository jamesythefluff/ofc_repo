import numpy as np
import matplotlib.pyplot as plt
from model import Parameters


def plot_time_series(sizes):
    time = np.arange(len(sizes))
    scaled_sizes = sizes / np.max(sizes)
    plt.figure()
    plt.bar(time, scaled_sizes)
    plt.title("Earthquake size over time")
    plt.xlabel("Time")
    plt.ylabel("Scaled Earthquake Size")
    plt.show(block = True)



def plot_R(R_arr):
    x = np.arange(len(R_arr))
    plt.figure()
    plt.scatter(x, R_arr)
    plt.title("Synchronisation of entire lattice over time")
    plt.xlabel("Time")
    plt.ylabel("Modulus of R")
    plt.show(block = True)



def plot_domain_r(domains, r_arr, r_ref_arr, domain_no = 1):

    domain_processing_number = 0

    for i in range(len(domains)):
            
            if domain_processing_number < domain_no:

                domain_size = len(domains[i])
                r_mod_domain = r_arr[i]
                r_ref_mod_domain = r_ref_arr[i]

                print("Domain size =", domain_size)
                
                # Normalise
                r_norm_domain = np.array(r_mod_domain) / domain_size 
                r_ref_norm_domain =  np.array(r_ref_mod_domain) / domain_size
                
                time_arr = np.arange(len(r_norm_domain))
                
                # Plot magnitude of r over time for each domain
                plt.figure()
                plt.scatter(time_arr, r_ref_norm_domain)
                plt.scatter(time_arr, r_norm_domain)
                plt.title(f"Synchronisation of domain {i} over time")
                plt.xlabel("Time")
                plt.ylabel("Normalised Magnitude of r")
                plt.show(block = True)
                
                domain_processing_number += 1
        


def plot_sizes(sizes, params = Parameters()):
     
    iterations = params.iterations

    unique_sizes, counts = np.unique(sizes, return_counts=True) # Count the number of occurrences of each earthquake size
    prob = counts / iterations

    # Earthquake probability vs size plot
    plt.figure()
    plt.scatter(unique_sizes, prob)
    plt.xlabel("Earthquake size")
    plt.ylabel("Probability")
    plt.show(block = True)



def plot_sizes_poisson(sizes, params = Parameters()):
     
    iterations = params.iterations

    unique_sizes, counts = np.unique(sizes, return_counts=True) # Count the number of occurrences of each earthquake size
    prob = counts / iterations
    uncertainty = np.sqrt(counts) / iterations # Poisson statistics uncertainty

    # Earthquake probability vs size plot log-log plot with Poisson errorbars
    plt.figure()
    plt.errorbar(unique_sizes, prob, yerr = uncertainty, fmt = ".")
    plt.loglog()
    plt.xlabel("Earthquake size")
    plt.ylabel("Probability")
    plt.show(block = True)
