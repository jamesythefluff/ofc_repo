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



def plot_lattice_R(lattice):
    R_lattice = lattice.history[::1000]
    x = np.arange(len(R_lattice))
    plt.figure()
    plt.scatter(x, R_lattice)
    plt.title("Synchronisation of entire lattice over time")
    plt.xlabel("Time")
    plt.ylabel("Modulus of R")
    plt.show(block = True)



def plot_domain_R(domains):

    for domain in domains:
            
            R_domain = domain.history[::1000]

            print("Domain size =", domain.size)
            time_arr = np.arange(len(R_domain))
            
            # Plot magnitude of R over time for each domain
            plt.figure()
            plt.scatter(time_arr, R_domain)
            plt.title(f"Synchronisation of domain size {domain.size} over time")
            plt.xlabel("Time")
            plt.ylabel("Normalised Magnitude of R")
            plt.show(block = True)

        


def plot_sizes(sizes, params):
     
    iterations = params.iterations

    unique_sizes, counts = np.unique(sizes, return_counts=True) # Count the number of occurrences of each earthquake size
    prob = counts / iterations

    # Earthquake probability vs size plot
    plt.figure()
    plt.scatter(unique_sizes, prob)
    plt.xlabel("Earthquake size")
    plt.ylabel("Probability")
    plt.show(block = True)



def plot_sizes_poisson(sizes, params):
     
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
