# OFC Earthquake Model with Domain Synchronisation Analysis

## Overview

This project implements the two-dimensional Olami–Feder–Christensen (OFC) earthquake model and provides tools for analysing the synchronisation of lattice sites during its evolution.

The primary aim is to investigate the desynchronisation of earthquake domains over time. Synchronisation is quantified using a Kuramoto-style complex order parameter, allowing both individual domains and the entire lattice to be monitored over time.

---

## Model

Each lattice site stores a force value between 0 and the threshold force (Fth).

The simulation repeatedly performs two stages:

1. **Global Driving**

   * The force on every lattice site is increased such that the most stressed site is pushed to threshold, triggering an earthquake.

2. **Avalanche**

   * The site at threshold topples.
   * Its force is reset to zero.
   * A fraction 0 < alpha < 0.25 of the released force is redistributed to each nearest neighbour site.
   * Redistribution may trigger further topplings, producing an avalanche.

The avalanche continues until every site is below threshold, at which point the system is once again driven and the process continues.

---

## Synchronisation

Each lattice force is mapped onto the unit circle using

phase_vectors = exp(2 * pi * j * forces)

For a domain containing N sites, the complex order parameter is

R = sum(phase_vectors[i,j])

where the sum runs over sites (i,j) in the domain.

The recorded synchronisation measure is then

|R| / N,

which ranges between

* **0** — completely desynchronised
* **1** — perfectly synchronised

Both individual tracked domains and the entire lattice are monitored throughout the simulation.

---

# Project Structure

```
ofc_model/
│
├── main.py
│   Entry point of the simulation.
│
├── model.py
│   OFC dynamics.
│   • Parameter class
│   • Driving
│   • Avalanche propagation
│
├── simulation.py
│   Controls the simulation loop.
│   Creates tracked domains.
│   Records synchronisation histories.
│
├── analysis.py
│   Plotting and statistical analysis.
│
├── visualiser.py
│   Live visualisation tools.
│   • Force lattice
│   • Phase-circle synchronisation visualiser
│
└── README.md
```

---

## Simulation Workflow

```
Initialise force lattice
        │
        ▼
Compute initial phase vectors
        │
        ▼
Repeat for each iteration
        │
        ├── Drive lattice
        ├── Generate avalanche
        ├── Update force visualisation
        ├── Update phase visualisation
        ├── Record avalanche size
        ├── Track suitable domains
        ├── Update domain synchronisation
        └── Update lattice synchronisation
        │
        ▼
Perform analysis and plotting
```

---

## Main Classes

### Parameters

Stores all simulation parameters including:

* lattice size
* threshold force
* elastic constants
* number of iterations
* synchronisation tracking settings

---

### TrackedDomain

Represents a collection of lattice sites whose synchronisation is monitored throughout the simulation.

Each tracked domain stores

* coordinates
* size
* current order parameter
* synchronisation history

---

## Visualisations

### Force lattice

Displays the force stored at each lattice site during the avalanche process.

### Phase-circle visualisation

Maps phase vectors onto the unit circle for all the sites in a domain.

Earthquake domains are expected to be maximally synchronised just after their creation, with their phase vectors spreading our over time into a random configuration about the unit circle.

---

## Outputs

The simulation records

* avalanche size time series
* synchronisation history of tracked domains
* synchronisation history of the entire lattice

These can be analysed using the plotting functions in `analysis.py`.

---

## Future Improvements

Potential future developments include

* reducing force threshold rather than driving up all sites for improved prerformance
* incremental updates of phase vectors for improved performance
* extending time series analysis to check agreement with the Gutenberg-Richter law
* implementing test of Omori's law
* further synchronisation analysis
* vectorised avalanche propagation, JIT compilation with numba, GPU version (CuPy)

```
```
