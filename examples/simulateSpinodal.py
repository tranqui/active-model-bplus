#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from activemodelbplus.integrator import Model, Stencil, Integrator

# Parameters for normal equilibrium spinodal decomposition.
stencil = Stencil(dt=1e-2, dx=1, dy=1)
model = Model(a=-0.25, b=0, c=0.25, kappa=1, lamb=0, zeta=0, T=0)
phi0 = 0.

Nx, Ny = 256, 256
initial = 1e-1 * np.random.random((Ny, Nx))
initial += phi0 - np.average(initial)

tfinal = 1e3
sim = Integrator(initial, stencil, model)
sim.run_for_time(tfinal, show_progress=True, max_updates=10)
