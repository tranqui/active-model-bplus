#!/usr/bin/env python3

import sys
import numpy as np
from .integrator import Model, Stencil, Integrator as BaseIntegrator

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from tqdm import tqdm
from IPython.display import display, clear_output


def plot_field(field, fig=None, axes=None, fontsize=8):
    """Show the evolving field during program execution."""

    if fig is not None:
        ax, im, cbar = axes
        im.set_data(field)
        im.set_norm(Normalize(field.min(), vmax=field.max()))
        return fig, axes

    fig = plt.figure()
    ax = plt.gca()

    im = ax.imshow(field)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=2e-2, location='right')
    cbar.set_label(r'$\phi$', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.set_xlabel('$x$', fontsize=fontsize)
    ax.set_ylabel('$y$', fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, (ax, im, cbar)


def steps_from_time(dt, tfinal):
    """Find number of steps that must be taken to achieve a total
    elapsed time.

    Args:
        dt: timestep (in simulation time units).
        tfinal: final time (in simulation time units).
                Should be a multiple of dt.

    Raises:
        ValueError: if t is not a multiple of the timestep.
    """

    nsteps = int(np.round(tfinal / dt))
    error = np.abs(nsteps - tfinal / dt)
    if not np.isclose(error, 0):
        raise ValueError(f'tfinal={tfinal} is not a multiple of dt={dt}!')

    return nsteps


def step_iterator(dt, tfinal, max_updates, min_update_time):
    """Find partitioning of simulation execution into approximately
    even steps to 
    Run the simulation for specified time.

    Args:
        t: time (in simulation time units) to run for.
            Should be a multiple of the timestep in the stencil.
        show_progress: if True, will show animation 

    Raises:
        ValueError: if t is not a multiple of the timestep.
    """

    # Minimum number of steps between updates.
    min_steps = int(np.round(min_update_time / dt))
    min_steps = max(min_steps, 1)

    # Rough heuristic to find a reasonable number of updates.
    nsteps = steps_from_time(dt, tfinal)
    nupdates = nsteps // min_steps
    nupdates = min(nupdates, max_updates)
    nupdates = max(nupdates, 1)

    # Timestep before/after each update.
    timestep = np.linspace(0, nsteps, nupdates + 1, dtype=int)
    # Infer step sizes needed to achieve these updates.
    step_size = np.diff(timestep)
    assert len(step_size) >= 1
    assert np.sum(step_size) == nsteps
    assert np.all(step_size >= 0)

    if len(step_size) > 1: return tqdm(step_size)
    else: return step_size


class Integrator(BaseIntegrator):
    def run_for_time(self, t, show_progress=False, max_updates=100, min_update_time=10):
        """Run the simulation for specified time.

        Args:
            t: time (in simulation time units) to run for.
               Should be a multiple of the timestep in the stencil.
            show_progress: if True, will show animation 

        Raises:
            ValueError: if t is not a multiple of the timestep.
        """

        nsteps = steps_from_time(self.stencil.dt, t)

        if not show_progress:
            self.run(nsteps)
            return

        initial_timestep = self.timestep
        fig, axes = plot_field(self.field)

        for steps in step_iterator(self.stencil.dt, t, max_updates, min_update_time):
            for out in [sys.stdout, sys.stderr]: out.flush()
            self.run(steps)

            # Show latest field in output (may only work in Jupyter).
            clear_output(wait=True)
            plot_field(self.field, fig, axes)
            display(fig)

        plt.close()
        assert (self.timestep - initial_timestep) == nsteps
