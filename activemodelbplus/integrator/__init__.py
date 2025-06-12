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

    nsteps = np.asarray(np.round(tfinal / dt), dtype=int)
    error = np.abs(nsteps - tfinal / dt)
    if not np.all(np.isclose(error, 0)):
        raise ValueError(f'tfinal={tfinal} is not a multiple of dt={dt}!')

    return nsteps.item() if np.isscalar(nsteps) else nsteps


def step_iterator(dt, tfinal, max_updates, min_update_time,
                  t_interrupt=None):
    """Find partitioning of simulation execution into approximately
    even steps to provide updates of simulation progress.

    Args:
        dt: timestep (in simulation time units).
        tfinal: final time (in simulation time units).
                Should be a multiple of dt.
        t_interrupt: times (in simulation time units) to stop at (e.g. to
                      sample simulation state). Should be multiple of dt.

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

    # Make sure we also stop on any interruption times.
    if t_interrupt is not None:
        t_interrupt = np.asarray(t_interrupt).reshape(-1)
        interrupts = steps_from_time(dt, t_interrupt)
        timestep = np.concatenate([timestep, interrupts])
        timestep = np.unique(timestep)

    # Infer step sizes needed to achieve these updates.
    step_size = np.diff(timestep)
    assert len(step_size) >= 1
    assert np.sum(step_size) == nsteps
    assert np.all(step_size >= 0)

    return step_size

class Integrator(BaseIntegrator):
    def run_for_time(self, t, show_progress=False,
                     t_interrupt=None, f_interrupt=None,
                     max_updates=100, min_update_time=10):
        """Run the simulation for specified time.

        Args:
            t: time (in simulation time units) to run for.
               Should be a multiple of the timestep in the stencil.
            show_progress: if True, will show animation.
            t_interrupt: times (in simulation time units) to stop at (e.g. to
                         sample simulation state). Should be multiple of
                         timestep in the stencil.
            f_interrupt: function to call during interruption to take samples.

        Raises:
            ValueError: if t is not a multiple of the timestep.
        """

        nsteps = steps_from_time(self.stencil.dt, t)

        if t_interrupt is not None:
            assert f_interrupt is not None
            if np.any(t_interrupt < 0) or np.any(t_interrupt > t):
                raise ValueError('interrupt outside sim time window!')

        initial_timestep = self.timestep
        initial_time = self.time
        if show_progress: fig, axes = plot_field(self.field)

        steps = step_iterator(self.stencil.dt, t,
                              max_updates, min_update_time,
                              t_interrupt=t_interrupt)
        if show_progress: steps = tqdm(steps)

        for n in steps:
            if show_progress:
                for out in [sys.stdout, sys.stderr]: out.flush()
    
            if t_interrupt is not None:
                if np.any(np.isclose(self.time - initial_time, t_interrupt)):
                    f_interrupt(self)

            assert n > 0
            self.run(n)

            if show_progress:
                # Show latest field in output (may only work in Jupyter).
                clear_output(wait=True)
                plot_field(self.field, fig, axes)
                display(fig)

        if show_progress: plt.close()
        assert (self.timestep - initial_timestep) == nsteps

        if t_interrupt is not None:
            if np.any(np.isclose(self.time - initial_time, t_interrupt)):
                f_interrupt(self)
