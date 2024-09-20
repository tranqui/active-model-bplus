#!/usr/bin/env python3

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from activemodelbplus.integrator import Model, Stencil, Integrator

def test_model():
    params = (1, 2, 3, 4, 5, 6, 7)
    model = Model(*params)
    copy = Model(model)

    assert params == model.as_tuple()
    assert model.a == params[0]
    assert model.b == params[1]
    assert model.c == params[2]
    assert model.kappa == params[3]
    assert model.lamb == params[4]
    assert model.zeta == params[5]
    assert model.T == params[6]
    assert copy is not model
    assert copy == model
    assert copy.as_tuple() == model.as_tuple()
    assert copy.as_dict() == model.as_dict()

    copy.a += 1
    assert copy != model
    assert copy.as_tuple() != model.as_tuple()
    assert copy.as_dict() != model.as_dict()

def test_stencil():
    params = (1, 2, 3)
    stencil = Stencil(*params)
    copy = Stencil(stencil)

    assert params == stencil.as_tuple()
    assert stencil.dt == params[0]
    assert stencil.dx == params[1]
    assert stencil.dy == params[2]
    assert copy is not stencil
    assert copy == stencil
    assert copy.as_tuple() == stencil.as_tuple()
    assert copy.as_dict() == stencil.as_dict()

    copy.dx += 1
    assert copy.as_tuple() != stencil.as_tuple()
    assert copy.as_dict() != stencil.as_dict()

def test_integrator():
    stencil = Stencil(1e-2, 1, 1)
    model = Model(0.25, 0.5, 0.25, 1, 1, 1, 0)

    Nx, Ny = 256, 128
    initial = np.random.random((Ny, Nx))

    sim = Integrator(initial, stencil, model)
    assert (sim.field == initial).all()
    assert sim.stencil is not stencil
    assert sim.model is not model
    assert sim.stencil == stencil
    assert model == model
    assert sim.stencil.as_tuple() == stencil.as_tuple()
    assert sim.model.as_tuple() == model.as_tuple()

    assert sim.timestep == 0
    sim.run(1)
    assert (sim.field != initial).any()
    assert sim.timestep == 1
    assert sim.time == stencil.dt
