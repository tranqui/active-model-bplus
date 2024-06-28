#!/usr/bin/env python3

import pytest

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from activemodelbplus.integrator import Model, Stencil

def test_model():
    params = (1, 2, 3, 4, 5, 6)
    model = Model(*params)
    copy = Model(model)

    assert params == model.as_tuple()
    assert copy is not model
    assert copy.as_tuple() == model.as_tuple()
    assert copy.as_dict() == model.as_dict()

def test_stencil():
    params = (1, 2, 3)
    stencil = Stencil(*params)
    copy = Stencil(stencil)

    assert params == stencil.as_tuple()
    assert copy is not stencil
    assert copy.as_tuple() == stencil.as_tuple()
    assert copy.as_dict() == stencil.as_dict()