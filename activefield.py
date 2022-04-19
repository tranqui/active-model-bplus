#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
import sympy as sp

from interpolate import HermiteInterpolator
from ode import WeakFormProblem1d
from cache import cache, cached_property

class symbols:
    """Variables common across expressions."""

    # Dimensionality:
    d = sp.Symbol('d')

    # Geometric variables:
    x = sp.Symbol('x')
    r = sp.Symbol('r')
    droplet_radius = sp.Symbol('R')
    domain_size = sp.Symbol('L')
    interfacial_width = sp.Symbol(r'\xi')
    # Aliases:
    R = droplet_radius
    L = domain_size
    xi = interfacial_width

    # State variables:
    density = sp.Symbol('\phi')
    pseudodensity = sp.Symbol('\psi')
    pressure = sp.Symbol('p')
    pseudopressure = sp.Symbol('P')
    free_energy_density = sp.Symbol('f')
    pseudopotential = sp.Symbol('g')

    # Passive parameters in free energy functional:
    quadratic_coefficient = sp.Symbol('t')
    quartic_coefficient = sp.Symbol('u')
    square_grad_coefficient = sp.Symbol('K')
    # Aliases:
    t = quadratic_coefficient
    u = quartic_coefficient
    K = square_grad_coefficient

    # Activity parameters in Active Model-B+:
    zeta = sp.Symbol('\zeta')
    lamb = sp.Symbol('\lambda')

class Expression:
    arguments = []
    parameters = []

    def __init__(self, *args):
        """Instantiate expression with specific parameters.

        Args:
            *args: values of parameters in same order as parameters class variable.
        """
        assert len(args) is len(self.parameters)
        self.parameter_values = args

    @classmethod
    @property
    def argument(cls):
        assert len(cls.arguments) == 1
        return cls.arguments[0]

    @classmethod
    @property
    def expression(cls):
        """Symbolic expression that must be defined in derived expressions."""
        raise NotImplementedError('should only be called from derived expression!')

    @classmethod
    @property
    def expr(cls):
        """Alias for expression."""
        return cls.expression

    @classmethod
    def diff(cls, n=1):
        """Derivative of expression with respect to its arguments.

        Arg:
            n: order of derivative
        Returns:
            Derivative expression if there is a single argument, or an array-like object of
              expressions when there are multiple arguments (with entries for each derivative).
        """
        if len(cls.arguments) == 1: args = cls.arguments[0]
        else: args = cls.arguments
        expr = cls.expression.diff([args, n])
        return expr

    @classmethod
    @property
    def variables(cls):
        """Variables taken as arguments to expression in numerical evaluations (i.e. including
        parameters)."""
        return cls.arguments + cls.parameters

    @classmethod
    def numerical_implementation(cls, derivative=0):
        """Expression to pass for numerical implementation, that may differ in form from the exact
        expression in order to better handle e.g. numerical instabilities."""
        return cls.diff(derivative)

    @classmethod
    @cache
    #@disk.cache
    def compiled_function(cls, derivative=0):
        """Compile the function so that it can be numerically evaluated with native python
        structures, if not already done so for this expression.

        We cache the result (at the class-level) to prevent unnecessary recompilations for many
        instances of the same expression (with e.g. different choices of parameters).

        Args:
            derivative: order of derivative to compile into an evaluatable function
        """
        return sp.lambdify(cls.variables, cls.numerical_implementation(derivative))

    def __call__(self, *args, derivative=0):
        """Numerically evaluate expression."""
        cls = self.__class__
        f = cls.compiled_function(derivative)
        with np.errstate(invalid='ignore'):
            return f(*args + self.parameter_values)

class PseudoCoefficient(Expression):
    arguments = []
    parameters = [symbols.zeta, symbols.lamb, symbols.K]

    @classmethod
    @property
    def expression(cls):
        return (symbols.zeta - 2*symbols.lamb) / symbols.K

class Pseudodensity(Expression):
    arguments = [symbols.density]
    parameters = [symbols.zeta, symbols.lamb, symbols.K]

    @classmethod
    @property
    def expression(cls):
        return (sp.exp(PseudoCoefficient.expr*symbols.density) - 1) / PseudoCoefficient.expr

    @classmethod
    def numerical_implementation(cls, derivative=0):
        expr = cls.expr.diff(cls.argument, derivative)
        # The expression is numerically unstable as the activity coefficients cancel out,
        # so we switch to a Taylor series expansion there:
        order, threshold = 4, 1e-4
        expansion = expr.series(symbols.zeta, 2*symbols.lamb, order).removeO().simplify()
        expr = sp.Piecewise( (expansion, sp.Abs(symbols.zeta - 2*symbols.lamb) < threshold),
                             (expr, True) )
        return expr

class Density(Expression):
    arguments = [symbols.pseudodensity]
    parameters = [symbols.zeta, symbols.lamb, symbols.K]

    @classmethod
    @property
    def expression(cls):
        return sp.log(1 + PseudoCoefficient.expr*symbols.pseudodensity) / PseudoCoefficient.expr

    @classmethod
    def numerical_implementation(cls, derivative=0):
        expr = cls.expression.diff(cls.argument, derivative)
        # The expression is numerically unstable as the activity coefficients cancel out,
        # so we switch to a Taylor series expansion there:
        order, threshold = 4, 1e-4
        expansion = expr.series(symbols.zeta, 2*symbols.lamb, order).removeO().simplify()
        expr = sp.Piecewise( (expansion, sp.Abs(symbols.zeta - 2*symbols.lamb) < threshold),
                             (expr, True) )
        return expr

class Pseudopotential(Expression):
    arguments = [symbols.density]
    parameters = [symbols.zeta, symbols.lamb, symbols.K]

    @classmethod
    @property
    def free_energy_density(cls):
        raise NotImplementedError

    @property
    def f_params(self):
        return self.parameter_values[3:]
    @property
    def f(self):
        return self.free_energy_density(*self.f_params)

    @property
    def pseudodensity_params(self):
        return self.parameter_values[:3]
    @property
    def pseudodensity(self):
        return Pseudodensity(*self.pseudodensity_params)

    @classmethod
    @property
    @cache
    def expression(cls):
        phi = symbols.density
        f = cls.free_energy_density.expression
        df = f.diff(phi)
        psi = Pseudodensity.expression
        dpsi = psi.diff(phi)

        pseudopotential = sp.integrate(df*dpsi, phi)
        # Should be piecewise expression with second term case where zeta = 2*lambda (giving g=f)
        assert (pseudopotential.args[1][0] - f).simplify() == 0

        # Extract nontrivial result.
        pseudopotential = pseudopotential.args[0][0]

        # Choose integration constant so that g(0) = f(0)
        f0 = f.subs(phi, 0)
        g0 = pseudopotential.subs(phi, 0)
        pseudopotential += f0 - g0

        return pseudopotential.simplify()

    @classmethod
    def numerical_implementation(cls, derivative=0):
        expr = cls.expression.diff(cls.argument, derivative)
        # The expansion code below is unstable unless we perform change variables beforehand.
        epsilon = sp.Symbol('e')
        expr = expr.subs(symbols.zeta, 2*symbols.lamb + epsilon).simplify()
        # The expression is numerically unstable as the activity coefficients cancel out,
        # so we switch to a Taylor series expansion there:
        order, threshold = 4, 1e-4
        expansion = expr.series(epsilon, 0, 4).removeO()
        expr = sp.Piecewise( (expansion, sp.Abs(epsilon) < threshold),
                             (expr, True) )
        expr = expr.subs(epsilon, symbols.zeta - 2*symbols.lamb)
        return expr

class Phi4FreeEnergyDensity(Expression):
    arguments = [symbols.density]
    parameters = [symbols.t, symbols.u]

    @classmethod
    @property
    def expression(cls):
        return (symbols.t*symbols.density**2)/2 + (symbols.u*symbols.density**4)/4

class Phi4Pseudopotential(Pseudopotential):
    arguments = [symbols.density]
    parameters = [symbols.zeta, symbols.lamb, symbols.K, symbols.t, symbols.u]

    @classmethod
    @property
    def free_energy_density(cls):
        return Phi4FreeEnergyDensity

class ActiveModelBSphericalInterface(WeakFormProblem1d):
    argument = sp.Symbol('r')
    parameters = [symbols.droplet_radius, symbols.domain_size,
                  symbols.zeta, symbols.lamb,
                  symbols.K, symbols.t, symbols.u, symbols.d]

    @classmethod
    @property
    def analytic_solution(cls):
        raise RuntimeError('no known analytic solution to problem!')

    @classmethod
    @property
    def bulk_free_energy(cls):
        return Phi4FreeEnergyDensity.expression

    @classmethod
    @property
    def free_energy_terms(cls):
        r, d, K = cls.argument, symbols.d, symbols.K
        phi = cls.unknown_function(r)

        # Terms from bulk free energy density (Ginzburg-Landau phi4 model).
        df = cls.bulk_free_energy.diff(symbols.density)
        df = df.subs(symbols.density, phi)
        # Add square-gradient term (which is a Laplacian after the functional integral):
        df -= K * (phi.diff(r,2) + (d-1) * phi.diff(r) / r)
        return df

    @classmethod
    @property
    def local_term(cls):
        r, d, K, zeta, lamb = cls.argument, symbols.d, symbols.K, symbols.zeta, symbols.lamb
        phi = cls.unknown_function(r)
        return cls.free_energy_terms + (lamb - zeta/2) * phi.diff(r)**2

    @classmethod
    @property
    def nonlocal_term(cls):
        r, d, K, zeta, lamb = cls.argument, symbols.d, symbols.K, symbols.zeta, symbols.lamb
        phi = cls.unknown_function(r)
        return -zeta * (d-1) * phi.diff(r)**2 / r

    @classmethod
    @property
    @cache
    def strong_form(cls):
        """dmu/dr."""
        r = cls.argument
        expr = cls.local_term.diff(r) + cls.nonlocal_term
        return expr.simplify()

    @classmethod
    @property
    def test_function(cls):
        r = cls.argument
        return r**2 * cls.b(r)

    @classmethod
    @property
    @cache
    def weak_form(cls):
        r = cls.argument
        expr = -cls.test_function.diff(r) * cls.local_term + cls.nonlocal_term * cls.test_function
        #expr = (cls.local_term.diff(r) + cls.nonlocal_term) * cls.test_function
        return expr.simplify()

    @classmethod
    @property
    @cache
    def natural_boundary_condition(cls):
        expr = cls.local_term * cls.test_function
        return expr.simplify()

    # Left boundary is not a true boundary in polar coordinates.
    boundary_left = False

    @classmethod
    @property
    def boundary_conditions(cls):
        r, phi = cls.argument, cls.unknown_function

        return {(0, phi(r).diff(r), 0),
                (symbols.droplet_radius, phi(r), 0),
                (symbols.droplet_radius, cls.strong_form, 0),
                (symbols.domain_size, phi(r).diff(r), 0)}
