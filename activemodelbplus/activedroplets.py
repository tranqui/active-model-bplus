#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
from scipy.optimize import newton_krylov
from scipy.special import erf, erfinv
import sympy as sp

from .interpolate import HermiteInterpolator
from .activefield import symbols, Phi4Pseudopotential, ActiveModelBPlanarInterface, ActiveModelBSphericalInterface
from .cache import cache, cached_property, disk_cache

class ProbabilityDistribution:
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
        return sp.Symbol('x')

    @classmethod
    @property
    def probability_density_function(cls):
        raise NotImplementedError

    @classmethod
    @property
    def cumulative_distribution_function(cls):
        raise NotImplementedError

    @classmethod
    @property
    def inverse_distribution_function(cls):
        raise NotImplementedError

    # Aliases.

    def call(self, f, x):
        f = f.subs({p: val for p, val in zip(self.parameters, self.parameter_values)})
        f = sp.lambdify(self.argument, f)
        return f(x)

    def pdf(self, x):
        return self.call(self.probability_distribution_function, x)

    def cdf(self, x):
        return self.call(self.cumulative_distribution_function, x)

    def idf(self, x):
        return self.call(self.inverse_distribution_function, x)

class LogNormalDistribution(ProbabilityDistribution):
    mu, sigma = sp.symbols('mu sigma')
    parameters = [mu, sigma]

    @classmethod
    @property
    def probability_density_function(cls):
        x = cls.argument
        return sp.exp( (sp.log(x) - cls.mu)**2 / (2*cls.sigma**2) ) / (x*cls.sigma*sp.sqrt(2*sp.pi))

    @classmethod
    @property
    def cumulative_distribution_function(cls):
        x = cls.argument
        return 0.5 * (1 + sp.erf( (sp.log(x) - cls.mu) / (sp.sqrt(2)*cls.sigma) ) )

    @classmethod
    @property
    def inverse_distribution_function(cls):
        x = cls.argument
        return sp.exp(cls.mu + (sp.sqrt(2)*cls.sigma) * sp.erfinv(2*x - 1))

class SechDistribution(ProbabilityDistribution):
    mu, xi = sp.symbols('mu sigma')
    parameters = [mu, xi]

    @classmethod
    @property
    def probability_density_function(cls):
        x = cls.argument
        return sp.sech( (x - cls.mu) / cls.xi ) / (sp.pi * cls.xi / 2)

    @classmethod
    @property
    def cumulative_distribution_function(cls):
        x = cls.argument
        return ( 1 + 4*sp.atan( sp.tanh( (x - cls.mu) / (2*cls.xi) ) )/sp.pi ) / 2

    @classmethod
    @property
    def inverse_distribution_function(cls):
        x = cls.argument
        return cls.mu + 2*cls.xi * sp.atanh( sp.tan(sp.pi * (2*x - 1) / 4) )

class SechSquaredDistribution(ProbabilityDistribution):
    mu, xi = sp.symbols('mu sigma')
    parameters = [mu, xi]

    @classmethod
    @property
    def probability_density_function(cls):
        x = cls.argument
        tanh_const = sp.tanh(cls.mu / cls.xi)
        return sp.sech( (x - cls.mu) / cls.xi )**2 / (cls.xi * (1 + tanh_const))

    @classmethod
    @property
    def cumulative_distribution_function(cls):
        x = cls.argument
        tanh_const = sp.tanh(cls.mu / cls.xi)
        return (sp.tanh( (x - cls.mu) / cls.xi ) + tanh_const) / (1 + tanh_const)

    @classmethod
    @property
    def inverse_distribution_function(cls):
        x = cls.argument
        tanh_const = sp.tanh(cls.mu / cls.xi)
        return cls.mu + cls.xi * sp.atanh(x - tanh_const*(1 - x))

class ActiveBinodal(HermiteInterpolator):
    @classmethod
    @property
    def guess_profile_distribution(cls):
        return SechDistribution

    @classmethod
    @cache
    @disk_cache
    def guess_profile_function(cls, deriv=0):
        f = cls.guess_profile_distribution
        phi0, phi1 = sp.symbols('phi0, phi1')
        guess = phi0 + (phi1 - phi0) * f.cumulative_distribution_function
        arguments = [f.argument, phi0, phi1] + f.parameters
        return sp.lambdify(arguments, guess.diff(f.argument, deriv))

    @classmethod
    def guess_profile(cls, field_theory, phi0, phi1, domain_size=None, order=4,
                      xsample=None, npoints=101, interfacial_width=None, **kwargs):
        """Possible kwargs are arguments to ActiveDroplet.__init__."""
        if interfacial_width is None: interfacial_width = field_theory.passive_bulk_interfacial_width
        if domain_size is None: domain_size = 50*interfacial_width

        if xsample is None:
            xdist = SechDistribution(0, interfacial_width)

            l = 0.5*domain_size
            x0 = np.linspace(xdist.cdf(-l), xdist.cdf(l), npoints)
            xsample = xdist.idf(x0)
            # Correct for any numerical errors at the boundary.
            xsample[0] = -l
            xsample[-1] = l

        weights = np.zeros((len(xsample), order))
        for deriv in range(order):
            with np.errstate(invalid='ignore', divide='ignore'):
                f = cls.guess_profile_function(deriv)
                weights[:,deriv] = f(xsample, phi0, phi1, *xdist.parameter_values)
        weights[~np.isfinite(weights)] = 0

        return xsample, weights

    @classmethod
    def from_guess(cls, field_theory, *args, **kwargs):
        xsample, weights = cls.guess_profile(field_theory, *args, **kwargs)
        drop = cls(field_theory, xsample, weights, **kwargs)
        return drop

    def __init__(self, field_theory, *args, print_updates=None, **kwargs):
        """Possible kwargs are arguments to ode.WeakFormProblem1d.__init__."""
        self.field_theory = field_theory
        super().__init__(*args, **kwargs)
        self.solve(print_updates=print_updates, **kwargs)

    def __repr__(self):
        return '<ActiveBinodal zeta=%g lamb=%g phi=[%.4f->%.4f] nnodes=%d>' % (self.field_theory.zeta, self.field_theory.lamb, self.phi0, self.phi1, len(self.x))

    @property
    def summary(self):
        P0, P1 = self.pseudopressure0, self.pseudopressure1
        return 'zeta=%g lamb=%g: phi=[%.4f->%.4f] P=[%.4f->%.4f] dP=%.4f nnodes=%d' % (self.field_theory.zeta, self.field_theory.lamb, self.phi0, self.phi1, P0, P1, P0-P1, len(self.x))

    @property
    def ode(self):
        constant_params = tuple(self.field_theory.pseudopotential.parameter_values + (self.field_theory.d,))
        return ActiveModelBPlanarInterface(self.domain_size, *constant_params)

    @property
    def interface_location(self):
        return 0

    def solve(self, *args, **kwargs):
        with np.errstate(all='raise'):
            self.weights = self.ode.solve(self.x, self.weights, *args, **kwargs)
        assert np.sign(self(self.interface_location, derivative=1)) == np.sign(self.phi1 - self.phi0)

    @property
    def domain_size(self):
        return self.x[-1] - self.x[0]

    def refine(self, refinement_tol=1e-6, max_refinement_iters=None, nrefinement_iters=0,
               max_points=int(1e4), **kwargs):
        """Possible kwargs are arguments to ActiveDroplet.__init__."""
        if max_refinement_iters and nrefinement_iters >= max_refinement_iters:
            raise RuntimeError('result not converging after %d mesh refinement iterations!' % niters)

        if refinement_tol is np.inf: return self

        ode = self.ode
        h = ode.strong_form
        error_estimator = h**2
        parameters = {p: v for p, v in zip(ode.parameters, ode.parameter_values)}

        #I = self.analytic_integral(error_estimator, f, x)
        I = self.numerical_integral(error_estimator, ode.unknown_function, ode.argument, parameters)
        errors = np.abs(np.diff(I.weights[:,0]))

        if np.any(errors > refinement_tol):
            elements_to_split = errors > refinement_tol
            element_midpoints = 0.5*(self.x[1:] + self.x[:-1])
            new_x = element_midpoints[elements_to_split]
            new_x = np.sort( np.concatenate((I.x, new_x)) )

            if new_x.size > max_points:
                raise RuntimeError('possible singular behaviour - trying to create too many points (%d) during mesh refinement!' % len(new_x))

            _, order = self.weights.shape
            new_weights = np.empty((new_x.size, order))
            for c in range(order):
                new_weights[:,c] = self(new_x, c)

            self.global_nodes = new_x
            self.weights = new_weights
            self.refine(refinement_tol, max_refinement_iters, nrefinement_iters+1, max_points, **kwargs)

    @property
    def d(self):
        return self.field_theory.d

    @property
    def pseudopotential(self):
        return self.field_theory.pseudopotential

    @property
    def phi0(self):
        return self.weights[0,0]

    @property
    def phi1(self):
        return self.weights[-1,0]

    def pseudopressure(self, x):
        phi = self(x)
        return self.field_theory.bulk_pseudopressure(phi)

    @property
    def pseudopressure0(self):
        return self.pseudopressure(self.x[0])

    @property
    def pseudopressure1(self):
        return self.pseudopressure(self.x[-1])

    @property
    def pseudopressure_drop(self):
        return self.pseudopressure0 - self.pseudopressure1

    @cached_property
    def mu(self):
        x, phi = ode.argument, ode.unknown_function
        ode = self.ode
        mu = ode.mu_term
        mu = mu.subs({p: v for p, v in zip(ode.parameters, ode.parameter_values)})
        return self.evaluate(mu, phi, x, order=1)

    @property
    def mu0(self):
        return self.mu(self.x[0])

    @property
    def mu1(self):
        return self.mu(self.x[-1])

    @classmethod
    @property
    @cache
    def surface_tension_integrand_expression(cls):
        phi = sp.Function('\phi')
        x = symbols.x
        zeta, lamb, K = symbols.zeta, symbols.lamb, symbols.K

        exp_factor = (zeta - 2*lamb) / K
        s0_integrand = zeta * phi(x).diff(x)**2
        s1_integrand = -2*lamb * sp.exp( phi(x) * exp_factor ) * phi(x).diff(x)**2
        integrand = (s0_integrand + s1_integrand) / exp_factor

        # The expression is numerically unstable as the activity coefficients cancel out,
        # so we switch to a Taylor series expansion there:
        order, threshold = 4, 1e-4
        expansion = integrand.series(symbols.zeta, 2*symbols.lamb, order).removeO().simplify()
        integrand = sp.Piecewise( (expansion, sp.Abs(symbols.zeta - 2*symbols.lamb) < threshold),
                                  (integrand, True) )

        return integrand

    @classmethod
    @property
    @cache
    def surface_tension_integrand_expression2(cls):
        phi = sp.Function('\phi')
        x = symbols.x
        zeta, lamb, K = symbols.zeta, symbols.lamb, symbols.K
        phi0 = sp.Symbol('\phi0')

        exp_factor = (zeta - 2*lamb) / K
        s0_integrand = zeta * sp.exp( phi0 * exp_factor ) * phi(x).diff(x)**2
        s1_integrand = -2*lamb * sp.exp( phi(x) * exp_factor ) * phi(x).diff(x)**2
        integrand = (s0_integrand + s1_integrand) / exp_factor

        # The expression is numerically unstable as the activity coefficients cancel out,
        # so we switch to a Taylor series expansion there:
        order, threshold = 4, 1e-4
        expansion = integrand.series(symbols.zeta, 2*symbols.lamb, order).removeO().simplify()
        integrand = sp.Piecewise( (expansion, sp.Abs(symbols.zeta - 2*symbols.lamb) < threshold),
                                  (integrand, True) )

        return sp.Lambda(phi0, integrand)

    @property
    def surface_tension_integrand(self):
        integrand = self.surface_tension_integrand_expression
        integrand = integrand.subs({p: v for p, v in zip(self.pseudopotential.parameters,
                                                         self.pseudopotential.parameter_values)})
        # Child droplet class will need to substitute their radius in also.
        try: integrand = integrand.subs(symbols.droplet_radius, self.R)
        except: pass
        return integrand

    @property
    def surface_tension_integrand2(self):
        integrand = self.surface_tension_integrand_expression2(self.phi0)
        integrand = integrand.subs({p: v for p, v in zip(self.pseudopotential.parameters,
                                                         self.pseudopotential.parameter_values)})
        # Child droplet class will need to substitute their radius in also.
        try: integrand = integrand.subs(symbols.droplet_radius, self.R)
        except: pass
        return integrand

    @property
    def surface_tension(self):
        integrand = self.surface_tension_integrand
        phi, x = sp.Function('\phi'), symbols.x
        return self.integrate(integrand, phi, x)

    @property
    def surface_tension2(self):
        integrand = self.surface_tension_integrand2
        phi, x = sp.Function('\phi'), symbols.x
        return self.integrate(integrand, phi, x)

class ActiveDroplet(ActiveBinodal):
    @classmethod
    def guess_profile(cls, field_theory, R, phi0, phi1, domain_size=None, order=4,
                      xsample=None, npoints=101, interfacial_width=None, **kwargs):
        """Possible kwargs are arguments to ActiveDroplet.__init__."""
        if domain_size is None: domain_size = 5*R
        if interfacial_width is None: interfacial_width = field_theory.passive_bulk_interfacial_width

        if xsample is None:
            xdist = SechDistribution(R, interfacial_width)

            # Generate points heavily distributed around the boundary by the above distribution.
            l = min(10*interfacial_width, R)
            ncentral_points = (9*npoints)//10
            npoints_left = (ncentral_points+1)//2
            npoints_right = ncentral_points - npoints_left
            x0 = np.concatenate((np.linspace(xdist.cdf(R-l), xdist.cdf(R), npoints_left),
                                 np.linspace(xdist.cdf(R), xdist.cdf(R+l), npoints_right+1)[1:]))
            xsample = xdist.idf(x0)
            xsample[0] = R-l
            xsample[-1] = R+l

            # Generate some points left and right of the boundary region.
            if l == R: npoints_left = 0
            else: npoints_left = (npoints - ncentral_points)//2
            npoints_right = npoints - ncentral_points - npoints_left
            xleft = np.linspace(0, xsample[0], npoints_left+1)[:-1]
            xright = np.linspace(xsample[-1], domain_size, npoints_right+1)[1:]
            xsample = np.concatenate((xleft, xsample, xright))

        weights = np.zeros((len(xsample), order))
        for deriv in range(order):
            with np.errstate(invalid='ignore', divide='ignore'):
                f = cls.guess_profile_function(deriv)
                weights[:,deriv] = f(xsample, phi0, phi1, *xdist.parameter_values)
        weights[~np.isfinite(weights)] = 0

        return xsample, weights

    @classmethod
    def from_guess(cls, field_theory, R, *args, **kwargs):
        xsample, weights = cls.guess_profile(field_theory, R, *args, **kwargs)
        drop = cls(field_theory, R, xsample, weights, **kwargs)
        return drop

    def __init__(self, field_theory, R, *args, **kwargs):
        """Possible kwargs are arguments to ode.WeakFormProblem1d.__init__."""
        self.R = R
        super().__init__(field_theory, *args, **kwargs)

    def __repr__(self):
        return '<ActiveDroplet zeta=%g lamb=%g d=%d R=%g nnodes=%d>' % (self.field_theory.zeta, self.field_theory.lamb, self.d, self.R, len(self.x))

    @property
    def summary(self):
        P0, P1 = self.pseudopressure0, self.pseudopressure1
        return 'zeta=%g lamb=%g R=%g d=%d: phi=[%.4f->%.4f] P=[%.4f->%.4f] dP=%.4f nnodes=%d' % (self.field_theory.zeta, self.field_theory.lamb, self.R, self.field_theory.d, self.phi0, self.phi1, P0, P1, P0-P1, len(self.x))

    @property
    def ode(self):
        constant_params = tuple(self.field_theory.pseudopotential.parameter_values + (self.field_theory.d,))
        return ActiveModelBSphericalInterface(self.R, self.domain_size, *constant_params)

    @property
    def interface_location(self):
        return self.R

    @property
    def domain_size(self):
        return self.x[-1]

    @classmethod
    @property
    def nonlocal_integrand_expression(cls):
        phi = sp.Function('\phi')
        r = symbols.r
        zeta, lamb, K, d = symbols.zeta, symbols.lamb, symbols.K, symbols.d

        integrand = zeta * (d-1) * phi(r).diff(r)**2 / r
        return integrand, phi, r

    @cached_property
    def mu(self):
        integrand, phi, r = self.nonlocal_integrand_expression

        ode = self.ode
        local_part = ode.local_term
        local_part = local_part.subs({p: v for p, v in zip(ode.parameters, ode.parameter_values)})
        local_part = local_part.subs({ode.argument: r, ode.unknown_function: phi})
        local_part = self.evaluate(local_part, phi, r, order=1, singularity_at_origin=True)

        integrand = integrand.subs({p: v for p, v in zip(ode.parameters, ode.parameter_values)})
        nonlocal_part = self.numerical_integral(integrand, phi, r)
        # Reverse order of integration so it measures the change in mu from r->\infty
        nonlocal_part.weights[:,0] = nonlocal_part.weights[-1,0] - nonlocal_part.weights[:,0]

        assert local_part.x.size == nonlocal_part.x.size
        assert local_part.weights.shape == nonlocal_part.weights.shape
        return HermiteInterpolator(local_part.x, local_part.weights + nonlocal_part.weights)

    @property
    def mu0(self):
        return self.mu(self.x[0])

    @property
    def mu1(self):
        return self.mu(self.x[-1])

    @classmethod
    @property
    @cache
    def surface_tension_integrand_expression(cls):
        phi = sp.Function('\phi')
        r = symbols.x
        R = symbols.droplet_radius
        zeta, lamb, K = symbols.zeta, symbols.lamb, symbols.K

        exp_factor = (zeta - 2*lamb) / K
        s0_integrand = zeta * phi(r).diff(r)**2 * (R / r)
        s1_integrand = -2*lamb * sp.exp( phi(r) * exp_factor ) * phi(r).diff(r)**2 * (R / r)
        integrand = (s0_integrand + s1_integrand) / exp_factor

        # The expression is numerically unstable as the activity coefficients cancel out,
        # so we switch to a Taylor series expansion there:
        order, threshold = 4, 1e-4
        expansion = integrand.series(symbols.zeta, 2*symbols.lamb, order).removeO().simplify()
        integrand = sp.Piecewise( (expansion, sp.Abs(symbols.zeta - 2*symbols.lamb) < threshold),
                                  (integrand, True) )

        return integrand

    @classmethod
    @property
    @cache
    def surface_tension_integrand_expression2(cls):
        phi = sp.Function('\phi')
        r = symbols.x
        R = symbols.droplet_radius
        zeta, lamb, K = symbols.zeta, symbols.lamb, symbols.K
        phi0 = sp.Symbol('\phi0')

        exp_factor = (zeta - 2*lamb) / K
        s0_integrand = zeta * sp.exp( phi0 * exp_factor ) * phi(r).diff(r)**2 * (R / r)
        s1_integrand = -2*lamb * sp.exp( phi(r) * exp_factor ) * phi(r).diff(r)**2 * (R / r)
        integrand = (s0_integrand + s1_integrand) / exp_factor

        # The expression is numerically unstable as the activity coefficients cancel out,
        # so we switch to a Taylor series expansion there:
        order, threshold = 4, 1e-4
        expansion = integrand.series(symbols.zeta, 2*symbols.lamb, order).removeO().simplify()
        integrand = sp.Piecewise( (expansion, sp.Abs(symbols.zeta - 2*symbols.lamb) < threshold),
                                  (integrand, True) )

        return sp.Lambda(phi0, integrand)

    @property
    def surface_tension_pseudopressure_drop(self):
        return (self.d-1) * self.surface_tension / self.R

    @property
    def surface_tension_pseudopressure_drop2(self):
        return (self.d-1) * self.surface_tension2 / self.R

class ActiveModelBPlus:
    @classmethod
    def phi4(cls, zeta, lamb, K=1, t=-0.25, u=0.25, *args, **kwargs):
        return Phi4Pseudopotential(zeta, lamb, K, t, u, *args)

    def __init__(self, *args, d=3, **kwargs):
        self.d = d
        self.pseudopotential = self.phi4(*args, **kwargs)
        self.free_energy_density = self.pseudopotential.f
        self.pseudodensity = self.pseudopotential.pseudodensity

    def __repr__(self):
        return '<ActiveModelBPlus zeta=%g lamb=%g>' % (self.zeta, self.lamb)

    @property
    def zeta(self):
        return self.pseudopotential.parameter_values[0]

    @property
    def lamb(self):
        return self.pseudopotential.parameter_values[1]

    @property
    def K(self):
        return self.pseudopotential.parameter_values[2]

    @property
    def t(self):
        return self.pseudopotential.parameter_values[3]

    @property
    def u(self):
        return self.pseudopotential.parameter_values[4]

    @property
    def parameters(self):
        params = {p: v for p, v in zip(self.pseudopotential.parameters, self.pseudopotential.parameter_values)}
        params[symbols.d] = self.d
        return params

    @property
    def passive_bulk_interfacial_width(self):
        return np.sqrt(-2*self.K/self.t)

    def bulk_pseudopressure(self, phi):
        return self.pseudodensity(phi)*self.free_energy_density(phi, derivative=1) - self.pseudopotential(phi)

    @property
    def binodal_densities(self, eps=1e-12):
        phi_guess = np.array([1, -1])
        if np.abs(self.zeta - 2*self.lamb) < eps:
            return phi_guess

        P = self.bulk_pseudopressure
        mu = lambda phi: self.free_energy_density(phi, derivative=1)

        residual = lambda x: (P(x[0]) - P(x[1]), mu(x[0]) - mu(x[1]))
        phi = newton_krylov(residual, phi_guess)

        return phi

    def binodal(self, domain_size=None, guess=None, **kwargs):
        phi0, phi1 = self.binodal_densities

        if guess is None:
            profile = ActiveBinodal.from_guess(self, phi0, phi1, domain_size=domain_size, **kwargs)
        else:
            profile = ActiveBinodal(self, R, guess.x, guess.weights, **kwargs)

        profile.refine(**kwargs)
        return profile

    def droplet(self, R, phi0=None, phi1=None, domain_size=None, guess=None, **kwargs):
        """Possible kwargs are arguments to ActiveDroplet.__init__ and ActiveDroplet.refine."""
        bulk_phi = self.binodal_densities
        if phi1 is None: phi1 = bulk_phi[1]
        if phi0 is None: phi0 = bulk_phi[np.argmax(np.abs(bulk_phi - phi1))]

        if guess is None:
            drop = ActiveDroplet.from_guess(self, R, phi0, phi1, domain_size=domain_size, **kwargs)
        else:
            guess_x = guess.x * R / guess.R
            drop = ActiveDroplet(self, R, guess_x, guess.weights, **kwargs)

        drop.refine(**kwargs)
        return drop

def binodal_densities(zeta_lamb, *args, **kwargs):
    """
    Args:
        zeta_lamb: the parameter $\zeta - 2*\lambda$.
    Returns:
        Phi0 and phi1 or a sequence of (phi0,phi1) values for each zeta_lamb parameter.
    """
    zeta = lambda x: 1
    lamb = lambda x: 0.5*(zeta(x) - x)
    phi = lambda x: tuple(ActiveModelBPlus(zeta(x), lamb(x), *args, **kwargs).binodal_densities)
    phi = np.vectorize(phi)
    return phi(zeta_lamb)
