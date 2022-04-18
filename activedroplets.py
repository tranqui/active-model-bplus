#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
from scipy.optimize import newton_krylov
from scipy.special import erf, erfinv
import sympy as sp

from interpolate import HermiteInterpolator
from activefield import symbols, Phi4Pseudopotential, ActiveModelBSphericalInterface

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

class ActiveDroplet(HermiteInterpolator):
    @classmethod
    def from_guess(cls, field_theory, R, phi0, phi1, domain_size=None, order=4,
                   xsample=None, npoints=51, interfacial_width=None, **kwargs):
        """Possible kwargs are arguments to ActiveDroplet.__init__."""
        if domain_size is None: domain_size = 5*R
        if interfacial_width is None: interfacial_width = field_theory.passive_bulk_interfacial_width

        if xsample is None:
            xdist = SechDistribution(R, interfacial_width)
            #xdist = SechSquaredDistribution(R, interfacial_width)
            #xdist = LogNormalDistribution(np.log(R), 0.25)

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

        guess = phi0 + (phi1 - phi0) * xdist.cumulative_distribution_function
        guess = guess.subs({p: v for p, v in zip(xdist.parameters, xdist.parameter_values)})

        weights = np.zeros((len(xsample), order))
        for c in range(order):
            f = guess.diff(xdist.argument, c)
            f = sp.lambdify(xdist.argument, f)
            with np.errstate(invalid='ignore', divide='ignore'):
                weights[:,c] = sp.lambdify(xdist.argument, guess.diff(xdist.argument, c))(xsample)
        weights[~np.isfinite(weights)] = 0

        drop = cls(field_theory, R, xsample, weights, **kwargs)
        return drop

    def __init__(self, field_theory, R, *args, phi1=None, print_updates=None, **kwargs):
        """Possible kwargs are arguments to ode.WeakFormProblem1d.__init__."""
        self.field_theory = field_theory
        self.R = R
        super().__init__(*args, **kwargs)
        self.solve(print_updates=print_updates, **kwargs)

    def __repr__(self):
        return '<ActiveDroplet zeta=%g lamb=%g R=%g nnodes=%d>' % (self.field_theory.zeta, self.field_theory.lamb, self.R, len(self.x))

    @property
    def summary(self):
        P0, P1 = self.pseudopressure0, self.pseudopressure1
        mu0, mu1 = self.mu0, self.mu1
        S = self.pseudopressure_drop
        return 'zeta=%g lamb=%g R=%g: phi=[%.4f->%.4f] mu=[%.4f->%.4f] P=[%.4f->%.4f] S=%.4f dP=%.4f dmu=%.4g dP-S=%.4g nnodes=%d' % (self.field_theory.zeta, self.field_theory.lamb, self.R, self.phi0, self.phi1, mu0, mu1, P0, P1, S, P0-P1, mu1-mu0, P0-P1-S, len(self.x))

    @property
    def ode(self):
        constant_params = tuple(self.field_theory.pseudopotential.parameter_values + (self.field_theory.d,))
        return ActiveModelBSphericalInterface(self.R, self.domain_size, *constant_params)

    def solve(self, *args, **kwargs):
        with np.errstate(all='raise'):
            self.weights = self.ode.solve(self.x, self.weights, *args, **kwargs)
        assert np.sign(self(self.R, derivative=1)) == np.sign(self.phi1 - self.phi0)

    @property
    def domain_size(self):
        return self.x[-1]

    def refine(self, refinement_tol=1e-6, max_refinement_iters=None, nrefinement_iters=0,
               max_points=int(1e6), **kwargs):
        """Possible kwargs are arguments to ActiveDroplet.__init__."""
        if max_refinement_iters and nrefinement_iters >= max_refinement_iters:
            raise RuntimeError('result not converging after %d mesh refinement iterations!' % niters)

        if refinement_tol is np.inf: return self

        f = sp.Function('f')
        x = symbols.x

        ode = self.ode
        h = ode.strong_form
        h = h.subs({p: v for p, v in zip(ode.parameters, ode.parameter_values)})
        h = h.subs({ode.argument: x, ode.unknown_function: f})
        error_estimator = h**2

        #I = self.analytic_integral(error_estimator, f, x)
        I = self.numerical_integral(error_estimator, f, x)
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

            drop = ActiveDroplet(self.field_theory, self.R, new_x, new_weights, **kwargs)
            return drop.refine(refinement_tol, max_refinement_iters, nrefinement_iters+1, max_points, **kwargs)

        return self

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

    @classmethod
    @property
    def nonlocal_integrand_expression(cls):
        phi = sp.Function('\phi')
        r = symbols.r
        zeta, lamb, K, d = symbols.zeta, symbols.lamb, symbols.K, symbols.d

        integrand = zeta * (d-1) * phi(r).diff(r)**2 / r
        return integrand, phi, r

    @property
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
        integrand, phi, r = self.nonlocal_integrand_expression
        integrand = integrand.subs({p: v for p, v in zip(self.pseudopotential.parameters,
                                                         self.pseudopotential.parameter_values)})
        integrand = integrand.subs(symbols.d, self.d)

        local_part = self.field_theory.free_energy_density(self.phi0, derivative=1) - self.field_theory.K*self(self.x[0],2)
        nonlocal_part = self.integrate(integrand, phi, r)
        return local_part + nonlocal_part

    @property
    def mu1(self):
        return self.field_theory.free_energy_density(self.phi1, derivative=1) - self.field_theory.K*self(self.x[-1],2)

    @classmethod
    @property
    def surface_tension_integrand_expression(cls):
        phi = sp.Function('\phi')
        r = symbols.r
        zeta, lamb, K = symbols.zeta, symbols.lamb, symbols.K
        phi0 = sp.Symbol('\phi0')

        s0_integrand = zeta * sp.exp( phi0 * (zeta - 2*lamb)/K ) * phi(r).diff(r)**2 / r
        s1_integrand = -2*lamb * sp.exp( phi(r) * (zeta - 2*lamb)/K ) * phi(r).diff(r)**2 / r
        integrand = (symbols.d - 1) * (s0_integrand + s1_integrand) * K / (zeta - 2*lamb)

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
        integrand = integrand.subs(symbols.d, self.d)
        return integrand

    @property
    def pseudopressure_drop(self):
        integrand = self.surface_tension_integrand(self.phi0)
        phi, r = sp.Function('\phi'), symbols.r
        return self.integrate(integrand, phi, r)

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
    def bulk_binodals(self, eps=1e-12):
        phi_guess = np.array([-1, 1])
        if np.abs(self.zeta - 2*self.lamb) < eps:
            return phi_guess

        P = self.bulk_pseudopressure
        mu = lambda phi: self.free_energy_density(phi, derivative=1)

        residual = lambda x: (P(x[0]) - P(x[1]), mu(x[0]) - mu(x[1]))
        phi = newton_krylov(residual, phi_guess)

        return phi

    def droplet(self, R, phi0=None, phi1=None, domain_size=None, guess=None, **kwargs):
        """Possible kwargs are arguments to ActiveDroplet.__init__ and ActiveDroplet.refine."""
        bulk_phi = self.bulk_binodals
        if phi1 is None: phi1 = bulk_phi[0]
        if phi0 is None: phi0 = bulk_phi[np.argmax(np.abs(bulk_phi - phi1))]

        if guess is None:
            drop = ActiveDroplet.from_guess(self, R, phi0, phi1, domain_size=domain_size, **kwargs)
        else:
            guess_x = guess.x * R / guess.R
            drop = ActiveDroplet(self, R, guess_x, guess.weights, **kwargs)

        return drop.refine(**kwargs)

def bulk_binodals(zeta_lamb, *args, **kwargs):
    """
    Args:
        zeta_lamb: the parameter $\zeta - 2*\lambda$.
    Returns:
        Phi1 and phi2 or a sequence of (phi1,phi2) values for each zeta_lamb parameter.
    """
    zeta = lambda x: 1
    lamb = lambda x: 0.5*(zeta(x) - x)
    phi = lambda x: tuple(ActiveModelBPlus(zeta(x), lamb(x), *args, **kwargs).bulk_binodals)
    phi = np.vectorize(phi)
    return phi(zeta_lamb)
