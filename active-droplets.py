#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import newton_krylov
from scipy.special import erf, erfinv
import sympy as sp

from interpolate import HermiteInterpolator
from ode import WeakFormProblem1d

from cache import lru_cache, disk

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

class symbols:
    """Variables common across expressions."""

    # Dimensionality:
    d = sp.Symbol('d')

    # Geometric variables:
    x = sp.Symbol('x')
    r = sp.Symbol('r')
    droplet_radius = sp.Symbol('R')
    domain_size = sp.Symbol('L')
    interfacial_width = sp.Symbol('xi')
    # Aliases:
    R = droplet_radius
    L = domain_size
    xi = interfacial_width

    # State variables:
    density = sp.Symbol('phi')
    pseudodensity = sp.Symbol('psi')
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
    zeta = sp.Symbol('zeta')
    lamb = sp.Symbol('lambda')

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
    def numerical_implementation(cls, deriv=0):
        """Expression to pass for numerical implementation, that may differ in form from the exact
        expression in order to better handle e.g. numerical instabilities."""
        return cls.diff(deriv)

    @classmethod
    @lru_cache
    #@disk.cache
    def compiled_function(cls, deriv=0):
        """Compile the function so that it can be numerically evaluated with native python
        structures, if not already done so for this expression.

        We cache the result (at the class-level) to prevent unnecessary recompilations for many
        instances of the same expression (with e.g. different choices of parameters).

        Args:
            deriv: order of derivative to compile into an evaluatable function
        """
        return sp.lambdify(cls.variables, cls.numerical_implementation(deriv))

    def __call__(self, *args, deriv=0):
        """Numerically evaluate expression."""
        cls = self.__class__
        f = cls.compiled_function(deriv)
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
    def numerical_implementation(cls, deriv=0):
        expr = cls.expr.diff(cls.argument, deriv)
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
    def numerical_implementation(cls, deriv=0):
        expr = cls.expression.diff(cls.argument, deriv)
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
    @lru_cache
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
    def numerical_implementation(cls, deriv=0):
        expr = cls.expression.diff(cls.argument, deriv)
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
    density0, density1 = sp.symbols('phi0 phi1')
    parameters = [density1,
                  symbols.droplet_radius, symbols.domain_size,
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
    @lru_cache
    def strong_form(cls):
        r, d, K, zeta, lamb = cls.argument, symbols.d, symbols.K, symbols.zeta, symbols.lamb
        phi = cls.unknown_function(r)

        local_term = cls.free_energy_terms + (lamb - zeta/2) * phi.diff(r)**2
        nonlocal_term = -zeta * (d-1) * phi.diff(r)**2 / r
        expr = r**2 * ( local_term.diff(r) + nonlocal_term )
        return expr.simplify()

    @classmethod
    @property
    @lru_cache
    def weak_form(cls):
        r, d, K, zeta, lamb = cls.argument, symbols.d, symbols.K, symbols.zeta, symbols.lamb
        phi = cls.unknown_function(r)
        test_function = cls.b(r)

        local_term = cls.free_energy_terms + (lamb - zeta/2) * phi.diff(r)**2
        nonlocal_term = -zeta * (d-1) * phi.diff(r)**2 / r
        expr = -local_term * (r**2*test_function).diff(r) + nonlocal_term * (r**2*test_function)
        return expr.simplify()

    @classmethod
    @property
    @lru_cache
    def natural_boundary_condition(cls):
        r, d, K, zeta, lamb = cls.argument, symbols.d, symbols.K, symbols.zeta, symbols.lamb
        phi = cls.unknown_function(r)
        test_function = cls.b(r)

        expr = cls.free_energy_terms * r**2 * test_function
        return expr.simplify()

    @classmethod
    @property
    def boundary_conditions(cls):
        r, phi = cls.argument, cls.unknown_function

        return {(0, phi(r).diff(r), 0),
                (symbols.droplet_radius, phi(r), 0),
                (symbols.domain_size, phi(r), cls.density1),
                (symbols.domain_size, phi(r).diff(r), 0)}

class ActiveDroplet(HermiteInterpolator):
    @classmethod
    def from_guess(cls, field_theory, R, phi0, phi1,
                   domain_size=None, order=2,
                   x=None, npoints=50, interfacial_width=None, print_updates=None):
        if domain_size is None: domain_size = 5*R
        if interfacial_width is None: interfacial_width = 0.5*R

        if x is None:
            xdist = SechSquaredDistribution(R, interfacial_width)
            #xdist = LogNormalDistribution(np.log(R), 0.25)

            # Generate points by the above distribution that contain the (i) origin, (ii) R
            # and (iii) the domain size.
            npoints_left = npoints//2
            npoints_right = npoints - npoints_left
            x = np.concatenate((np.linspace(0, xdist.cdf(R), npoints_left),
                                np.linspace(xdist.cdf(R), xdist.cdf(domain_size), npoints_right+1)[1:]))
            x = xdist.idf(x)

            # Remove any rounding errors from the inverse distribution so we exactly obtain
            # the three chosen points.
            x[0] = 0
            x[npoints_left-1] = R
            x[-1] = domain_size

        guess = phi0 + (phi1 - phi0) * xdist.cumulative_distribution_function
        guess = guess.subs({p: v for p, v in zip(xdist.parameters, xdist.parameter_values)})

        weights = np.zeros((len(x), order))
        for c in range(order):
            f = guess.diff(xdist.argument, c)
            f = sp.lambdify(xdist.argument, f)
            with np.errstate(invalid='ignore', divide='ignore'):
                weights[:,c] = sp.lambdify(xdist.argument, guess.diff(xdist.argument, c))(x)
        weights[~np.isfinite(weights)] = 0

        return cls(field_theory, R, x, weights, print_updates=print_updates)

    def __init__(self, field_theory, R, *args, print_updates=None, **kwargs):
        self.field_theory = field_theory
        self.R = R
        super().__init__(*args, **kwargs)
        self.solve(print_updates=print_updates)

    def solve(self, print_updates=None):
        constant_params = tuple(self.field_theory.pseudopotential.parameter_values + (self.field_theory.d,))
        problem = ActiveModelBSphericalInterface(self.phi1, self.R, self.domain_size, *constant_params)
        with np.errstate(all='raise'):
            self.weights = problem.solve(self.x, self.weights, print_updates=print_updates)

    @property
    def domain_size(self):
        return self.x[-1]

    def refine(self, refinement_tol, print_updates=None, max_iters=None, niters=0,
               max_points=int(1e5)):
        if max_iters and niters >= max_iters:
            raise RuntimeError('result not converging after %d mesh refinement iterations!' % niters)

        f = sp.Function('f')
        x = symbols.x
        error_estimator = f(x).diff(x)**2

        I = self.analytic_integral(error_estimator, f, x)
        errors = np.diff(I.weights[:,0]) * np.diff(I.nodes)

        if np.any(errors > refinement_tol):
            elements_to_split = errors > refinement_tol
            print('splitting:', np.sum(elements_to_split))
            element_midpoints = 0.5*(self.x[1:] + self.x[:-1])
            new_x = element_midpoints[elements_to_split]
            new_x = np.sort( np.concatenate((I.x, new_x)) )

            if new_x.size > max_points:
                raise RuntimeError('possible singular behaviour - trying to create too many points (%d) during mesh refinement!' % len(new_x))

            _, order = self.weights.shape
            new_weights = np.empty((new_x.size, order))
            for c in range(order):
                new_weights[:,c] = self(new_x, c)

            drop = ActiveDroplet(self.field_theory, self.R, new_x, new_weights,
                                 print_updates=print_updates)
            plt.plot(drop.x/drop.R, drop(drop.x), lw=0.5)
            return drop.refine(refinement_tol, print_updates, max_iters, niters+1, max_points)

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
    @lru_cache
    def chemical_potential_gradient_integrand_expression(cls):
        phi = sp.Function('phi')
        r = symbols.r
        zeta, lamb, K, d = symbols.zeta, symbols.lamb, symbols.K, symbols.d

        integrand = zeta * (d-1) * phi(r).diff(r)**2 / r
        return integrand, phi, r

    @property
    def mu0(self):
        integrand, phi, r = self.chemical_potential_gradient_integrand_expression
        integrand = integrand.subs({p: v for p, v in zip(self.pseudopotential.parameters,
                                                         self.pseudopotential.parameter_values)})
        integrand = integrand.subs(symbols.d, self.d)

        bulk_part = self.field_theory.free_energy_density(self.phi0, deriv=1) - self.field_theory.K*self(self.x[0],2)
        confined_part = self.integrate(integrand, phi, r)
        confined_part2 = self.analytic_integral((r*integrand).simplify(), phi, r)(self.x[-1]) / self.R
        print(bulk_part, confined_part, confined_part2, bulk_part + confined_part, self.mu1)
        return bulk_part + confined_part

    @property
    def mu1(self):
        return self.field_theory.free_energy_density(self.phi1, deriv=1) - self.field_theory.K*self(self.x[-1],2)

    @classmethod
    @property
    @lru_cache
    def surface_tension_integrand_expression(cls):
        phi = sp.Function('phi')
        r = symbols.r
        zeta, lamb, K = symbols.zeta, symbols.lamb, symbols.K
        phi0 = sp.Symbol('phi0')

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
    @lru_cache
    def pseudopressure_drop(self):
        integrand = self.surface_tension_integrand(self.phi0)
        phi, r = sp.Function('phi'), symbols.r
        return self.integrate(integrand, phi, r)

class ActiveModelBPlus:
    @classmethod
    def phi4(cls, zeta, lamb, K=1, t=-0.25, u=0.25, *args, **kwargs):
        return Phi4Pseudopotential(zeta, lamb, K, t, u)

    def __init__(self, *args, d=3, **kwargs):
        self.d = d
        self.pseudopotential = self.phi4(*args, **kwargs)
        self.free_energy_density = self.pseudopotential.f
        self.pseudodensity = self.pseudopotential.pseudodensity

    @property
    def zeta(self):
        return self.pseudopotential.parameter_values[0]

    @property
    def lamb(self):
        return self.pseudopotential.parameter_values[1]

    @property
    def K(self):
        return self.pseudopotential.parameter_values[2]

    def bulk_pseudopressure(self, phi):
        return self.pseudodensity(phi)*self.free_energy_density(phi, deriv=1) - self.pseudopotential(phi)

    @property
    @lru_cache
    def bulk_binodals(self, eps=1e-12):
        phi_guess = np.array([-1, 1])
        if np.abs(self.zeta - 2*self.lamb) < eps:
            return phi_guess

        P = self.bulk_pseudopressure
        mu = lambda phi: self.free_energy_density(phi, deriv=1)

        residual = lambda x: (P(x[0]) - P(x[1]), mu(x[0]) - mu(x[1]))
        phi = newton_krylov(residual, phi_guess)

        return phi

    def droplet(self, R, domain_size=None, order=2, refinement_tol=1e-4, print_updates=None,
                phi1=None):
        bulk_phi = self.bulk_binodals
        if phi1 is None: phi1 = bulk_phi[1]
        phi0 = bulk_phi[np.argmax(np.abs(bulk_phi - phi1))]

        initial_droplet = lambda p: ActiveDroplet.from_guess(self, R, phi0, p,
                                                             domain_size=domain_size,
                                                             order=order,
                                                             print_updates=print_updates)
        droplet = lambda p: initial_droplet(p).refine(refinement_tol, print_updates=print_updates)

        pseudopressure_balance = lambda droplet: droplet.pseudopressure_drop - (droplet.pseudopressure0 - droplet.pseudopressure1)
        residual = lambda p: pseudopressure_balance(droplet(p))
        #phi1 = newton_krylov(residual, phi_guess)

        return droplet(phi1)

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    np.set_printoptions(4, suppress=True, linewidth=10000)

    K, t, u = 1, -0.25, 0.25
    constant_parameters = (K, t, u)
    R = 10

    zeta, lamb = -4, -1
    #zeta, lamb = 0, 0
    model = ActiveModelBPlus(zeta, lamb, *constant_parameters)

    try: drop = model.droplet(R, refinement_tol=1e-4, phi1=-0.9)
    except Exception as e:
        print(e)
        plt.show()
        #raise e from None
        import sys; sys.exit(0)

    plt.axvline(x=1, ls='dashed', lw=0.5)
    plt.axhline(y=0, ls='dashed', lw=0.5)
    P0, P1 = drop.pseudopressure0, drop.pseudopressure1
    mu0, mu1 = drop.mu0, drop.mu1
    S = drop.pseudopressure_drop
    print('z=%g l=%g: phi=[%.4f->%.4f] mu=[%.4f->%.4f] P=[%.4f->%.4f] S=%.4f dP=%.4f' % (zeta, lamb, drop.phi0, drop.phi1, mu0, mu1, P0, P1, S, P0-P1))
    xx = np.linspace(np.min(drop.x), np.max(drop.x), 1001)
    plt.plot(xx/R, drop(xx), '-', lw=0.5)
    plt.show()
    import sys; sys.exit(0)

    phi1 = np.linspace(-0.9, -1.2, 21)
    dP = np.empty(phi1.shape)
    S = np.empty(phi1.shape)
    for i, p in enumerate(phi1):
        print(p)
        drop = model.droplet(R, phi1=p)
        dP[i] = drop.pseudopressure0 - drop.pseudopressure1
        S[i] = drop.pseudopressure_drop
        print('%.4f %.4g %.4g' % (p, dP[i], S[i]))

    plt.plot(phi1, dP, label=r'$\delta P$')
    plt.plot(phi1, S, label=r'$S$')
    plt.legend(loc='best')

    plt.show()
    import sys; sys.exit(0)

    for zeta, lamb in [(0, 0), (-4, -1), (4, 1)]:
        model = ActiveModelBPlus(zeta, lamb, *constant_parameters)
        drop = model.droplet(R)

        P0, P1 = drop.pseudopressure0, drop.pseudopressure1
        mu0, mu1 = drop.mu0, drop.mu1
        S = drop.pseudopressure_drop
        print('z=%g l=%g: phi=[%.4f->%.4f] mu=[%.4f->%.4f] P=[%.4f->%.4f] S=%.4f dP=%.4f' % (zeta, lamb, drop.phi0, drop.phi1, mu0, mu1, P0, P1, S, P0-P1))

        xx = np.linspace(np.min(drop.x), np.max(drop.x), 1001)

        label = r'$\zeta=%g, \; \lambda=%g$' % (zeta, lamb)
        plt.plot(xx/R, drop(xx), '-', lw=0.5, label=label)

    plt.legend(loc='best')
    plt.ylim([-2, 2])
    plt.xlabel('$r/R$')
    plt.ylabel('$\phi$')

    # from progressbar import ProgressBar, Percentage, Bar, ETA
    # widgets = [Percentage(), ' ', Bar(), ' ', ETA()]
    # progress = ProgressBar(widgets=widgets)

    # R = np.arange(10, 41, 1)
    # phi0 = np.empty(R.shape)
    # phi1 = np.empty(R.shape)
    # for i,r in enumerate(progress(R)):
    #     f, _ = model.droplet(r)
    #     phi0[i] = f(0)
    #     phi1[i] = f(f.x[-1])

    # bulk1, bulk0 = model.bulk_binodals

    # plt.figure()
    # plt.plot(R, phi0)
    # plt.axhline(y=bulk0, lw=0.5, ls='dashed')
    # plt.xlabel('$R$')
    # plt.ylabel('$\phi_+$')

    # plt.figure()
    # plt.plot(R, phi1)
    # plt.axhline(y=bulk1, lw=0.5, ls='dashed')
    # plt.xlabel('$R$')
    # plt.ylabel('$\phi_-$')

    plt.show()

    # sys.exit(0)

    # K = 1
    # t = -0.25
    # u = 0.25
    # phi = np.linspace(-1, 1, 1000)
    # f = Phi4FreeEnergyDensity(t, u)(phi)
    # plt.plot(phi, f)
    # for z, l in [(0.1,0.1), (1,1), (1,1.1)]:
    #     g = Phi4Pseudopotential(z, l, K, t, u)(phi)
    #     pl, = plt.plot(phi, g)
    #     #plt.plot(phi, g, '-.', lw=0.5, c=pl.get_color())
    #     dpsi = Phi4Pseudopotential(z, l, K, t, u)(phi, deriv=1)
    #     plt.plot(phi, dpsi, '--', c=pl.get_color())

    # plt.figure()
    # z = np.linspace(-6, 6, 100)
    # #print(bulk_coexistence(z))
    # phi1, phi2 = bulk_binodals(z)
    # #print(np.array((z,phi1,phi2)).T)
    # plt.plot(z, phi1, label='$\phi_1$')
    # plt.plot(z, phi2, label='$\phi_2$')
    # plt.xlabel('$\zeta - 2\lambda$')
    # plt.ylabel('$\phi$')
    # plt.legend(loc='best')

    # plt.show()
