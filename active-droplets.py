#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import newton_krylov

import sympy as sp

class symbols:
    """Variables common across expressions."""

    # Geometric variables:
    x = sp.Symbol('x')
    r = sp.Symbol('r')
    droplet_radius = sp.Symbol('R')
    # Aliases:
    R = droplet_radius

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

from ode import Expression

class Derived(Expression):
    arguments = sp.symbols('a b')

class PseudoCoefficient(Expression):
    arguments = []
    parameters = [symbols.zeta, symbols.lamb, symbols.K]

    @classmethod
    def expression(cls):
        return (symbols.zeta - 2*symbols.lamb) / symbols.K

class Pseudodensity(Expression):
    arguments = [symbols.density]
    parameters = [symbols.zeta, symbols.lamb, symbols.K]

    @classmethod
    def expression(cls):
        return (sp.exp(PseudoCoefficient.expr()*symbols.density) - 1) / PseudoCoefficient.expr()

    @classmethod
    def numerical_implementation(cls, deriv=0):
        expr = cls.diff(deriv)
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

    # def __init__(self):
    #     super().__init__()

    @classmethod
    def expression(cls):
        return sp.log(1 + PseudoCoefficient.expr()*symbols.pseudodensity) / PseudoCoefficient.expr()

    @classmethod
    def numerical_implementation(cls, deriv=0):
        expr = cls.expression(deriv)
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
    def free_energy_density(cls):
        raise NotImplementedError

    @property
    def f_params(self):
        return self.parameter_values[3:]
    @property
    def f(self):
        return self.free_energy_density()(*self.f_params)

    @classmethod
    def expression(cls):
        # Ensure we only do an expensive symbolic integration once per execution.
        try: return cls._expression
        except:
            phi = symbols.density
            f = cls.free_energy_density().expression()
            df = f.diff(phi)
            psi = Pseudodensity.expression()
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

            cls._expression = pseudopotential
            return cls._expression

class Phi4FreeEnergyDensity(Expression):
    arguments = [symbols.density]
    parameters = [symbols.t, symbols.u]

    @classmethod
    def expression(cls):
        return 0.5*symbols.t*symbols.density**2 + 0.25*symbols.u*symbols.density**4

class Phi4Pseudopotential(Pseudopotential):
    arguments = [symbols.density]
    parameters = [symbols.zeta, symbols.lamb, symbols.K, symbols.t, symbols.u]

    @classmethod
    def free_energy_density(cls):
        return Phi4FreeEnergyDensity

class ActiveModelBPlus:
    def __init__(self, pseudopotential, d=3):
        self.pseudopotential = pseudopotential
        self.free_energy_density = self.pseudopotential.f
        self.pseudodensity = Pseudodensity(*self.pseudopotential.parameter_values[:3])

    def phi4(zeta, lamb, K=1, t=-0.25, u=0.25, **kwargs):
        g = Phi4Pseudopotential(zeta, lamb, K, t, u)
        return ActiveModelBPlus(g, **kwargs)

    def bulk_pseudopressure(self, phi):
        return self.pseudodensity(phi)*self.free_energy_density(phi, deriv=1) - self.pseudopotential(phi)

    @property
    def bulk_binodals(self, eps=1e-4):
        P = self.bulk_pseudopressure
        mu = lambda phi: self.free_energy_density(phi, deriv=1)

        phi_guess = np.array([-1, 1])
        residual = lambda x: (P(x[0]) - P(x[1]), mu(x[0]) - mu(x[1]))
        phi = newton_krylov(residual, phi_guess)

        return phi

def bulk_binodals(zeta_lamb, *args, **kwargs):
    zeta = lambda x: 1
    lamb = lambda x: 0.5*(zeta(x) - x)
    phi = lambda x: tuple(ActiveModelBPlus.phi4(zeta(x), lamb(x), *args, **kwargs).bulk_binodals)
    phi = np.vectorize(phi)
    return phi(zeta_lamb)

#def pseudopressure

#system = ActiveModelBPlus(0, 0)
#phi = 0.5
#print(system.numerical_pseudopotential(phi), system.pseudopotential(phi))

# system = ActiveModelBPlus(0, 0)
# phi0, phi1 = 1, -1
# R = 5
# a,b = system.density_profile(phi0, phi1, R, N=101)
# print(np.array((a,b)).T)
# plt.plot(a/R, b, 'o-', mfc='None')
# plt.axvline(x=1, ls='dashed')
# plt.axhline(y=0.5*(phi0+phi1), ls='dashed')
# plt.xlabel('$r/R$')
# plt.ylabel('$\phi(r)$')
# plt.show()

# import sys; sys.exit(0)

Pseudodensity(0,0,1)
#print(pseudopotential.series(symbols.zeta, 2*symbols.lamb, 1))
#print(Expression.expression())
K = 1
t = -0.25
u = 0.25
phi = np.linspace(-1, 1, 1000)
f = Phi4FreeEnergyDensity(t, u)(phi)
plt.plot(phi, f)
for z, l in [(0.1,0.1), (1,1), (1,1.1)]:
    g = Phi4Pseudopotential(z, l, K, t, u)(phi)
    pl, = plt.plot(phi, g)
    #plt.plot(phi, g, '-.', lw=0.5, c=pl.get_color())
    dpsi = Phi4Pseudopotential(z, l, K, t, u)(phi, deriv=1)
    plt.plot(phi, dpsi, '--', c=pl.get_color())

    #abm = ActiveModelBPlus2(z, l)
    # f2 = abm.f(phi)
    # g2 = abm.pseudopotential(phi)
    # #plt.plot(phi, f2, '--', c=pl.get_color())
    # plt.plot(phi, g2, ':', c=pl.get_color())
    #assert np.allclose(phi, Density(z, l, K)(psi))
#plt.show()

# import sys; sys.exit(0)

# psi = np.linspace(-2, 2, 1000)
# system = ActiveModelBPlus(0, 0)
# assert np.allclose(system.f(psi), system.pseudopotential(system.density(psi)))

# for z, l in [(0,0), (1,0.6), (1,0.4), (3,1), (3,1.5)]:
#     system = ActiveModelBPlus(z, l)
#     pl, = plt.plot(psi, system.pseudopotential(system.density(psi)), label=(r'$\zeta = %r; \lambda = %r$' % (z, l)))
#     phi = system.bulk_binodals
#     plt.plot(system.pseudodensity(phi), system.pseudopotential(phi), 'o', c=pl.get_color(), zorder=10, mfc='None')


# K = 1
# phi = np.linspace(-1, 1, 100)

# plt.figure()
# for z, l in [(0,0), (0.1,0.1), (1,1), (1,1.1)]:
#     psi = Pseudodensity(z, l, K)(phi)
#     abm = ActiveModelBPlus(z, l)
#     pl, = plt.plot(phi, psi, lw=0.5)
#     psi2 = abm.pseudodensity(phi)
#     plt.plot(phi, psi2, '--', c=pl.get_color())
#     assert np.allclose(phi, Density(z, l, K)(psi))
# plt.show()
# import sys; sys.exit(0)

# plt.ylim([-0.15, 0.15])
# plt.legend(loc='best')
# plt.xlabel('$\psi$')
# plt.ylabel('$g$')

plt.figure()
z = np.linspace(-6, 6, 100)
#print(bulk_coexistence(z))
phi1, phi2 = bulk_binodals(z)
#print(np.array((z,phi1,phi2)).T)
plt.plot(z, phi1, label='$\phi_1$')
plt.plot(z, phi2, label='$\phi_2$')
plt.xlabel('$\zeta - 2\lambda$')
plt.ylabel('$\phi$')
plt.legend(loc='best')

plt.show()
