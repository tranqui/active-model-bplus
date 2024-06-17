#!/usr/bin/env python3

import numpy as np
from scipy.linalg import solve_banded
from scipy.integrate._quadrature import _cached_roots_legendre
import sympy as sp

from .interpolate import HermiteInterpolatingPolynomial, HermiteInterpolator
from . import differentiate
from .cache import cache, cached_property, disk_cache

import sys
# print_compilation_updates = None
print_compilation_updates = sys.stderr

class WeakFormProblem1d:
    argument = sp.Symbol('x')
    unknown_function = sp.Function('unknown_function')
    basis_function = sp.Function('basis_function')

    parameters = []

    def __init__(self, *args):
        """Instantiate problem with specific parameters.

        Args:
            *args: values of parameters in same order as parameters class variable.
        """
        assert len(args) is len(self.parameters)
        self.parameter_values = args

    @classmethod
    @property
    def name(cls):
        return cls.__name__

    @classmethod
    @property
    def x(cls):
        return cls.argument

    @classmethod
    @property
    def u(cls):
        return cls.unknown_function

    @classmethod
    @property
    def b(cls):
        return cls.basis_function

    @classmethod
    @property
    def strong_form(cls):
        raise NotImplementedError('strong form not defined for this problem!')

    @classmethod
    @property
    def weak_form(cls):
        raise NotImplementedError('this problem has not been specified!')

    @classmethod
    @property
    def natural_boundary_condition(cls):
        return

    # Options to turn on/off natural boundary conditions on each boundary.
    # Use case: turning off e.g. left boundary in polar coordinates when it is not a real
    #           boundary condition.
    boundary_left = True
    boundary_right = True

    @classmethod
    @property
    def boundary_conditions(cls):
        raise NotImplementedError('boundary conditions have not been specified!')

    @classmethod
    @property
    @cache
    def analytic_solution(cls):
        bcs = {expr.subs(cls.argument, point): value for point, expr, value in cls.boundary_conditions}
        return sp.dsolve(cls.strong_form, cls.unknown_function(cls.argument), ics=bcs).rhs

    @classmethod
    @property
    @cache
    def compiled_exact_solution(cls):
        return sp.lambdify([cls.argument] + cls.parameters, cls.analytic_solution)

    def exact_solution(self, x):
        return self.compiled_exact_solution(x, *self.parameter_values)

    @classmethod
    def elemental_variables(cls, order=1):
        """Variables taken as arguments to expression in numerical evaluations (i.e. including
        parameters)."""
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)
        return [polynomial.x0, polynomial.x1] + polynomial.weight_variables + cls.parameters

    @classmethod
    @cache
    def natural_boundary_condition_expressions(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)

        expressions = []
        basic_expression = cls.natural_boundary_condition.subs(cls.unknown_function,
                                                               sp.Lambda(cls.argument, polynomial.general_expression))

        for i,w in enumerate(polynomial.general_weight_functions):
            specific_expression = basic_expression.subs(cls.basis_function,
                                                        sp.Lambda(cls.argument, w)).doit()
            expressions += [specific_expression]

        # We need different expressions for each side of the [-1,1] domain.
        left, right = ([e.subs(cls.argument, x).doit() for e in expressions] for x in (polynomial.x0, polynomial.x1))
        return left, right

    @classmethod
    @cache
    @disk_cache
    def compiled_natural_boundary_condition_expressions(cls, order=1, *args, **kwargs):
        if print_compilation_updates:
            print_compilation_updates.write('%s: compiling natural boundary conditions...' % cls.name)
            print_compilation_updates.flush()

        expressions = cls.natural_boundary_condition_expressions(order, *args, **kwargs)
        compiled_expressions = []

        arguments = cls.elemental_variables(order)
        for boundary in expressions: # loop over left and right expressions
            compiled_expressions += [[sp.lambdify(arguments, e) for e in boundary]]

        if print_compilation_updates: print_compilation_updates.write(' done.\n')
        return compiled_expressions

    @classmethod
    @cache
    def natural_boundary_condition_jacobians(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)
        J = []
        left, right = cls.natural_boundary_condition_expressions(order)
        for expressions in [left, right]:
            J += [[[e.diff(w).doit() for w in polynomial.weight_variables] for e in expressions]]
        return J

    @classmethod
    @cache
    @disk_cache
    def compiled_natural_boundary_condition_jacobians(cls, order=1, *args, **kwargs):
        if print_compilation_updates:
            print_compilation_updates.write('%s: compiling natural boundary Jacobians...' % cls.name)
            print_compilation_updates.flush()

        compiled_jacobians = []
        left, right = cls.natural_boundary_condition_jacobians(order, *args, **kwargs)
        for jacobians in [left, right]:
            boundary_jacobians = []
            for row in jacobians:
                compiled_row = []
                for expression in row:
                    compiled_row += [sp.lambdify(cls.elemental_variables(order), expression)]
                boundary_jacobians += [compiled_row]
            compiled_jacobians += [boundary_jacobians]

        if print_compilation_updates: print_compilation_updates.write(' done.\n')
        return compiled_jacobians

    @classmethod
    @cache
    def boundary_condition_expressions(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)

        expressions = []
        for point, lhs, rhs in cls.boundary_conditions:
            expression = lhs - rhs
            expression = expression.subs(
                {cls.unknown_function: sp.Lambda(cls.argument, polynomial.general_expression),
                 cls.argument: point}).doit()

            expressions += [(point, expression)]

        return expressions

    @classmethod
    @cache
    @disk_cache
    def compiled_boundary_condition_expressions(cls, order=1, *args, **kwargs):
        if print_compilation_updates:
            print_compilation_updates.write('%s: compiling boundary conditions...' % cls.name)
            print_compilation_updates.flush()

        expressions = cls.boundary_condition_expressions(order, *args, **kwargs)
        compiled_expressions = []

        arguments = cls.elemental_variables(order)
        for point, expression in expressions:
            compiled_expressions += [(point, sp.lambdify(arguments, expression))]

        if print_compilation_updates: print_compilation_updates.write(' done.\n')
        return compiled_expressions

    @classmethod
    @cache
    def boundary_condition_jacobians(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)
        J = []
        for point, expression in cls.boundary_condition_expressions(order):
            J += [(point, [expression.diff(w).doit() for w in polynomial.weight_variables])]
        return J

    @classmethod
    @cache
    @disk_cache
    def compiled_boundary_condition_jacobians(cls, order=1, *args, **kwargs):
        if print_compilation_updates:
            print_compilation_updates.write('%s: compiling boundary Jacobians...' % cls.name)
            print_compilation_updates.flush()

        jacobians = cls.boundary_condition_jacobians(order, *args, **kwargs)
        compiled_jacobians = []
        for point, row in jacobians:
            compiled_row = []
            for expression in row:
                compiled_row += [sp.lambdify(cls.elemental_variables(order), expression)]
            compiled_jacobians += [(point, compiled_row)]

        if print_compilation_updates: print_compilation_updates.write(' done.\n')
        return compiled_jacobians

    @classmethod
    @cache
    def elemental_residuals(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)

        residuals = []
        basic_integrand = cls.weak_form.subs(cls.unknown_function,
                                             sp.Lambda(cls.argument, polynomial.general_expression))

        # We use a fixed-order Gaussian quadrature rule for the integration, so we need to
        # determine the location of points to sample and the weights. These are pre-calculated
        # in numpy in the [-1, 1] interval:
        roots, weights = _cached_roots_legendre(2*order+1)
        # Transform to general interval [x0, x1]:
        x = polynomial.inverse_coordinate_transform
        dxds = sp.Lambda(cls.argument, x.diff(cls.argument))
        x = sp.Lambda(cls.argument, x)
        weights = [w*dxds(r) for r,w in zip(roots,weights)]
        roots = [x(r) for r in roots]

        for i,w in enumerate(polynomial.general_weight_functions):
            specific_integrand = basic_integrand.subs(cls.basis_function,
                                                      sp.Lambda(cls.argument, w)).doit()
            result = sum([w*specific_integrand.subs(cls.argument, p).doit() for p, w in zip(roots, weights)])

            residuals += [result]

        return residuals

    @classmethod
    @cache
    @disk_cache
    def compiled_elemental_residuals(cls, order, *args, **kwargs):
        if print_compilation_updates:
            print_compilation_updates.write('%s: compiling main residuals...' % cls.name)
            print_compilation_updates.flush()

        residuals = cls.elemental_residuals(order, *args, **kwargs)
        compiled_residuals = []
        for expression in residuals:
            compiled_residuals += [sp.lambdify(cls.elemental_variables(order), expression)]

        if print_compilation_updates: print_compilation_updates.write(' done.\n')
        return compiled_residuals

    def residuals(self, nodes, weights, *args, **kwargs):
        nelements, order = weights.shape
        nelements -= 1

        R = np.zeros(weights.shape)

        polynomial = HermiteInterpolatingPolynomial.from_cache(order, self.argument)
        variables = polynomial.weight_variables
        functions = self.compiled_elemental_residuals(order, *args, **kwargs)

        xleft, xright = nodes[:-1], nodes[1:]
        w = np.hstack((weights[:-1], weights[1:]))

        for var, func in zip(variables, functions):
            with np.errstate(all='raise'):
                r = func(xleft, xright, *w.T, *self.parameter_values)

            boundary, deriv = var.indices
            if boundary == 0: R[:-1,deriv] += r
            elif boundary == 1: R[1:,deriv] += r
            else: raise RuntimeError('unknown variable indices during residual calculation!')

        # Apply natural boundary condition needed to make weak form valid (these conditions
        # arise from surface terms left over from e.g. integration by parts).
        if self.natural_boundary_condition:
            left, right = self.compiled_natural_boundary_condition_expressions(order)
            for c, (l, r) in enumerate(zip(left, right)):
                with np.errstate(divide='raise'):
                    if self.boundary_left:
                        R[c//order, c%order] += l(xleft[0], xright[0], *w[0], *self.parameter_values)
                    if self.boundary_right:
                        R[-2 + c//order, c%order] += r(xleft[-1], xright[-1], *w[-1], *self.parameter_values)

        # Evaluate residual contributions from specific boundary conditions.
        element_edges = nodes[:-1]
        bcs = {}
        for point, func in self.compiled_boundary_condition_expressions(order):
            # If point is an analytic expression we may need to evaluate it.
            point = sp.lambdify(self.parameters, point)(*self.parameter_values)

            # Evaluate boundary condition on the local element
            element = np.digitize(point, element_edges)-1
            xleft, xright = nodes[element:element+2]
            value = func(xleft, xright, *w[element], *self.parameter_values)

            # We will place the boundary condition on a residual entry for the closest node,
            # because it should depend on local weights there.
            closest_node = np.abs(nodes - point).argmin()
            try: bcs[closest_node] += [value]
            except: bcs[closest_node] = [value]

        # Make sure boundary conditions fall on distinct residual entries for the selected nodes.
        for node, conditions in bcs.items():
            for i, value in enumerate(conditions):
                R[node,i] = value

        return R.reshape(-1)

    @classmethod
    @cache
    def elemental_jacobians(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)
        J = []
        for R in cls.elemental_residuals(order):
            J += [[R.diff(w).doit() for w in polynomial.weight_variables]]
        return J

    @classmethod
    @cache
    @disk_cache
    def compiled_elemental_jacobians(cls, order, *args, **kwargs):
        if print_compilation_updates:
            print_compilation_updates.write('%s: compiling main Jacobians...' % cls.name)
            print_compilation_updates.flush()

        jacobians = cls.elemental_jacobians(order, *args, **kwargs)
        compiled_jacobians = []
        for row in jacobians:
            compiled_row = []
            for expression in row:
                compiled_row += [sp.lambdify(cls.elemental_variables(order), expression)]
            compiled_jacobians += [compiled_row]

        if print_compilation_updates: print_compilation_updates.write(' done.\n')
        return compiled_jacobians

    def jacobian(self, nodes, weights, *args, **kwargs):
        nelements, order = weights.shape
        nelements -= 1

        # Equations depend on 3*order variables (1 x order for each of left, central and right
        # points) in each element.
        J = np.zeros((3*order, nelements+1, order))

        polynomial = HermiteInterpolatingPolynomial.from_cache(order, self.argument)
        variables = polynomial.weight_variables
        functions = self.compiled_elemental_jacobians(order, *args, **kwargs)

        xleft, xright = nodes[:-1], nodes[1:]
        w = np.hstack((weights[:-1], weights[1:]))

        for var, row in zip(variables, functions):
            boundary, deriv = var.indices

            for i, func in enumerate(row):
                j = func(xleft, xright, *w.T, *self.parameter_values)
                if boundary == 0:
                    J[i+order,:-1,deriv] += j
                elif boundary == 1:
                    J[i,1:,deriv] += j
                else: raise RuntimeError('unknown variable indices during residual calculation!')

        J = J.reshape(len(J), -1)

        # Apply natural boundary condition needed to make weak form valid (these conditions
        # arise from surface terms left over from e.g. integration by parts).
        if self.natural_boundary_condition:
            left, right = self.compiled_natural_boundary_condition_jacobians(order)
            for c, (l, r) in enumerate(zip(left, right)):
                starting_row = len(J)//2 - c

                eqn = -2*order+c
                if self.boundary_right:
                    r = [f(xleft[-1], xright[-1], *w[-1], *self.parameter_values) for f in r]
                    J[:len(r), eqn] += r

                eqn = c
                if self.boundary_left:
                    l = [f(xleft[0], xright[0], *w[0], *self.parameter_values) for f in l]
                    J[order:order+len(l), eqn] += l

        # Apply boundary conditions.
        element_edges = nodes[:-1]
        bcs = {}
        for point, row in self.compiled_boundary_condition_jacobians(order):
            # If point is an analytic expression we may need to evaluate it.
            point = sp.lambdify(self.parameters, point)(*self.parameter_values)

            element = np.digitize(point, element_edges)-1
            closest_node = np.abs(nodes - point).argmin()
            xleft, xright = nodes[element:element+2]
            values = [func(xleft, xright, *w[element], *self.parameter_values) for func in row]

            boundary_on_left = closest_node == element
            if boundary_on_left: entry = np.concatenate((np.zeros(order), values))
            else: entry = np.concatenate((values, np.zeros(order)))

            try: bcs[closest_node] += [entry]
            except: bcs[closest_node] = [entry]

        # Ensure boundary conditions are placed on distinct rows
        for node, conditions in bcs.items():
            for c, entry in enumerate(conditions):
                eqn = node*order + c
                J[:, eqn] = entry

        # Each row now contains the nonzero Jacobian entries for each residual, but
        # we have to shift these to correspond to the matrix format needed by
        # scipy.linalg.solve_banded.

        # Add extra rows and shift columns so that each equation's own node runs along the diagonal.
        ncols = J.shape[1]
        J = np.vstack((np.zeros((order-1, ncols)), J))
        for c in range(1,order):
            J[:,c::order] = np.roll(J[:,c::order], -c, axis=0)

        # Shift the elements for each equation so elements of a single equation are stored
        # diagonally (cf. scipy.linalg.solve_banded which documents the matrix storage format).
        J = np.flipud(J)
        for c in range(len(J)):
            J[c] = np.roll(J[c], len(J)//2-c)

        return J

    def numerical_jacobian(self, nodes, weights, dx=1e-4):
        from differentiate import gradient
        f = lambda w: self.residuals(nodes, w)
        return gradient(f, weights, dx=dx).T

    def full_jacobian(self, J):
        """Convert banded jacobian into full square matrix. Useful for testing."""
        u = (J.shape[0]-1) // 2
        nnodes = J.shape[1]
        Jfull = np.zeros((nnodes, nnodes))

        for i in range(nnodes):
            for j in range(nnodes):
                if (u + i - j) < 0 or (u + i - j) >= len(J): continue
                Jfull[i, j] = J[u + i - j, j]
        return Jfull

    def solve(self, nodes, weights,
              newton_atol=1e-6, newton_rtol=1e-8, max_newton_iters=25, exceed_max_iters='warn',
              print_updates=None, **kwargs):
        nelements, order = weights.shape
        nelements -= 1

        R = self.residuals(nodes, weights)
        assert nodes.size == nelements+1
        assert R.size == order*(nelements+1)

        iters = 0
        if print_updates:
            update = lambda R: np.hstack((nodes.reshape(-1,1), R.reshape(-1,order)))
            print_updates.write('iter residuals\n%r %r\n' % (iters, update(R)))

        while np.any(np.abs(R) > newton_atol):
            if iters >= max_newton_iters:
                message = 'did not converge during ODE solve step after %d iters with %d nodes!' % (iters, nodes.size)
                if exceed_max_iters == 'warn':
                    sys.stderr.write('warning: %s\n' % message)
                    break
                elif exceed_max_iters == 'raise':
                    raise RuntimeError(message)

                # import sys
                # sys.stderr.write('nnodes: %d\n' % nodes.size)
                # sys.stderr.write('residuals: %r\n' % R)

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.plot(nodes, R.reshape(nodes.size,-1), lw=0.5)
                # plt.title('residuals')

                # plt.figure()
                # plt.plot(nodes, delta, lw=0.5)
                # plt.title('delta')

                # plt.figure()
                # pl, = plt.plot(nodes, weights[:,0], '.')
                # xx = np.linspace(nodes[0], nodes[-1], 1000)
                # f = HermiteInterpolator(nodes, weights)
                # plt.plot(xx, f(xx), lw=0.5, c=pl.get_color())
                # plt.title('current solution')

                # plt.figure()
                # plt.plot(nodes, relative_change.reshape(nodes.size,-1), lw=0.5)
                # plt.title('relative change')

                # plt.show()

            J = self.jacobian(nodes, weights)

            u = (J.shape[0]-1) // 2
            delta = solve_banded((u, u), J, -R).reshape(weights.shape)
            # from scipy.optimize import line_search
            # line_search(obj_func, obj_grad, start_point, search_gradient)
            if print_updates: print_updates.write('%r delta=%r\n' % (iters, delta))
            weights += delta

            R = self.residuals(nodes, weights)
            iters += 1
            if print_updates: print_updates.write('%r R=%r\n' % (iters, update(R)))

            with np.errstate(invalid='ignore', divide='ignore'):
                relative_change = np.divide(delta, weights).reshape(-1)
            relative_change[~np.isfinite(relative_change)] = 0
            relative_change[np.abs(R) < newton_atol] = 0
            if np.linalg.norm(relative_change) < newton_rtol: break
            #import sys; sys.stderr.write('%d %.4g\n' % (iters, np.linalg.norm(relative_change)))
            #if np.linalg.norm(delta) < newton_rtol: break

        if print_updates:
            print_updates.write('\nsolution:\n%r\n' % np.hstack([nodes.reshape(-1,1),weights]))

        return weights

class HeatEquation(WeakFormProblem1d):
    parameters = []

    @classmethod
    @property
    def strong_form(cls):
        x, u, b = cls.x, cls.u, cls.b
        return u(x).diff(x,2)

    @classmethod
    @property
    def weak_form(cls):
        x, u, b = cls.x, cls.u, cls.b
        return -u(x).diff()*b(x).diff()

    @classmethod
    @property
    def boundary_conditions(cls):
        return {(0, cls.u(cls.x), 1), (1, cls.u(cls.x), 0)}

class GinzburgLandauFlatInterface(WeakFormProblem1d):
    a, g, k = sp.symbols('a g K')
    parameters = [a, g, k]

    @classmethod
    @property
    def binodal(cls):
        a, g = cls.a, cls.g
        return sp.sqrt(-a/g)

    @classmethod
    @property
    def interfacial_width(cls):
        a, k = cls.a, cls.k
        return sp.sqrt(-2*k/a)

    @classmethod
    @property
    def domain_size(cls):
        return 10*cls.interfacial_width

    @property
    def numerical_domain_size(self):
        return sp.lambdify(self.parameters, self.domain_size)(*self.parameter_values)

    @classmethod
    @property
    def analytic_solution(cls):
        """Equation is not analytically solvable with the methods available to Sympy, so we
        hard-code the true solution here."""
        return sp.tanh(cls.x/cls.interfacial_width)

    @classmethod
    @property
    def strong_form(cls):
        x, u, b, a, g, k = cls.x, cls.u, cls.b, cls.a, cls.g, cls.k
        return a*u(x) + g*u(x)**3 - k*u(x).diff(x,2)

    @classmethod
    @property
    def weak_form(cls):
        x, u, b, a, g, k = cls.x, cls.u, cls.b, cls.a, cls.g, cls.k
        return (a*u(x) + g*u(x)**3)*b(x) + k*u(x).diff(x)*b(x).diff(x)

    @classmethod
    @property
    def natural_boundary_condition(cls):
        x, u, b, a, g, k = cls.x, cls.u, cls.b, cls.a, cls.g, cls.k
        return -k * u(x).diff(x) * b(x)

    @classmethod
    @property
    def boundary_conditions(cls):
        x, u = cls.x, cls.u
        return {(-cls.domain_size, u(x), -cls.binodal),
               (0, u(x), 0),
               (cls.domain_size, u(x), cls.binodal)}

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    np.set_printoptions(4, suppress=True, linewidth=10000)

    p = GinzburgLandauFlatInterface(-0.25, 0.25, 1)
    print('   solving ODE:', p.strong_form)
    print('exact solution:', p.analytic_solution)

    N = 100
    x = p.numerical_domain_size * np.linspace(-1, 1, N)**3
    order = 2
    w = np.zeros((len(x), order))
    exact = p.analytic_solution.subs({p: v for p, v in zip(p.parameters, p.parameter_values)})
    for c in range(order):
        w[:,c] = sp.lambdify(p.argument, exact.diff(p.argument, c))(x)

    w = p.solve(x, w, print_updates=sys.stderr)
    f = HermiteInterpolator(x, w)
    pl, = plt.plot(x, w[:,0], 'o', mfc='None')

    xx = np.linspace(np.min(x), np.max(x), 2001)
    plt.plot(xx, f(xx), '-', c=pl.get_color())
    try: plt.plot(xx, p.exact_solution(xx), '--')
    except: plt.plot(xx, [p.exact_solution(xx)]*len(xx), '--')
    plt.title(r'$\phi^4$ kink')

    plt.figure()
    plt.plot(xx, p.exact_solution(xx) - f(xx))
    plt.title('errors')

    plt.show()
