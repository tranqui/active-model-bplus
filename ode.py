#!/usr/bin/env python3

import numpy as np
from scipy.linalg import solve_banded
from scipy.integrate._quadrature import _cached_roots_legendre
import sympy as sp

from functools import lru_cache

from interpolate import HermiteInterpolatingPolynomial, HermiteInterpolator
import differentiate

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
    def expression(cls):
        """Symbolic expression that must be defined in derived expressions."""
        raise NotImplementedError('should only be called from derived expression!')

    @classmethod
    def expr(cls):
        """Alias for expression."""
        return cls.expression()

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
        expr = cls.expression().diff([args, n])
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

class WeakFormProblem1d:
    argument = sp.Symbol('x')
    unknown_function = sp.Function('u')
    basis_function = sp.Function('b')

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

    @classmethod
    @property
    def boundary_conditions(cls):
        raise NotImplementedError('boundary conditions have not been specified!')

    @classmethod
    @property
    @lru_cache
    def analytic_solution(cls):
        bcs = {expr.subs(cls.argument, point): value for point, expr, value in cls.boundary_conditions}
        return sp.dsolve(cls.strong_form, cls.unknown_function(cls.argument), ics=bcs).rhs

    @classmethod
    @property
    @lru_cache
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
    @lru_cache
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
        left, right = ([e.subs(cls.argument, x) for e in expressions] for x in (polynomial.x0, polynomial.x1))
        return left, right

    @classmethod
    @lru_cache
    def compiled_natural_boundary_condition_expressions(cls, order=1, *args, **kwargs):
        expressions = cls.natural_boundary_condition_expressions(order, *args, **kwargs)
        compiled_expressions = []

        arguments = cls.elemental_variables(order)
        for boundary in expressions: # loop over left and right expressions
            compiled_expressions += [[sp.lambdify(arguments, e) for e in boundary]]
        return compiled_expressions

    @classmethod
    @lru_cache
    def natural_boundary_condition_jacobians(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)
        J = []
        left, right = cls.natural_boundary_condition_expressions(order)
        for expressions in [left, right]:
            J += [[[e.diff(w).doit() for w in polynomial.weight_variables] for e in expressions]]
        return J

    @classmethod
    @lru_cache
    def compiled_natural_boundary_condition_jacobians(cls, order=1, *args, **kwargs):
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
        return compiled_jacobians

    @classmethod
    @lru_cache
    def boundary_condition_expressions(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)
        local_coordinate = sp.Function('s')

        expressions = []
        for point, lhs, rhs in cls.boundary_conditions:
            expression = lhs - rhs
            expression = expression.subs(
                {cls.unknown_function: sp.Lambda(cls.argument, polynomial.general_expression),
                 cls.argument: point})

            expressions += [(point, expression)]

        return expressions

    @classmethod
    @lru_cache
    def compiled_boundary_condition_expressions(cls, order=1, *args, **kwargs):
        expressions = cls.boundary_condition_expressions(order, *args, **kwargs)
        compiled_expressions = []

        arguments = cls.elemental_variables(order)
        for point, expression in expressions:
            compiled_expressions += [(point, sp.lambdify(arguments, expression))]
        return compiled_expressions

    @classmethod
    @lru_cache
    def boundary_condition_jacobians(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)
        J = []
        for point, expression in cls.boundary_condition_expressions(order):
            J += [(point, [expression.diff(w).doit() for w in polynomial.weight_variables])]
        return J

    @classmethod
    @lru_cache
    def compiled_boundary_condition_jacobians(cls, order=1, *args, **kwargs):
        jacobians = cls.boundary_condition_jacobians(order, *args, **kwargs)
        compiled_jacobians = []
        for point, row in jacobians:
            compiled_row = []
            for expression in row:
                compiled_row += [sp.lambdify(cls.elemental_variables(order), expression)]
            compiled_jacobians += [(point, compiled_row)]
        return compiled_jacobians

    @classmethod
    @lru_cache
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
        roots = [x(r) for r in roots]
        weights = [w*dxds(w) for w in weights]

        for i,w in enumerate(polynomial.general_weight_functions):
            specific_integrand = basic_integrand.subs(cls.basis_function,
                                                      sp.Lambda(cls.argument, w)).doit()
            result = sum([w*specific_integrand.subs(cls.argument, p).doit() for p, w in zip(roots, weights)])

            residuals += [result]

        return residuals

    @classmethod
    @lru_cache
    def compiled_elemental_residuals(cls, order, *args, **kwargs):
        residuals = cls.elemental_residuals(order, *args, **kwargs)
        compiled_residuals = []
        for expression in residuals:
            compiled_residuals += [sp.lambdify(cls.elemental_variables(order), expression)]
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
                R[c//order, c%order] += l(xleft[0], xright[0], *w[0], *self.parameter_values)
                R[-2 + c//order, c%order] += r(xleft[-1], xright[-1], *w[-1], *self.parameter_values)

        # Evaluate residual contributions from specific boundary conditions.
        element_edges = nodes[:-1]
        bcs = {}
        for point, func in self.compiled_boundary_condition_expressions(order):
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
    @lru_cache
    def elemental_jacobians(cls, order=1):
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, cls.argument)
        J = []
        for R in cls.elemental_residuals(order):
            J += [[R.diff(w).doit() for w in polynomial.weight_variables]]
        return J

    @classmethod
    @lru_cache
    def compiled_elemental_jacobians(cls, order, *args, **kwargs):
        jacobians = cls.elemental_jacobians(order, *args, **kwargs)
        compiled_jacobians = []
        for row in jacobians:
            compiled_row = []
            for expression in row:
                compiled_row += [sp.lambdify(cls.elemental_variables(order), expression)]
            compiled_jacobians += [compiled_row]
        return compiled_jacobians

    def jacobian(self, nodes, weights, *args, **kwargs):
        nelements, order = weights.shape
        nelements -= 1

        J = np.zeros((2*(order+1)+1, nelements+1, order))
        polynomial = HermiteInterpolatingPolynomial.from_cache(order, self.argument)
        variables = polynomial.weight_variables
        functions = self.compiled_elemental_jacobians(order, *args, **kwargs)

        xleft, xright = nodes[:-1], nodes[1:]
        w = np.hstack((weights[:-1], weights[1:]))

        for var, row in zip(variables, functions):
            boundary, deriv = var.indices

            for i, func in enumerate(row):
                i = len(J)-2-i
                j = func(xleft, xright, *w.T, *self.parameter_values)
                if boundary == 0: J[i-order,:-1,deriv] += j
                elif boundary == 1: J[i,1:,deriv] += j
                else: raise RuntimeError('unknown variable indices during residual calculation!')

        J = J.reshape(len(J), -1)

        # Apply natural boundary condition needed to make weak form valid (these conditions
        # arise from surface terms left over from e.g. integration by parts).
        if self.natural_boundary_condition:
            left, right = self.compiled_natural_boundary_condition_jacobians(order)
            for c, (l, r) in enumerate(zip(left, right)):
                l = np.flipud([f(xleft[0], xright[0], *w[0], *self.parameter_values) for f in l])
                r = np.flipud([f(xleft[-1], xright[-1], *w[-1], *self.parameter_values) for f in r])

                starting_row = len(J)-1-len(variables)
                eqn = -2*order+c
                rows = np.arange(starting_row, starting_row+len(l)) % len(J)
                J[rows, eqn] += r

                eqn = c
                starting_row -= order
                rows = np.arange(starting_row, starting_row+len(l)) % len(J)
                J[rows, eqn] += l

        # Apply boundary conditions.
        element_edges = nodes[:-1]
        bcs = {}
        for point, row in self.compiled_boundary_condition_jacobians(order):
            element = np.digitize(point, element_edges)-1
            closest_node = np.abs(nodes - point).argmin()
            xleft, xright = nodes[element:element+2]
            values = np.flipud([func(xleft, xright, *w[element], *self.parameter_values) for func in row])

            starting_row = len(J)-1-len(variables)
            boundary_on_left = closest_node == element
            if boundary_on_left: starting_row -= order

            entry = np.zeros(len(J))
            entry[starting_row:starting_row+len(values)] = values
            try: bcs[closest_node] += [entry]
            except: bcs[closest_node] = [entry]

        # Ensure boundary conditions are placed on distinct rows
        for node, conditions in bcs.items():
            for c, entry in enumerate(conditions):
                index = node*order + c
                J[:,index] = entry

        # Each column currently contains the Jacobian entries for each residual, but
        # we have to shift these to correspond to the matrix format needed by
        # scipy.linalg.solve_banded.

        # Shift columns for each type of local weight so that they align with their own equations.
        for c in range(1,order):
            J[:,c::order] = np.roll(J[:,c::order], c, axis=0)

        # Shift the elements for each equation so elements of a single equation are stored
        # diagonally (cf. scipy.linalg.solve_banded which documents the matrix storage format).
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

        #u = order
        for i in range(nnodes):
            for j in range(nnodes):
                if (u + i - j) < 0 or (u + i - j) >= len(J): continue
                Jfull[i, j] = J[u + i - j, j]
        return Jfull

    def solve(self, nodes, weights):
        nelements, order = weights.shape
        nelements -= 1

        R = self.residuals(nodes, weights)

        iters = 0
        while np.linalg.norm(R) > 1e-6 and iters < 5:
            print(iters, R)
            J = self.jacobian(nodes, weights)

            # full = self.full_jacobian(J)
            # num = self.numerical_jacobian(self.weights)
            # print(full)
            # print(num)
            # print(full-num)
            # assert np.allclose(full, num)

            weights = weights + solve_banded((order+1, order+1), J, -R).reshape(weights.shape)
            R = self.residuals(nodes, weights)
            iters += 1

        print()
        print(nodes)
        print(weights)
        print(R)
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

class DummyProblem(WeakFormProblem1d):
    a = sp.Symbol('a')
    parameters = [a]

    @classmethod
    @property
    def strong_form(cls):
        x, u, b, a = cls.x, cls.u, cls.b, cls.a
        #return u(x).diff(x,4)
        #return a
        #return u(x).diff(x,2) +
        return u(x).diff(x,2) - a

    @classmethod
    @property
    def weak_form(cls):
        x, u, b, a = cls.x, cls.u, cls.b, cls.a
        #return u(x).diff(x,2)*b(x).diff(x,2) + a*u(x)*b(x)
        #return u(x).diff(x,2)*b(x).diff(x,2)
        return -u(x).diff(x)*b(x).diff(x) - a*b(x)
        #return u(x).diff(x)*b(x) - a*b(x) #a*u(x)*b(x)
        #return u(x).diff(x,2)*b(x)#.diff(x,1) #a*b(x)#*u(x)*b(x)

    @classmethod
    @property
    def natural_boundary_condition(cls):
        x, u, b, a = cls.x, cls.u, cls.b, cls.a
        return u(x).diff(x) * b(x)

    @classmethod
    @property
    def boundary_conditions(cls):
        x, u, b, a = cls.x, cls.u, cls.b, cls.a
        up = lambda x2: u(x).diff(x).subs(x, x2)
        #return {(0, u(x), 1), (1, up(x), -1)}
        #return {(0, u(x), 1), (0, up(x), 0), (1, u(x), 0), (1, up(x), 0)}
        return {(0, u(x), 1), (1, u(x), 1)}
        #return {(0, u(x), 1)}
        #return {(0, up(x), -1)}
        #return {u(0): 1, up(0): 0, u(1): 0, up(1): 0}
        #return {u(0): 1, u(1): 0}

# print(HeatEquation.elemental_residuals(1))
# print(HeatEquation.elemental_jacobian(1))
# print()
# print(HeatEquation.elemental_residuals(2))
# print()
# print(HeatEquation.elemental_jacobian(2))

p = DummyProblem(2)
print(p.analytic_solution)
#import sys; sys.exit(0)
#p = HeatEquation()
x = np.linspace(0, 1, 5)
w = np.ones((len(x), 2))
#print(p.elemental_residuals(2))
np.set_printoptions(4, suppress=True, linewidth=10000)
#print(p.residuals(x, w))

J1 = p.numerical_jacobian(x, w)
J = p.jacobian(x, w)
J2 = p.full_jacobian(J)

print(J1)
print()
print(J2)
print()
print(J1-J2)
#import sys; sys.exit(0)

import matplotlib.pyplot as plt

w = p.solve(x, w)
f = HermiteInterpolator(x, w)
pl, = plt.plot(x, w[:,0], 'o', mfc='None')

xx = np.linspace(np.min(x), np.max(x), 1000)
plt.plot(xx, f(xx), '-', c=pl.get_color())
try: plt.plot(xx, p.exact_solution(xx), '--')
except: plt.plot(xx, [p.exact_solution(xx)]*len(xx), '--')
plt.show()

#print()
#print(J)

#print()
#print(HeatEquation.residuals(4))

#print()
#print(DummyProblem.residuals(2))
import sys; sys.exit(0)

class FiniteElement1dODESolver(HermiteInterpolator):
    def __init__(self, problem, nodes, order=1, weights_guess=None,
                 integrate_analytically=False, print_updates=None):
        """
        Args:
            equation: weak-form integrand for ODE we are solving.
            test_function: symbol for test function specified in equation
        """
        if weights_guess is None:
            weights_guess = np.zeros((nodes.size,order))
        else:
            assert weights_guess.shape == (len(nodes), order)

        super().__init__(nodes, weights_guess)
        self.problem = problem

        self.integrate_analytically = integrate_analytically
        self.print_updates = print_updates

        if self.print_updates:
            R = self.problem.residuals(self.weights)
            J = self.problem.jacobian(self.weights)

            self.print_updates.write('iterating to solve ODE...\n')
            np.set_printoptions(4, suppress=True, linewidth=10000)

            self.print_updates.write(repr(R) + '\n')
            self.print_updates.write(repr(self.full_jacobian(J)) + '\n')

        R = self.residuals(self.weights)
        iters = 0
        while np.linalg.norm(R) > 1e-6 and iters < 5:
            print(iters, R)
            J = self.problem.jacobian(self.weights)

            full = self.full_jacobian(J)
            num = self.numerical_jacobian(self.weights)
            print(full)
            print(num)
            print(full-num)
            assert np.allclose(full, num)

            self.weights += solve_banded((self.degree, self.degree), J, -R).reshape(self.weights.shape)
            R = self.problem.residuals(self.weights)
            iters += 1

        if self.print_updates:
            self.print_updates.write('final residuals: %r\n' % self.global_residuals(self.weights))

        #assert np.allclose(self.global_residuals(self.weights),
        #                   np.zeros(self.weights.shape))
        sys.exit(0)

    @property
    def element_integration_limits(self):
        return self.xvar, self.local_node_variables[0], self.local_node_variables[-1]

    def compile_function(self, expr):
        return sp.lambdify(self.local_node_variables + self.local_node_weight_variables, expr)

    def residuals(self, weights):
        R = np.zeros(self.nodes.shape)
        local_weights = [weights[i] for i in self.local_indices]
        for res, nodes in zip(self.local_residuals, self.local_indices):
            R[nodes] += res(*self.local_nodes, *local_weights)
        R[0] = weights[0]-1
        # m = len(self.global_nodes)//2
        # R[m] = weights[m]-0.5
        R[-1] = weights[-1]
        return R

    def jacobian(self, weights):
        J = np.zeros((2*self.degree+1, len(self.global_nodes)))
        diagonal = self.degree
        local_weights = [weights[i] for i in self.local_indices]
        for local, (jac, nodes) in enumerate(zip(self.local_jacobians,
                                                 self.local_indices)):
            for row, func in enumerate(jac):
                offset = local - row
                row = diagonal + offset
                J[row,nodes-offset] += func(*self.local_nodes, *local_weights)

        # Boundary conditions.
        def clear_jacobian_row(self, J, row):
            l = max(-self.degree, i-len(J))
            u = min(self.degree, i)
            for i in range(l, 1+u): J[diagonal+i,-i+row] = 0

        for i in range(-self.degree,0): J[diagonal+i,-i] = 0
        J[diagonal,0] = 1

        # m = len(self.global_nodes)//2
        # for i in range(-self.degree,self.degree+1): J[diagonal+i,-i+m] = 0
        # J[diagonal,m] = 1

        for i in range(self.degree+1): J[diagonal+i,-i-1] = 0
        J[diagonal,-1] = 1

        return J

    def numerical_jacobian(self, weights):
        return gradient(self.residuals, weights, dx=1e-4).T

    def full_jacobian(self, J):
        """Convert banded jacobian into full square matrix. Useful for testing."""
        Jfull = np.zeros((len(self.global_nodes), len(self.global_nodes)))
        for i in range(len(Jfull)):
            for j in range(len(Jfull)):
                if (self.degree + i - j) < 0 or (self.degree + i - j) > 2*self.degree: continue
                Jfull[i,j] = J[self.degree + i - j, j]
        return Jfull

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    print_updates = sys.stderr
    #print_updates = None

    # plt.figure()
    # poly = HermiteInterpolatingPolynomial(2)
    # x = np.linspace(-1, 1, 1000)
    # for w in poly.weight_functions:
    #     f = sp.lambdify(poly.x, w)
    #     plt.plot(x, f(x))
    # #plt.show()

    # #sys.exit(0)

    # plt.figure()
    # x = np.linspace(0, 1, 2)
    # y = x**5
    # yp = 5*x**4
    # ypp = 20*x**3
    # f = HermiteInterpolator(x, [y,yp,ypp])
    # pl, = plt.plot(x, y, 'o', mfc='None')
    # x = np.linspace(x[0], x[-1], 1000)
    # plt.plot(x, f(x), lw=0.5, c=pl.get_color())
    # plt.plot(x, x**5, '--', lw=0.5)
    # # plt.figure()
    # # plt.plot(x, f(x)-x**5)
    # plt.show()

    # sys.exit(0)

    #degree = 2
    #nelements = 3
    #xmax = nelements*degree
    #xmax = 1

    #print(sp.dsolve(eqn, u(x), ics={u(-sp.oo): 1, u(0): 0.5, u(sp.oo): 0}))
    #print(eqn)
    #print(solution)
    #weak_form = -(f.diff(u(x)) - u(x).diff(x,2)) * w(x).diff()
    #weak_form = u(x).diff(x,2) * w(x).diff()
    #print(weak_form)
    #sys.exit(0)
    #print(eqn.subs(u(x), sp.tanh(x/a)).doit().simplify())
    #problem = DummyProblem(1)
    problem = HeatEquation()
    nodes = np.linspace(0, 1, 100)
    #solver = FiniteElement1dODESolver(problem, nodes, order=2, print_updates=print_updates)

    #pl, = plt.plot(solver.x, solver.w, 'o', mfc='None', label='central nodes')
    #plt.plot(solver.x[::degree], solver.w[::degree], 'o', c=pl.get_color(), label='end nodes')
    x = np.linspace(0, nodes[-1], 1000)
    #plt.plot(x, solver(x), '-', c=pl.get_color(), zorder=-10, label='FE solution')

    exact = sp.lambdify(problem.argument, problem.exact_solution)
    plt.plot(x, exact(x), '--')#, c=pl.get_color(), zorder=-10, label='exact solution')
    plt.legend(loc='best')

    plt.show()
