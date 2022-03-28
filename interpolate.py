#!/usr/bin/env python3

import numpy as np
from scipy.special import factorial

import sympy as sp
from sympy.polys.specialpolys import interpolating_poly

from functools import lru_cache

class LagrangeInterpolator:
    """A simple 1d interpolator using Lagrange polynomials. This is convenient for
    representing finite element solutions to ODEs."""

    def __init__(self, global_nodes, weights, degree, x=None, u=None):
        assert (len(global_nodes)-1)%degree == 0
        assert len(global_nodes) == len(global_nodes)

        self.global_nodes = global_nodes
        self.weights = weights
        self.degree = degree

        if x is not None: self.interpolating_variable = x
        else: self.interpolating_variable = sp.Symbol('x')
        if u is not None: self.interpolating_function = u
        else: self.interpolating_function = sp.Function('u')

        self.local_node_variables = [sp.Symbol('x%d'%i, real=True, constant=True) for i in range(degree+1)]
        self.local_node_weight_variables = [sp.Symbol('w%d'%i, real=True, constant=True) for i in range(degree+1)]
        poly = interpolating_poly(degree+1, self.xvar,
                                  self.local_node_variables, self.local_node_weight_variables)
        self.local_polynomial = sp.Lambda(self.xvar, poly)

        self.weight_functions = [sp.Lambda(self.xvar, self.local_polynomial(self.xvar).diff(w)) for w in self.local_node_weight_variables]

    @property
    def npoints(self):
        return self.global_nodes.size

    @property
    def xvar(self):
        """Short-hand for independent variable."""
        return self.interpolating_variable

    @property
    def uvar(self):
        """Short-hand for dependent variable."""
        return self.interpolating_function

    @property
    def x(self):
        """Short-hand for node positions."""
        return self.global_nodes

    @property
    def w(self):
        """Short-hand for node weights."""
        return self.weights

    @property
    def nelements(self):
        """Number of distinct elements over which we interpolate via local polynomials
        (i.e. with finite support) of finite degree."""
        return (len(self.global_nodes)-1) // degree

    @property
    def local_indices(self):
        """Bookkeeping for local nodes y0,...,ydeg for each element."""
        return [np.arange(self.npoints+1)[i::self.degree][:self.nelements] for i in range(self.degree+1)]

    @property
    def local_nodes(self):
        """Positions of local nodes within each element.

        Returns:
            List of length degree+1. Each entry is a vector (of size nelements) containing
            positions of the ith node.
        """
        return [self.global_nodes[i] for i in self.local_indices]

    @property
    def element_edges(self):
        """Left edges of each element for binning coordinates to determine their element."""
        return self.x[:-1:self.degree]

    def find_element(self, x):
        assert np.all(x >= self.x[0])
        assert np.all(x <= self.x[-1])
        elements = np.digitize(x, self.element_edges)-1
        return elements

    def element_weight_functions(self, element):
        functions = []

        lower_limit = self.global_nodes[element*self.degree]
        upper_limit = self.global_nodes[(element+1)*self.degree]
        for i in range(self.degree+1):
            w = self.weight_functions[i]
            for j in range(self.degree+1):
                w = w.subs(self.local_node_variables[j], self.global_nodes[self.degree*element + j])

            w = sp.Lambda( self.xvar, sp.Piecewise( (0, self.xvar < lower_limit),
                                                    (0, self.xvar >= upper_limit),
                                                    (w(self.xvar), True) ) )
            functions += [w]

        return functions

    def weight_function(self, i):
        local_index = i % self.degree
        element = i // self.degree

        if local_index > 0:
            w = self.element_weight_functions(element)[local_index]
        else:
            left = self.element_weight_functions(element-1)[-1]
            try: right = self.element_weight_functions(element)[0]
            except: right = sp.Lambda(self.xvar, 0) # fails with boundary on far right, so we just set it to zero there
            location = self.global_nodes[element*self.degree]
            w = sp.Lambda( self.xvar, sp.Piecewise( (left(self.xvar), self.xvar < location),
                                                    (right(self.xvar), True) ) )

        return sp.lambdify(self.xvar, w(self.xvar))

    def __call__(self, x, derivative=0):
        # Determine which element each coordinate is in so we use the correct local polynomial.
        elements = self.find_element(x)
        local_nodes = [self.global_nodes[indices[elements]] for indices in self.local_indices]

        # Build Lagrange-polynomials which are our basis functions:
        u = np.zeros(x.shape)
        for w, indices in zip(self.weight_functions, self.local_indices):
            if derivative > 0: w = sp.Lambda(self.xvar, w(self.xvar).diff(self.xvar, derivative))
            w = sp.lambdify([self.xvar] + self.local_node_variables, w(self.xvar))
            weights = self.weights[indices[elements]]
            u += weights * w(x, *local_nodes)

        return u

class HermiteInterpolatingPolynomial:
    @classmethod
    @lru_cache
    def from_cache(cls, *args, **kwargs):
        """It can be expensive to calculate the coefficients in the polynomial (especially as
        the order is increased), so this construction method caches the resulting polynomials to
        speed up multiple instantiations."""
        return HermiteInterpolatingPolynomial(*args, **kwargs)

    def __init__(self, order, x=sp.Symbol('x'), w=sp.IndexedBase('w')):
        self.x = x
        self.w = w

        # Polynomial must have enough degrees of freedom to match constraints on bouth boundaries
        # (which we call the "order" of the interpolation), so we must double the order.
        self.degree = 2*order-1

        # We work out coefficients for polynomial p = a_n*x**n + ... + a_0 by
        # solving linear system:
        #   M * (a_n, a_(n-1), ... , a_1, a_0) = (f(-1), f(1), f'(-1), f'(1), ..., f^(order)(-1), f^(order)(1))
        # where M is a matrix of values of (x**n, x**(n-1), ... , x, 1) for each equation.

        # First, we work out polynomial coefficients on unit interval [-1, 1], then we determine
        # the transformations onto an arbitrary interval [x0, x1].

        # Build matrix M based on values of p^(n) at x = -1 and 1.
        M = np.zeros((self.degree+1, self.degree+1), dtype=int)
        powers = np.flipud(np.arange(self.degree+1))
        for derivative in range(order):
            # Coefficients at x = 1:
            nonzero = self.degree+1 - derivative
            coefficients = factorial(powers[:nonzero]) / factorial(powers[:nonzero]-derivative)
            M[derivative+order,:nonzero] = coefficients

            # Coefficients at x = -1:
            coefficients[::2] *= -1
            if derivative%2 == 1: coefficients *= -1
            M[derivative,:nonzero] = coefficients

        # Invert the matrix M to obtain the polynomial coefficients.
        M = sp.Matrix(M)
        Minv = M.inv()
        self.weights = np.array([(w[0,i], w[1,i]) for i in range(order)]).T.reshape(-1)
        self.coefficients = Minv*self.weights.reshape(-1,1)

        # Explicit expressions for the polynomial and its decomposition into weight functions.
        self.expression = self.coefficients.dot(x**powers)
        self.weight_functions = [self.expression.diff(w) for w in self.weights]

        # Transformations of node weights for mapping between [-1, 1] and [x0, x1].
        self.x0, self.x1 = sp.symbols('x_0 x_1')
        self.coordinate_transform = 2*(x - self.x0) / (self.x1 - self.x0) - 1
        self.inverse_coordinate_transform = self.x0 + (self.x1 - self.x0) * (1 + self.x) / 2
        q = sp.IndexedBase('q') # temporary label for transformed weights
        transformed_exp = self.expression.subs(w, q).subs(x, self.coordinate_transform)

        # Equate the transformed and untransformed expressions to obtain the transformations
        # of the node weights:
        self.transformed_weights = []
        for derivative in range(order):
            expr1 = self.expression.diff(x, derivative) if derivative > 0 else self.expression
            expr2 = transformed_exp.diff(x, derivative) if derivative > 0 else transformed_exp
            self.transformed_weights += [
                (sp.solve(sp.Eq(expr1.subs(x, -1),
                                expr2.subs(x, self.x0).simplify()),
                          q[0, derivative])[0],
                 sp.solve(sp.Eq(expr1.subs(x, 1),
                                expr2.subs(x, self.x1).simplify()),
                          q[1, derivative])[0])]

        self.transformed_weights = np.array(self.transformed_weights).T.reshape(-1)

    @property
    def weight_variables(self):
        return self.weights.tolist()

    def transform_coordinate(self, x0, x1, x):
        s = sp.lambdify([self.x0, self.x1, self.x], self.coordinate_transform)
        return s(x0, x1, x)

    def transform_weights(self, x0, x1, weights):
        w = sp.lambdify([self.x0, self.x1] + self.weight_variables, self.transformed_weights)
        return np.array(w(x0, x1, *weights.T)).T

    @property
    def general_expression(self):
        replacements = {w1: w2 for w1, w2 in zip(self.weight_variables,
                                                 self.transformed_weights)}
        replacements[self.x] = self.coordinate_transform
        return self.expression.subs(replacements).simplify()

    @property
    def general_weight_functions(self):
        replacements = {w1: w2 for w1, w2 in zip(self.weight_variables,
                                                 self.transformed_weights)}
        replacements[self.x] = self.coordinate_transform

        weight_functions = []
        for w in self.weight_functions:
            weight_functions += [w.subs(replacements).simplify()]

        return weight_functions

class HermiteInterpolator:
    """An interpolator on the line interval [-1,1] which matches values and derivatives on the
    boundaries. This is helpful for representing finite element solutions to ODEs where
    continuity of derivatives is required."""

    def __init__(self, nodes, weights):#, xvar=None, u=None):
        try:
            shape = weights.shape
        except:
            weights = np.vstack(weights).T
            shape = weights.shape

        num_nodes, self.order = shape
        assert len(nodes) == num_nodes

        self.nodes = nodes
        self.weights = weights

        self.interpolating_polynomial = HermiteInterpolatingPolynomial.from_cache(self.order)

    @property
    def npoints(self):
        return self.global_nodes.size

    @property
    def nelements(self):
        """Number of distinct elements over which we interpolate via local polynomials
        (i.e. with finite support) of finite degree."""
        return self.npoints - 1

    @property
    def x(self):
        """Short-hand for node positions."""
        return self.nodes

    @property
    def w(self):
        """Short-hand for node weights."""
        return self.weights

    def find_element(self, x):
        assert np.all(x >= self.x[0])
        assert np.all(x <= self.x[-1])
        elements = np.digitize(x, self.x[:-1])-1
        return elements

    @property
    def local_variable(self):
        return self.interpolating_polynomial.x

    @property
    def local_node_variables(self):
        return self.interpolating_polynomial.weight_variables

    def __call__(self, x):#, derivative=0):
        # Determine which element each coordinate is in so we use the correct local polynomial.
        elements = self.find_element(x)
        weights = np.hstack((self.weights[elements], self.weights[elements+1]))

        # Transform into the local coordinate bounded in [-1, 1] for each element:
        xleft, xright = self.x[elements], self.x[elements+1]
        s = self.interpolating_polynomial.transform_coordinate(xleft, xright, x)
        weights = self.interpolating_polynomial.transform_weights(xleft, xright, weights)

        f = sp.lambdify([self.local_variable] + self.local_node_variables, self.interpolating_polynomial.expression)
        return f(s, *weights.T)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure()
    order = 2
    poly = HermiteInterpolatingPolynomial(order)
    x = np.linspace(-1, 1, 1000)
    for i,w in enumerate(poly.weight_functions):
        f = sp.lambdify(poly.x, w)
        plt.plot(x, f(x), label=('weight function %d' % i))
    plt.legend(loc='best')
    plt.title('Hermite weight functions (order=%d)' % order)

    plt.figure()
    x = np.linspace(0, 1, 2)
    y = x**5
    yp = 5*x**4
    ypp = 20*x**3
    f = HermiteInterpolator(x, [y,yp,ypp])
    pl, = plt.plot(x, y, 'o', mfc='None')
    x = np.linspace(x[0], x[-1], 1000)
    plt.plot(x, f(x), lw=0.5, c=pl.get_color(), label='interpolation')
    plt.plot(x, x**5, '--', lw=0.5, label='exact')
    plt.legend(loc='best')
    plt.title(r'Hermite interpolation of $y=x^5$')

    plt.show()
