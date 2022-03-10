#!/usr/bin/env python3

import sys
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import factorial
import sympy as sp
from sympy.polys.specialpolys import interpolating_poly

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
    def compiled_function(cls, deriv=0):
        """Compile the function so that it can be numerically evaluated with native python
        structures, if not already done so for this expression.

        We cache the result (at the class-level) to prevent unnecessary recompilations for many
        instances of the same expression (with e.g. different choices of parameters).

        Args:
            deriv: order of derivative to compile into an evaluatable function
        """
        try: return cls._compiled_function[deriv]
        except:
            func = sp.lambdify(cls.variables(), cls.numerical_implementation(deriv))
            if '_compiled_function' not in cls.__dict__:
                cls._compiled_function = dict()
            cls._compiled_function[deriv] = func
            return cls._compiled_function[deriv]

    def __call__(self, *args, deriv=0):
        """Numerically evaluate expression."""
        cls = self.__class__
        f = cls.compiled_function(deriv)
        with np.errstate(invalid='ignore'):
            return f(*args + self.parameter_values)

# class DiffusionResidual(Expression):
#     arguments = []
#     parameters = [sp.Symbol('D')]

#     @classmethod
#     def expression(cls):
#         return u(x).diff(x,3)

# sys.exit(0)

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
    # def __init__(self, global_nodes, weights, degree, x=None, u=None):
    #     assert (len(global_nodes)-1)%degree == 0
    #     assert len(global_nodes) == len(global_nodes)

    #     self.global_nodes = global_nodes
    #     self.weights = weights
    #     self.degree = degree

    #     if x is not None: self.interpolating_variable = x
    #     else: self.interpolating_variable = sp.Symbol('x')
    #     if u is not None: self.interpolating_function = u
    #     else: self.interpolating_function = sp.Function('u')

    #     self.local_node_variables = [sp.Symbol('x%d'%i, real=True, constant=True) for i in range(degree+1)]
    #     self.local_node_weight_variables = [sp.Symbol('w%d'%i, real=True, constant=True) for i in range(degree+1)]
    #     poly = interpolating_poly(degree+1, self.xvar,
    #                               self.local_node_variables, self.local_node_weight_variables)
    #     self.local_polynomial = sp.Lambda(self.xvar, poly)

    #     self.weight_functions = [sp.Lambda(self.xvar, self.local_polynomial(self.xvar).diff(w)) for w in self.local_node_weight_variables]

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
        self.nodes = np.array([(w[0,i], w[1,i]) for i in range(order)]).T.reshape(-1)
        Minv = np.linalg.inv(M)
        self.coefficients = Minv.dot(self.nodes)

        # Explicit expressions for the polynomial and its decomposition into weight functions.
        self.expression = self.coefficients.dot(x**powers)
        self.weight_functions = [self.expression.diff(w) for w in self.nodes]

    @property
    def weight_variables(self):
        return self.nodes.tolist()

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

        self.interpolating_polynomial = HermiteInterpolatingPolynomial(self.order)
        if True: return

        sys.exit(0)

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
    def nelements(self):
        """Number of distinct elements over which we interpolate via local polynomials
        (i.e. with finite support) of finite degree."""
        return self.npoints - 1

    # @property
    # def xvar(self):
    #     """Short-hand for independent variable."""
    #     return self.interpolating_polynomial.x

    # @property
    # def uvar(self):
    #     """Short-hand for dependent variable."""
    #     return self.interpolating_function

    @property
    def x(self):
        """Short-hand for node positions."""
        return self.nodes

    @property
    def w(self):
        """Short-hand for node weights."""
        return self.weights

    # @property
    # def local_indices(self):
    #     """Bookkeeping for local nodes y0,...,ydeg for each element."""
    #     return [np.arange(self.npoints+1)[i::self.degree][:self.nelements] for i in range(self.degree+1)]

    # @property
    # def local_nodes(self):
    #     """Positions of local nodes within each element.

    #     Returns:
    #         List of length degree+1. Each entry is a vector (of size nelements) containing
    #         positions of the ith node.
    #     """
    #     return [self.global_nodes[i] for i in self.local_indices]

    # @property
    # def element_edges(self):
    #     """Left edges of each element for binning coordinates to determine their element."""
    #     return self.x[:-1:self.degree]

    def find_element(self, x):
        assert np.all(x >= self.x[0])
        assert np.all(x <= self.x[-1])
        elements = np.digitize(x, self.x[:-1])-1
        return elements

    # def element_weight_functions(self, element):
    #     functions = []

    #     lower_limit = self.global_nodes[element*self.degree]
    #     upper_limit = self.global_nodes[(element+1)*self.degree]
    #     for i in range(self.degree+1):
    #         w = self.weight_functions[i]
    #         for j in range(self.degree+1):
    #             w = w.subs(self.local_node_variables[j], self.global_nodes[self.degree*element + j])

    #         w = sp.Lambda( self.xvar, sp.Piecewise( (0, self.xvar < lower_limit),
    #                                                 (0, self.xvar >= upper_limit),
    #                                                 (w(self.xvar), True) ) )
    #         functions += [w]

    #     return functions

    # def weight_function(self, i):
    #     local_index = i % self.degree
    #     element = i // self.degree

    #     if local_index > 0:
    #         w = self.element_weight_functions(element)[local_index]
    #     else:
    #         left = self.element_weight_functions(element-1)[-1]
    #         try: right = self.element_weight_functions(element)[0]
    #         except: right = sp.Lambda(self.xvar, 0) # fails with boundary on far right, so we just set it to zero there
    #         location = self.global_nodes[element*self.degree]
    #         w = sp.Lambda( self.xvar, sp.Piecewise( (left(self.xvar), self.xvar < location),
    #                                                 (right(self.xvar), True) ) )

    #     return sp.lambdify(self.xvar, w(self.xvar))

    @property
    def local_variable(self):
        return self.interpolating_polynomial.x

    @property
    def local_node_variables(self):
        return self.interpolating_polynomial.weight_variables

    def __call__(self, x):#, derivative=0):
        # Determine which element each coordinate is in so we use the correct local polynomial.
        elements = self.find_element(x)
        # Transform into the local coordinate bounded in [-1, 1] for each element:
        xleft, xright = self.x[elements], self.x[elements+1]
        local_coordinates = 2*(x - xleft) / (xright - xleft) - 1

        element_weights = np.hstack((self.weights[elements], self.weights[elements+1]))
        f = sp.lambdify([self.local_variable] + self.local_node_variables, self.interpolating_polynomial.expression)
        return f(local_coordinates, *element_weights.T)

        # # Build Lagrange-polynomials which are our basis functions:
        # u = np.zeros(x.shape)
        # for w in self.weight_functions:
        #     if derivative > 0: w = w(self.xvar).diff(self.xvar, derivative)
        #     w = sp.lambdify([local_variable] + self.local_node_variables, w)
        #     weights = self.weights[indices[elements]]
        #     u += weights * w(x, *local_nodes)

        # return u

class FiniteElement1dODESolver(LagrangeInterpolator):
    def __init__(self, equation, test_function, u, x, global_nodes, degree, u_guess=None,
                 integrate_analytically=False, print_updates=None):
        """
        Args:
            equation: weak-form integrand for ODE we are solving.
            test_function: symbol for test function specified in equation
        """
        if u_guess is None: u_guess = np.zeros(global_nodes.shape)
        super().__init__(global_nodes, u_guess, degree, x=x, u=u)
        self.equation = equation
        self.test_function = test_function
        self.integrate_analytically = integrate_analytically
        self.print_updates = print_updates

        if self.print_updates:
            R = self.global_residuals(self.weights)
            J = self.global_jacobian(self.weights)

            self.print_updates.write('iterating to solve ODE...\n')
            np.set_printoptions(4, suppress=True, linewidth=10000)

            self.print_updates.write(repr(R) + '\n')
            self.print_updates.write(repr(self.full_jacobian(J)) + '\n')

        R = self.global_residuals(self.weights)
        iters = 0
        while np.linalg.norm(R) > 1e-6 and iters < 5:
            print(iters, R)
            J = self.global_jacobian(self.weights)

            full = self.full_jacobian(J)
            num = self.numerical_jacobian(self.weights)
            print(full)
            print(num)
            print(full-num)
            assert np.allclose(full, num)

            self.weights += solve_banded((self.degree, self.degree), J, -R)
            R = self.global_residuals(self.weights)
            iters += 1

        if self.print_updates:
            self.print_updates.write('final residuals: %r\n' % self.global_residuals(self.weights))

        #assert np.allclose(self.global_residuals(self.weights),
        #                   np.zeros(self.weights.shape))

    @property
    def element_integration_limits(self):
        return self.xvar, self.local_node_variables[0], self.local_node_variables[-1]

    @property
    def local_residual_expressions(self):
        """Explicit expressions the contributions to residuals from nodes within
        a single element."""
        try: return self._local_residual_expressions
        except: pass

        self._local_residual_expressions = []

        residual_integrand = self.equation.subs(self.uvar, self.local_polynomial).doit()

        for i,w in enumerate(self.weight_functions):
            specific_integrand = residual_integrand.subs(self.test_function, w).doit()

            if self.integrate_analytically:
                if self.print_updates:
                    self.print_updates.write('integrating residual %d/%d...\n' % (i+1, self.degree+1))
                specific_integrand = specific_integrand.expand().collect(self.xvar)
                result = sp.integrate(specific_integrand, self.element_integration_limits).doit()
                if self.print_updates:
                    self.print_updates.write('simplifying residual %d/%d...\n' % (i+1, self.degree+1))
                result = result.simplify()

            else:
                if self.print_updates:
                    self.print_updates.write('integrating residual %d/%d...\n' % (i+1, self.degree+1))

                x, a, b = self.element_integration_limits
                # Change limits of integration in x from [a,b] to [-1,1].
                substitution = a + (b-a)*(1+x)/2
                specific_integrand = specific_integrand.subs(x, substitution).doit()
                specific_integrand *= substitution.diff(x)

                # We use a fixed-order Gaussian quadrature rule for the integration:
                from scipy.integrate._quadrature import _cached_roots_legendre
                roots, weights = _cached_roots_legendre(2*self.degree+1)
                result = sum([w*specific_integrand.subs(x, p).doit() for p, w in zip(roots, weights)])

                # if self.print_updates:
                #     self.print_updates.write('simplifying residual %d/%d...\n' % (i+1, self.degree+1))
                # result = result.simplify()

            self._local_residual_expressions += [result]

        return self._local_residual_expressions

    @property
    def local_jacobian_expressions(self):
        """Explicit expressions for the jacobian entries corresponding to a single element."""
        try: return self._local_jacobian_expressions
        except: pass

        self._local_jacobian_expressions = []

        for i,R in enumerate(self.local_residual_expressions):
            if self.print_updates:
                self.print_updates.write('finding jacobian entries for local node %d/%d...\n' % (i+1, self.degree+1))
            self._local_jacobian_expressions += [[R.diff(w).doit() for w in self.local_node_weight_variables]]

            if self.integrate_analytically:
                if self.print_updates:
                    self.print_updates.write('simplifying jacobian entries for local node %d/%d...\n' % (i+1, self.degree+1))
                self._local_jacobian_expressions = [[result.simplify() for result in entries] for entries in self._local_jacobian_expressions]

        return self._local_jacobian_expressions

    def compile_function(self, expr):
        return sp.lambdify(self.local_node_variables + self.local_node_weight_variables, expr)

    @property
    def local_residuals(self):
        try: return self._local_residuals
        except: pass

        if self.print_updates:
            self.local_residual_expressions # call so updates there occur first
            self.print_updates.write('compiling residuals...\n')

        self._local_residuals = [self.compile_function(expr) for expr in self.local_residual_expressions]
        return self._local_residuals

    @property
    def local_jacobians(self):
        try: return self._local_jacobians
        except: pass

        if self.print_updates:
            self.local_jacobian_expressions # call so updates there occur first
            self.print_updates.write('compiling jacobians...\n')

        self._local_jacobians = [[self.compile_function(expr) for expr in row] for row in self.local_jacobian_expressions]
        return self._local_jacobians

    def global_residuals(self, weights):
        R = np.zeros(self.global_nodes.shape)
        local_weights = [weights[i] for i in self.local_indices]
        for res, nodes in zip(self.local_residuals, self.local_indices):
            R[nodes] += res(*self.local_nodes, *local_weights)
        R[0] = weights[0]-1
        # m = len(self.global_nodes)//2
        # R[m] = weights[m]-0.5
        R[-1] = weights[-1]
        return R

    def global_jacobian(self, weights):
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
        from cripes.differentiate import gradient
        return gradient(self.global_residuals, weights, dx=1e-4).T

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

    print_updates = sys.stderr
    #print_updates = None

    plt.figure()
    poly = HermiteInterpolatingPolynomial(2)
    x = np.linspace(-1, 1, 1000)
    for w in poly.weight_functions:
        f = sp.lambdify(poly.x, w)
        plt.plot(x, f(x))
    #plt.show()

    #sys.exit(0)

    plt.figure()
    x = np.linspace(0, 1, 2)
    y = x**5
    yp = 5*x**4
    ypp = 20*x**3
    f = HermiteInterpolator(x, [y,yp,ypp])
    pl, = plt.plot(x, y, 'o', mfc='None')
    x = np.linspace(0, 1, 1000)
    plt.plot(x, f(x), lw=0.5, c=pl.get_color())
    plt.plot(x, x**5, '--', lw=0.5)
    plt.show()

    sys.exit(0)

    degree = 2
    nelements = 3
    #xmax = nelements*degree
    xmax = 1
    nodes = np.linspace(0, xmax, 1+nelements*degree)

    a = sp.Symbol('a')
    x = sp.Symbol('x')
    u = sp.Function('u')
    w = sp.Function('w')

    #f = -0.25*(0.5*u(x)**2 - 0.25*u(x)**4)
    #eqn = f.diff(u(x),2)*u(x).diff() - u(x).diff(x,3)

    eqn = u(x).diff(x,2) + u(x) #+ 8
    solution = sp.dsolve(eqn, u(x), ics={u(0): 1, u(xmax): 0})
    #solution = sp.dsolve(eqn, u(x), ics={u(0): 1, u(xmax): 0})
    #weak_form = -0.5*(u(x).diff(x,2)*w(x).diff() - u(x).diff(x,1)*w(x).diff(x,2)) + 8*w(x)
    weak_form = -u(x).diff(x,1)*w(x).diff(x) + u(x)*w(x) #+ u(x)*w(x) + 8*w(x)
    exact = sp.lambdify(x, solution.rhs)

    #print(sp.dsolve(eqn, u(x), ics={u(-sp.oo): 1, u(0): 0.5, u(sp.oo): 0}))
    print(eqn)
    print(solution)
    #weak_form = -(f.diff(u(x)) - u(x).diff(x,2)) * w(x).diff()
    #weak_form = u(x).diff(x,2) * w(x).diff()
    print(weak_form)
    #sys.exit(0)
    #print(eqn.subs(u(x), sp.tanh(x/a)).doit().simplify())
    u_guess = np.zeros(len(nodes))
    solver = FiniteElement1dODESolver(weak_form, w, u, x, nodes, degree, u_guess, print_updates=print_updates)

    import matplotlib.pyplot as plt
    pl, = plt.plot(solver.x, solver.w, 'o', mfc='None', label='central nodes')
    plt.plot(solver.x[::degree], solver.w[::degree], 'o', c=pl.get_color(), label='end nodes')
    x = np.linspace(0, nodes[-1], 1000)
    plt.plot(x, solver(x), '-', c=pl.get_color(), zorder=-10, label='FE solution')

    plt.plot(x, exact(x), '--', c=pl.get_color(), zorder=-10, label='exact solution')
    plt.legend(loc='best')

    plt.show()
