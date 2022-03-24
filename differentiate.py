#!/usr/bin/env python3
import numpy

def gradient(f, x, dx=1e-8, args=[]):
    """Numerically take the gradient of a scalar function at point x.

    This uses a central finite difference scheme to numerically find the derivatives.
    The objective function must return a scalar, but the inputs can be either a scalar
    or a vector.

    Args:
        f: objective function to differentiate
        x: scalar or vector argument to evaluate the derivatives
        dx: precision of derivatives for the finite difference method
        args: additional arguments to f
    Returns:
        g: gradient of the function at point x, with the same dimensionality as input x
    """
    f0 = f(x, *args)
    try: g = numpy.empty((x.size,len(f0)), dtype=numpy.longdouble)
    except: g = numpy.empty(x.size, dtype=numpy.longdouble)
    shape = x.shape

    x = x.reshape(-1)
    for i in range(x.size):
        x0 = x[i]
        x[i] += dx
        g[i] = (f(x.reshape(shape), *args)-f0) / (x[i]-x0) # using exact step after rounding
        x[i] = x0

    try: return g.reshape(shape)
    except: return g

def hessian(f, x, dx=1e-6):
    """Numerically take the second derivative (the hessian matrix) of a scalar function at
    point x.

    This uses a central finite difference scheme to numerically find the derivatives.
    The objective function must return a scalar, but the inputs can be either a scalar
    or a vector.

    Args:
        f: objective function to differentiate
        x: scalar or vector argument to evaluate the derivatives at
        dx: precision of derivatives for the finite difference method
    Returns:
        h: hessian of the function at point x, as a square matrix with width/length the same as
           the dimensionality of x
    """
    h = numpy.empty((x.size,x.size), dtype=numpy.longdouble)
    g0 = gradient(f,x,dx)
    shape = x.shape

    x = x.reshape(-1)
    for i in range(x.size):
        x0 = x[i]
        x[i] += dx
        h[i] = (gradient(f,x.reshape(shape),dx) - g0).reshape(-1) / (x[i]-x0) # using exact step after rounding
        x[i] = x0

    return h

def hessian_diagonal_entries(f, x, dx=1e-6):
    """Numerically take the second derivatives a scalar function at point x with respect to only
    the diagonal entries i.e. d^2f/dx_i^2.

    This uses a central finite difference scheme to numerically find the derivatives.
    The objective function must return a scalar, but the inputs can be either a scalar
    or a vector.

    Args:
        f: objective function to differentiate
        x: scalar or vector argument to evaluate the derivatives at
        dx: precision of derivatives for the finite difference method
    Returns:
        h: diagonal entries of the hessian at point x, as a vector with the same shape as x
    """
    shape = x.shape
    x = x.reshape(-1)

    h = numpy.empty(x.size, dtype=numpy.longdouble)
    f0 = f(x)

    for i in range(x.size):
        x0 = x[i]

        # We take the derivatives in this way to find the exact step taken after rounding errors
        x[i] += dx
        xp,fp = x[i], f(x.reshape(shape))
        x[i] = x0 - dx
        xm,fm = x[i], f(x.reshape(shape))
        x[i] = x0

        h[i] = (fp - 2*f0 + fm) / (0.5*(xp-xm))**2

    return h.reshape(shape)

def hessian_normal_modes(hessian, rigid_modes):
    """Find the normal modes of the hessian matrix after projecting out the rigid
    body modes.

    Args:
        hessian: hessian matrix containing second derivatives
        rigid_modes: vectors of rigid body motion (i.e. normal modes with zero frequencies)
    Returns:
        evals: eigenvalues of hessian orthogonal to rigid body modes (i.e. the squared normal mode frequencies)
        evecs: eigenvectors of each corresponding normal mode
    """

    num_normal_modes = len(hessian) - len(rigid_modes)

    # Orthonormalise a projection matrix by Gramm-Schmidt procedure
    projection = numpy.random.random((len(hessian),num_normal_modes))
    for col in range(num_normal_modes):
        for rigid_mode in rigid_modes:
            projection[:,col] -= projection[:,col].dot(rigid_mode) * rigid_mode
        for prev_col in range(col):
            projection[:,col] -= projection[:,col].dot(projection[:,prev_col]) * projection[:,prev_col]
        projection[:,col] /= numpy.linalg.norm(projection[:,col])

    # Compute the normal modes in the basis orthogonal to the rigid modes
    projected_hessian = projection.T.dot(hessian).dot(projection)

    evals,evecs = numpy.linalg.eigh(projected_hessian)
    # Revert back to the original basis.
    evecs = projection.dot(evecs)

    return evals,evecs
