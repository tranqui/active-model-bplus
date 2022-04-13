#!/usr/bin/env python3

import sys
import numpy as np, matplotlib.pyplot as plt
import sympy as sp

from activedroplets import *

def droplet_model(zeta, lamb, K=1, t=-0.25, u=0.25):
    constant_parameters = (K, t, u)
    return ActiveModelBPlus(zeta, lamb, *constant_parameters)

def test_droplet(zeta, lamb, R, phi0, phi1, K=1, t=-0.25, u=0.25, guess_drop=None,
                 order=2, refinement_tol=1e-4, print_updates=None):
    model = droplet_model(zeta, lamb, K, t, u)
    try:
        drop = model.droplet(R, order=order, refinement_tol=refinement_tol, phi0=phi0, phi1=phi1, guess=guess_drop, print_updates=print_updates)
        print(drop.summary)
        return drop
    except Exception as e:
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
        raise e from None

def grad_mu(drop, phi=sp.Function('phi'), r=symbols.r):
    ode = drop.ode
    f = ActiveModelBSphericalInterface.strong_form.subs({p: v for p, v in zip(ode.parameters, ode.parameter_values)})
    f = f.subs(ActiveModelBSphericalInterface.unknown_function, phi)
    f = f.subs(ActiveModelBSphericalInterface.argument, r)
    return f

def local_mu(drop, phi=sp.Function('phi'), r=symbols.r):
    ode = drop.ode
    f = ActiveModelBSphericalInterface.local_term.subs({p: v for p, v in zip(ode.parameters, ode.parameter_values)})
    f = f.subs(ActiveModelBSphericalInterface.unknown_function, phi)
    f = f.subs(ActiveModelBSphericalInterface.argument, r)
    return f

def debug_plot(drop):
    R = drop.R

    plt.figure()
    plt.axvline(x=1, ls='dashed', lw=0.5)
    plt.axhline(y=0, ls='dashed', lw=0.5)
    xx = np.linspace(np.min(drop.x), np.max(drop.x), 1001)
    plt.plot(xx/R, drop(xx), '-', lw=0.5)

    phi = sp.Function('phi')
    r = symbols.r

    plt.figure()
    f = -phi(r).diff(r)
    plt.plot(xx/R, drop.evaluate(f, phi, r)(xx), lw=0.5)
    # f = phi(r).diff(r)**2
    # plt.plot(xx/R, drop.evaluate(f, phi, r)(xx), lw=0.5)
    f = phi(r).diff(r,2)
    plt.plot(xx/R, drop.evaluate(f, phi, r)(xx), lw=0.5)
    f = phi(r).diff(r,3)
    plt.plot(xx/R, drop.evaluate(f, phi, r)(xx), lw=0.5)

    plt.figure()
    plt.plot(xx/R, drop.evaluate(grad_mu(drop, phi, r), phi, r)(xx), lw=0.5)

def test_vary_phi(zeta=0, lamb=0, R=100, refinement_tol=1e-2): #refinement_tol=np.inf):
    phi0 = 1

    N = 11
    phi1 = -np.linspace(0.9, 1.1, N)
    mu0 = np.empty(N)
    mu1 = np.empty(N)
    dp = np.empty(N)
    S = np.empty(N)

    for i,p in enumerate(phi1):
        drop = test_droplet(zeta, lamb, R, phi0, p, refinement_tol=refinement_tol)
        print(drop.summary)
        dp[i] = drop.pseudopressure0 - drop.pseudopressure1
        mu0[i] = drop.mu0
        mu1[i] = drop.mu1
        S[i] = drop.pseudopressure_drop

    plt.figure()
    plt.plot(phi1, mu1 - mu0, lw=0.5)
    plt.axhline(y=0, ls='dashed')
    plt.ylabel('$\Delta \mu$')
    plt.xlabel('$\phi_1$')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(phi1, dp-S, lw=0.5)
    plt.axhline(y=0, ls='dashed')
    plt.ylabel('$\Delta p - S$')
    plt.xlabel('$\phi_1$')
    plt.legend(loc='best')

    plt.show()

def test_plot(drops):
    plt.figure()
    axphi = plt.gca()
    plt.title('$\phi$')

    plt.figure()
    axphilog = plt.gca()
    plt.title('$\phi$')

    plt.figure()
    axphipp = plt.gca()
    plt.title(r"$\phi''(r)$")

    plt.figure()
    axdmudr = plt.gca()
    plt.title('$d\mu/dr$ (strong form)')

    plt.figure()
    axmu = plt.gca()
    plt.title('local $\mu$ (weak form)')

    plt.figure()
    axdev = plt.gca()
    plt.title('deviation from reference $\phi^4$ kink')

    axes = [axphilog, axphipp, axdmudr, axmu, axdev]
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel(r'$|r - R| / \xi$')
    axphi.set_xlabel(r'$(r - R) / \xi$')
    axphi.set_xlim([-10, 10])

    phi, r = sp.Function('phi'), sp.Symbol('r')

    print('plotting...')
    for drop in drops:
        R = drop.R

        strong = grad_mu(drop)

        xi = drop.field_theory.passive_bulk_interfacial_width
        eps = 1e-3
        xx = R + np.concatenate((-np.geomspace(R-drop.x[0], eps, 1000), [0],
                                 np.geomspace(eps, drop.x[-1]-R, 1000)))
        # Rounding errors can put these numbers outside the domain, so make sure they are inside.
        xx[0] = drop.x[0]
        xx[-1] = drop.x[-1]
        #xplot = lambda x: x/R
        xplot = lambda x: np.abs(x-R)/xi

        pl, = axphi.plot((drop.x-R)/xi, drop(drop.x), '.')
        axphi.plot((xx-R)/xi, drop(xx), '-', lw=0.5, c=pl.get_color())

        pl, = axphilog.plot(xplot(drop.x), drop(drop.x), '.')
        axphilog.plot(xplot(xx), drop(xx), '-', lw=0.5, c=pl.get_color())

        dmudr = drop.evaluate(grad_mu(drop), phi, r, order=1, singularity_at_origin=True)
        axdmudr.plot(xplot(xx), dmudr(xx), '-', lw=0.5)

        print(drop.summary)
        print('total dmu/dr != 0 error:', np.linalg.norm(dmudr(drop.x)))

        axmu.plot(xplot(xx), drop.mu(xx), '-', lw=0.5)

        pl, = axphipp.plot(xplot(drop.x), drop(drop.x, derivative=2), '.')
        axphipp.plot(xplot(xx), drop(xx, derivative=2), lw=0.5, c=pl.get_color())

        reference = -np.tanh((xx-R)/xi)
        axdev.plot(xplot(xx), drop(xx) - reference, lw=0.5)

    plt.show()

def optimise(zeta=0, lamb=0, R=100, refinement_tol=1e-2):
    model = droplet_model(zeta, lamb)
    phi1_guess, phi0 = model.bulk_binodals

    drop0 = test_droplet(zeta, lamb, R, phi0, phi1_guess, refinement_tol=1e-2*refinement_tol)
    droplet = lambda p: test_droplet(zeta, lamb, R, phi0, p, refinement_tol=refinement_tol, guess_drop=drop0)

    drop_residual = lambda drop: drop.mu1 - drop.mu0
    #drop_residual = lambda drop: (drop.mu1 - drop.mu0)**2 #+ (drop.pseudopressure0 - drop.pseudopressure1 - drop.pseudopressure_drop)**2
    residual = lambda p: drop_residual(droplet(p))

    phi1 = newton_krylov(residual, phi1_guess)
    #from scipy.optimize import minimize
    #phi1 = minimize(residual, phi1_guess).x
    print('final phi1: %.8g' % phi1)

    return droplet(phi1)
    # test_plot([drop])
    
def test_errors(zeta=0, lamb=-1, R=100, order=2, print_updates=None):
    model = droplet_model(zeta, lamb)
    phi1, phi0 = model.bulk_binodals
    phi1 = -0.8716205

    np.set_printoptions(4, linewidth=10000)

    try: drop1 = test_droplet(zeta, lamb, R, phi0, phi1, refinement_tol=np.inf, order=order, print_updates=print_updates)
    except: return
    print(drop1.summary)

    drop2 = drop1.refine(refinement_tol=1e-4)
    print(drop2.summary)

    test_plot([drop1, drop2])

if __name__ == '__main__':

    #np.set_printoptions(4, suppress=True, linewidth=10000)
    np.set_printoptions(4, linewidth=10000)

    # phi1 = np.linspace(-0.9, -1.2, 21)
    # dP = np.empty(phi1.shape)
    # S = np.empty(phi1.shape)
    # for i, p in enumerate(phi1):
    #     print(p)
    #     drop = model.droplet(R, phi1=p)
    #     dP[i] = drop.pseudopressure0 - drop.pseudopressure1
    #     S[i] = drop.pseudopressure_drop
    #     print('%.4f %.4g %.4g' % (p, dP[i], S[i]))

    # plt.plot(phi1, dP, label=r'$\delta P$')
    # plt.plot(phi1, S, label=r'$S$')
    # plt.legend(loc='best')

    # plt.show()
    # import sys; sys.exit(0)

    # for zeta, lamb in [(0, 0), (-4, -1), (4, 1)]:
    #     model = ActiveModelBPlus(zeta, lamb, *constant_parameters)
    #     drop = model.droplet(R)

    #     P0, P1 = drop.pseudopressure0, drop.pseudopressure1
    #     mu0, mu1 = drop.mu0, drop.mu1
    #     S = drop.pseudopressure_drop
    #     print('z=%g l=%g: phi=[%.4f->%.4f] mu=[%.4f->%.4f] P=[%.4f->%.4f] S=%.4f dP=%.4f' % (zeta, lamb, drop.phi0, drop.phi1, mu0, mu1, P0, P1, S, P0-P1))

    #     xx = np.linspace(np.min(drop.x), np.max(drop.x), 1001)

    #     label = r'$\zeta=%g, \; \lambda=%g$' % (zeta, lamb)
    #     plt.plot(xx/R, drop(xx), '-', lw=0.5, label=label)

    # plt.legend(loc='best')
    # plt.ylim([-2, 2])
    # plt.xlabel('$r/R$')
    # plt.ylabel('$\phi$')

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

    # plt.show()

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
    #     dpsi = Phi4Pseudopotential(z, l, K, t, u)(phi, derivative=1)
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
