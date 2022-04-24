#!/usr/bin/env python3

import sys
import numpy as np, matplotlib.pyplot as plt
import sympy as sp

from activedroplets import symbols, ActiveModelBSphericalInterface, ActiveModelBPlus

def field_theory(zeta, lamb, K=1, t=-0.25, u=0.25, d=3):
    constant_parameters = (K, t, u)
    return ActiveModelBPlus(zeta, lamb, *constant_parameters, d=d)

def droplet(zeta, lamb, R, K=1, t=-0.25, u=0.25, d=3, **kwargs):
    model = field_theory(zeta, lamb, K, t, u, d)
    try:
        drop = model.droplet(R, **kwargs)
        print(drop.summary)
        return drop
    except Exception as e:
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
        raise e from None

def grad_mu(drop, phi=sp.Function('\phi'), r=symbols.r):
    ode = drop.ode
    f = ActiveModelBSphericalInterface.strong_form.subs({p: v for p, v in zip(ode.parameters, ode.parameter_values)})
    f = f.subs(ActiveModelBSphericalInterface.unknown_function, phi)
    f = f.subs(ActiveModelBSphericalInterface.argument, r)
    return f

def plot_droplet_profile(drops):
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

    phi, r = sp.Function('\phi'), sp.Symbol('r')

    print('plotting...')
    for drop in drops:
        R = drop.R

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

def test_varying_radius(R, zeta=0, lamb=0, **kwargs):
    droplets = []
    last_droplet = None

    R = np.sort(R)
    #for r in np.flipud(R):
    for r in R:
        #last_droplet = droplet(zeta, lamb, r, guess=last_droplet, **kwarg)s
        try: last_droplet = droplet(zeta, lamb, r, **kwargs)
        except: last_droplet = droplet(zeta, lamb, r, guess=last_droplet, **kwargs)
        droplets += [last_droplet]
    #droplets.reverse()

    return droplets

def show_size_dependence(*args):
    drops = []
    for a in args: drops += a

    unique_states = {}
    for drop in drops:
        state = tuple(drop.field_theory.parameters.values())
        if state not in unique_states: unique_states[state] = []
        unique_states[state] += [drop]

    # Some bookkeeping so we can only show changing parameters in legend labels.
    labels = tuple(drop.field_theory.parameters.keys())
    states = np.array(list(unique_states.keys()))
    changing_columns = np.where(~np.all(states == states[0,:], axis = 0))[0]

    plt.figure()
    ax0 = plt.gca()
    plt.figure()
    ax1 = plt.gca()
    plt.figure()
    axp = plt.gca()

    axes = [ax0, ax1, axp]

    for state, drops in unique_states.items():
        bulk_phi1, bulk_phi0 = drops[0].field_theory.bulk_binodals

        R = np.empty(len(drops))
        dP = np.empty(len(drops))
        S = np.empty(len(drops))
        phi0 = np.empty(len(drops))
        phi1 = np.empty(len(drops))

        # R = np.arange(len(drops))
        # dP = np.arange(len(drops))
        # S = np.arange(len(drops))
        # phi0 = np.arange(len(drops))
        # phi1 = np.arange(len(drops))

        for i,drop in enumerate(drops):
            P0, P1 = drop.pseudopressure0, drop.pseudopressure1
            R[i] = drop.R
            dP[i] = P0-P1
            S[i] = drop.pseudopressure_drop
            phi0[i] = drop.phi0
            phi1[i] = drop.phi1
            print(drop)

        order = np.argsort(R)
        dP = dP[order]
        S = S[order]
        phi0 = phi0[order]
        phi1 = phi1[order]
        R = R[order]

        label = ''
        for c in changing_columns:
            label += str(labels[c]) + '= %.2g' % state[c]
            if c < len(changing_columns)-1: label += '; '
        label = '$%s$' % label

        pl, = ax0.plot(R, phi0, '.-', lw=0.5, label=label)
        ax0.axhline(y=bulk_phi0, ls='dashed', lw=0.5, c=pl.get_color())

        pl, = ax1.plot(R, phi1, '.-', lw=0.5, label=label)
        ax1.axhline(y=bulk_phi1, ls='dashed', lw=0.5, c=pl.get_color())

        pl, = axp.plot(R, dP, '-', lw=0.5, label=label)
        axp.plot(R, S, '--', lw=0.5, c=pl.get_color())
        #zeta, lamb = drop.field_theory.zeta, drop.field_theory.lamb
        #axp.plot(R, (zeta - lamb) * S, 'o:', lw=0.5, c=pl.get_color())
        #axp.plot(R, -S, 'o:', lw=0.5, c=pl.get_color())

    ax0.set_xlim([0, 40])
    ax0.set_ylim([1, 1.3])
    ax0.set_ylabel('$\phi_0$')

    ax1.set_xlim([0, 40])
    ax1.set_ylim([-1, -0.7])
    ax1.set_ylabel('$\phi_1$')

    for ax in axes:
        ax.legend(loc='best')
        ax.set_xlabel('$R$')

    # plt.figure()
    # plt.plot(R, dP+S, '.-', lw=0.5, label='$\Delta P + S$')
    # plt.legend(loc='best')
    # plt.xlabel('$R$')

    # plt.figure()
    # plt.plot(R, dP-S, '.-', lw=0.5, label='$\Delta P - S$')
    # plt.legend(loc='best')
    # plt.xlabel('$R$')

    plt.show()

def test_against_literature(R=np.arange(10, 101, 1)):
    import cloudpickle
    import os

    for d in [2, 3]:
        for zeta, lamb in [[-1, 0.5], [-4, -1]]:
            cache_path = 'zeta=%g_lamb=%g_d=%d.drops' % (zeta, lamb, d)
            if os.path.exists(cache_path): continue

            try:
                drops = test_varying_radius(R, zeta, lamb, d=d)

                with open(cache_path, 'wb') as f:
                    cloudpickle.dump(drops, f)
            except Exception as e:
                print(e)

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
