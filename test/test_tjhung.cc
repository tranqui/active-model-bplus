#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "integrator.h"
#include "tjhung_ambplus.h"
#include "test_utilities.h"

/// Test against integrator implemented by Tjhung et al. (2018).
TEST_CASE("IntegrationTest")
{
    int Nx{64}, Ny{32};
    Scalar dt{1e-2}, dx{1}, dy{1};
    Scalar A{0.25}, kappa{1}, lamb{1}, zeta{1}, T{0};

    Model model{-A, 0, A, kappa, lamb, zeta, T};
    Stencil stencil{dt, dx, dy};

    Field initial = Field::Random(Nx, Ny);

    TjhungIntegrator tjhung(Nx, Ny, dt, dx, dy, A, kappa, lamb, zeta, T);
    tjhung.phi = Field(initial);
    tjhung.calculate_dphi();

    Integrator simulation(initial, stencil, model);
    Field mueq = simulation.get_passive_chemical_potential();
    Field muact = simulation.get_active_chemical_potential();
    CHECK(is_equal<tight_tol>(mueq, tjhung.mueq));
    CHECK(is_equal<tight_tol>(muact, tjhung.muact));

    Current Jeq = simulation.get_passive_current();
    CHECK(is_equal<tight_tol>(Jeq[0], tjhung.Jxeq));
    CHECK(is_equal<tight_tol>(Jeq[1], tjhung.Jyeq));

    Current Jact = simulation.get_active_current();
    CHECK(is_equal<tight_tol>(Jact[0], tjhung.Jxact));
    CHECK(is_equal<tight_tol>(Jact[1], tjhung.Jyact));

    simulation.run(1);
    Field expected_divJ = -tjhung.dphi;
    Field actual_divJ = -(simulation.get_field() - initial) / dt;
    CHECK(is_equal<tight_tol>(expected_divJ, actual_divJ));
}
