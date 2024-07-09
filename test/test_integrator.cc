#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "integrator.h"
#include "for_each.h"
#include "test_utilities.h"

using namespace finite_difference;

TEST_CASE("Constructor")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);
    Stencil stencil{};
    Model model{};

    Integrator simulation(initial, stencil, model);

    CHECK(is_equal<tight_tol>(initial, simulation.get_field()));
    CHECK(is_equal(stencil, simulation.get_stencil()));
    CHECK(is_equal(model, simulation.get_model()));
}

TEST_CASE("MoveConstructor")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);

    {
        auto simulation = std::make_unique<Integrator>(initial, Stencil{}, Model{});
        auto expected = simulation->get_field();
        auto stencil = simulation->get_stencil();
        auto model = simulation->get_model();

        std::unique_ptr<Integrator> move = std::move(simulation);
        auto actual = move->get_field();

        CHECK(is_equal<tight_tol>(expected, actual));
        CHECK(is_equal(stencil, move->get_stencil()));
        CHECK(is_equal(model, move->get_model()));
    }

    // Check destructor did not call twice (which would raise a CUDA error upon trying
    // to release the same resource).
    kernel::throw_errors();
}

TEST_CASE("BulkCurrentTest")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);
    Scalar dt{1e-2};
    Stencil stencil{dt, 1, 0.75};
    Model model{1, 2, 3, 0, 0, 0};

    Integrator simulation(initial, stencil, model);
    Field field = simulation.get_field();

    // Bulk chemical potential is evaluated point-wise
    Field mu(Ny, Nx);
    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
            mu(i, j) = bulk_chemical_potential(field(i, j), model);

    // Current $\vec{J} = - \nabla \mu$:
    Current expected_J = gradient<Right>(mu, stencil);
    for (int c = 0; c < d; ++c) expected_J[c] *= -1;

    Current actual_J = simulation.get_current();
    CHECK(is_equal<tight_tol>(expected_J[0], actual_J[0]));
    CHECK(is_equal<tight_tol>(expected_J[1], actual_J[1]));

    simulation.run(1);
    Field actual_divJ = -(simulation.get_field() - field) / dt;

    {
        // Expect: $\nabla \cdot \vec{J} = -\nabla^2 \mu$.
        Field expected_divJ = -laplacian(mu, stencil);
        CHECK(is_equal<tight_tol>(expected_divJ, actual_divJ));
    }

    {
        // Alternative: $-\nabla \cdot \vec{J}$ directly.
        Field expected_divJ = divergence<Left>(actual_J, stencil);
        CHECK(is_equal<tight_tol>(expected_divJ, actual_divJ));
    }
}

TEST_CASE("SurfaceKappaCurrentTest")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);
    Scalar dt{1e-2};
    Stencil stencil{dt, 1, 0.75};
    Model model{0, 0, 0, 1, 0, 0};

    Integrator simulation(initial, stencil, model);
    Field field = simulation.get_field();

    // Current $\vec{J} = -\nabla \mu$ with $\mu = - \kappa \nabla^2 \phi$:
    Field mu = -model.kappa * laplacian(field, stencil);
    Current expected_J = gradient<Right>(mu, stencil);
    for (int c = 0; c < d; ++c) expected_J[c] *= -1;

    Current actual_J = simulation.get_current();
    CHECK(is_equal<tight_tol>(expected_J[0], actual_J[0]));
    CHECK(is_equal<tight_tol>(expected_J[1], actual_J[1]));

    simulation.run(1);
    Field actual_divJ = -(simulation.get_field() - field) / dt;

    {
        // Expect: $\nabla \cdot \vec{J} = -\nabla^2 \mu$.
        Field expected_divJ = -laplacian(mu, stencil);
        CHECK(is_equal<tight_tol>(actual_divJ, expected_divJ));
    }

    {
        // Alternative: $-\nabla \cdot \vec{J}$ directly.
        Field expected_divJ = divergence<Left>(actual_J, stencil);
        CHECK(is_equal<tight_tol>(actual_divJ, expected_divJ));
    }
}

TEST_CASE("SurfaceLambdaCurrentTest")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);
    Scalar dt{1e-2};
    Stencil stencil{dt, 1, 0.75};
    Model model{0, 0, 0, 0, 1, 0};

    Integrator simulation(initial, stencil, model);
    Field field = simulation.get_field();

    // Current $\vec{J} = -\nabla \mu$ with $\mu = \lambda |\nabla\phi|^2$:

    Gradient grad = gradient(field, stencil);
    Field mu = Field::Zero(Ny, Nx);
    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
            for (int c = 0; c < d; ++c)
                mu(i, j) += model.lambda * grad[c](i,j) * grad[c](i,j);

    Current expected_J = gradient<Right>(mu, stencil);
    for (int c = 0; c < d; ++c) expected_J[c] *= -1;

    Current actual_J = simulation.get_current();
    CHECK(is_equal<tight_tol>(expected_J[0], actual_J[0]));
    CHECK(is_equal<tight_tol>(expected_J[1], actual_J[1]));

    simulation.run(1);
    Field actual_divJ = -(simulation.get_field() - field) / dt;

    {
        // Expect: $\nabla \cdot \vec{J} = -\nabla^2 \mu$.
        Field expected_divJ = -laplacian(mu, stencil);
        CHECK(is_equal<tight_tol>(actual_divJ, expected_divJ));
    }

    {
        // Alternative: $-\nabla \cdot \vec{J}$ directly.
        Field expected_divJ = divergence<Left>(actual_J, stencil);
        CHECK(is_equal<tight_tol>(actual_divJ, expected_divJ));
    }
}

TEST_CASE("SurfaceZetaCurrentTest")
{
    // int Nx{64}, Ny{32};
    int Nx{8}, Ny{8};
    Field initial = Field::Random(Ny, Nx);
    Scalar dt{1e-2};
    Stencil stencil{dt, 1, 1.25};
    Model model{0, 0, 0, 0, 0, 1};

    Integrator simulation(initial, stencil, model);
    Field field = simulation.get_field();

    // Current $\vec{J} = (\nabla^2 \phi) \nabla \phi$:

    Field lap_y = laplacian<Right, Central>(field, stencil);
    Field lap_x = laplacian<Central, Right>(field, stencil);
    Gradient grad = gradient<Right>(field, stencil);

    Current expected_J{Field(Ny, Nx), Field(Ny, Nx)};
    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
        {
            expected_J[0](i,j) = model.zeta * lap_y(i,j) * grad[0](i,j);
            expected_J[1](i,j) = model.zeta * lap_x(i,j) * grad[1](i,j);
        }

    Current actual_J = simulation.get_current();
    CHECK(is_equal<tight_tol>(expected_J[0], actual_J[0]));
    CHECK(is_equal<tight_tol>(expected_J[1], actual_J[1]));

    simulation.run(1);
    Field actual_divJ = -(simulation.get_field() - field) / dt;
    Field expected_divJ = divergence<Left>(actual_J, stencil);
    CHECK(is_equal<tight_tol>(actual_divJ, expected_divJ));
}

TEST_CASE("ConservationTest")
{
    int Nx{512}, Ny{256};
    Field initial = 0.1 * Field::Random(Ny, Nx);
    initial -= Field::Constant(Ny, Nx, initial.mean());

    constexpr int nsteps = 1000;
    Stencil stencil{1e-2, 1, 1.25};

    Field previous;
    bool first{true};

    for (auto& model : {Model{-0.25, 0  , 0.25, 1, 0, 0},
                        Model{-0.25, 0.5, 0.25, 1, 0, 0},
                        Model{-0.25, 0.5, 0.25, 1, 1, 0},
                        Model{-0.25, 0.5, 0.25, 1, 1, 1}})
    {
        Integrator simulation(initial, stencil, model);
        Scalar expected_mass = simulation.get_field().sum();

        simulation.run(nsteps);
        Field field = simulation.get_field();

        Scalar actual_mass = field.sum();
        CHECK(is_equal<loose_tol>(expected_mass, actual_mass));

        // Different parameter sets should have different trajectories.
        if (not first) CHECK(not is_equal<loose_tol>(field, previous));
        first = false;
        previous = Field{std::move(field)};
    }
}

TEST_CASE("PhaseSeparationTest")
{
    int Nx{512}, Ny{256};
    Field initial = 0.1 * Field::Random(Ny, Nx);
    initial -= Field::Constant(Ny, Nx, initial.mean());

    Stencil stencil{1e-2, 1, 1.25};
    // Parameters for phase separation with binodal at \phi = \pm 1.
    Model model{-0.25, 0, 0.25, 1, 0, 0};

    Integrator simulation(initial, stencil, model);

    constexpr int nsteps = 100000;
    simulation.run(nsteps);
    Field field = simulation.get_field();

    // Check system is converging towards the binodal.
    Scalar max{field.maxCoeff()}, min{field.minCoeff()};
    constexpr Scalar tol = 0.1; // need very loose tolerance because system will not have converged
    CHECK(is_equal<tol>(max, 1));
    CHECK(is_equal<tol>(min, -1));
}

TEST_CASE("TimestepTest")
{
    int Nx{64}, Ny{64};
    Field initial = Field::Random(Ny, Nx);

    constexpr Scalar dt = 0.1;
    Integrator simulation(initial, Stencil{dt,1,1}, Model{});

    constexpr int nsteps = 10;
    CHECK(simulation.get_timestep() == 0);
    simulation.run(nsteps);
    CHECK(simulation.get_timestep() == nsteps);
    CHECK(is_equal<tight_tol>(simulation.get_time(), nsteps * dt));
}
