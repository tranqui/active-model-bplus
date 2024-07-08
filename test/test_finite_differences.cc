#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "test_utilities.h"

using namespace finite_difference;

// Check $\nabla \cdot (\nabla \phi) = \nabla^2 \phi$ to test that
// stagger grid set-up for gradients is working correctly.
TEST_CASE("StaggerTest")
{
    int Nx{64}, Ny{32};
    Stencil stencil{1e-2, 1, 0.75};

    Field field = Field::Random(Ny, Nx);
    Gradient grad = staggered_gradient<Right>(field, stencil);
    Field lap = laplacian(field, stencil);
    CHECK(is_equal<tight_tol>(lap, staggered_divergence<Left>(grad, stencil)));
}

/// Check derivatives of known analytic functions

// $\phi = x$:
TEST_CASE("LinearXTest")
{
    int Nx{64}, Ny{32};
    Stencil stencil{1e-2, 1, 1};

    Field field(Ny, Nx);
    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
            field(i, j) = j;

    Gradient grad = gradient(field, stencil);
    Field lap = laplacian(field, stencil);

    CHECK(is_equal<tight_tol>(lap(Ny/2, Nx/2), 0));
    CHECK(is_equal<tight_tol>(grad[0](Ny/2, Nx/2), 0));
    CHECK(is_equal<tight_tol>(grad[1](Ny/2, Nx/2), 1));
}

// $\phi = y$:
TEST_CASE("LinearYTest")
{
    int Nx{64}, Ny{32};
    Stencil stencil{1e-2, 1, 1};

    Field field(Ny, Nx);
    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
            field(i, j) = i;

    Gradient grad = gradient(field, stencil);
    Field lap = laplacian(field, stencil);

    CHECK(is_equal<tight_tol>(lap(Ny/2, Nx/2), 0));
    CHECK(is_equal<tight_tol>(grad[0](Ny/2, Nx/2), 1));
    CHECK(is_equal<tight_tol>(grad[1](Ny/2, Nx/2), 0));
}

// $\phi = (x^2) / 2$:
TEST_CASE("QuadraticXTest")
{
    int Nx{64}, Ny{32};
    Stencil stencil{1e-2, 1, 1};

    Field field(Ny, Nx);
    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
            field(i, j) = 0.5 * j*j;

    Gradient grad = gradient(field, stencil);
    Field lap = laplacian(field, stencil);

    CHECK(is_equal<tight_tol>(lap(Ny/2, Nx/2), 1));
    CHECK(is_equal<tight_tol>(grad[0](Ny/2, Nx/2), 0));
    CHECK(is_equal<tight_tol>(grad[1](Ny/2, Nx/2), Nx/2));
}

// $\phi = (y^2) / 2$:
TEST_CASE("QuadraticYTest")
{
    int Nx{64}, Ny{32};
    Stencil stencil{1e-2, 1, 1};

    Field field(Ny, Nx);
    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
            field(i, j) = 0.5 * i*i;

    Gradient grad = gradient(field, stencil);
    Field lap = laplacian(field, stencil);

    CHECK(is_equal<tight_tol>(lap(Ny/2, Nx/2), 1));
    CHECK(is_equal<tight_tol>(grad[0](Ny/2, Nx/2), Ny/2));
    CHECK(is_equal<tight_tol>(grad[1](Ny/2, Nx/2), 0));
}
