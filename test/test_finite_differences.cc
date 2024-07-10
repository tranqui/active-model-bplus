#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "test_utilities.h"

using namespace finite_difference;

Field quartic_field(int Nx, int Ny, Scalar dx=1, Scalar dy=1)
{
    Scalar Lx = Nx*dx;
    Scalar Ly = Ny*dy;

    Field field(Ny, Nx);
    for (int i = 0; i < Ny; ++i)
    {
        Scalar y = i * dy;
        for (int j = 0; j < Nx; ++j)
        {
            Scalar x = j * dx;
            field(i, j) = x * (Lx - x) * y * (Ly - y);
        }
    }

    return field;
}

// Check $\nabla \cdot (\nabla \phi) = \nabla^2 \phi$ to test that
// stagger grid set-up for gradients is working correctly.
TEST_CASE("StaggerTest")
{
    if constexpr (order != 2) return;

    int Nx{64}, Ny{32};
    Stencil stencil{1e-2, 1, 0.75};

    // Field field = quartic_field(Nx, Ny); // more controlled field, but still fails for order>2
    Field field = Field::Random(Ny, Nx); // more stringent test

    Gradient grad = gradient<Right>(field, stencil);
    Field expected_lap = laplacian(field, stencil);
    Field actual_lap = divergence<Left>(grad, stencil);
    Field relative_error = (actual_lap.array() / expected_lap.array()) - 1;
    CHECK(is_equal<loose_tol>(relative_error, 0));
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
