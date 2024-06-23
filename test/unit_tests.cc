#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "integrator.cuh"

void equality_test(const FieldRef& a, const FieldRef& b)
{
    REQUIRE(a.rows() == b.rows());
    REQUIRE(a.cols() == b.cols());
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            REQUIRE(a(i,j) == b(i,j));
}

TEST_CASE("Constructor")
{
    Scalar dt{1e-2}, dx{1}, dy{1};
    int Nx{64}, Ny{64};
    Field expected = Field::Random(Nx, Ny);

    Integrator simulation(expected, dt, dx, dy);
    auto actual = simulation.get_field();

    equality_test(expected, actual);
}

TEST_CASE("MoveConstructor")
{
    Scalar dt{1e-2}, dx{1}, dy{1};
    int Nx{64}, Ny{64};
    Field initial = Field::Random(Nx, Ny);

    Integrator simulation(initial, dt, dx, dy);
    auto expected = simulation.get_field();
    Integrator move(std::move(simulation));
    auto actual = move.get_field();

    equality_test(expected, actual);
}
