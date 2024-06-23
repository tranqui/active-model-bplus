#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "integrator.cuh"

TEST_CASE("Constructor")
{
    Scalar dt{1e-2}, dx{1}, dy{1};
    int Nx{1024}, Ny{1024};
    auto expected = Field::Random(Nx, Ny);

    Integrator simulation(expected, dt, dx, dy);
    auto actual = simulation.get_field();

    REQUIRE(actual.rows() == expected.rows());
    REQUIRE(actual.cols() == expected.cols());
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Nx; ++j)
            REQUIRE(actual(i,j) == expected(i,j));
}

TEST_CASE("MoveConstructor")
{
    Scalar dt{1e-2}, dx{1}, dy{1};
    int Nx{1024}, Ny{1024};
    auto initial = Field::Random(Nx, Ny);

    Integrator simulation(initial, dt, dx, dy);
    auto expected = simulation.get_field();
    Integrator move(std::move(simulation));
    auto actual = move.get_field();

    REQUIRE(actual.rows() == expected.rows());
    REQUIRE(actual.cols() == expected.cols());
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Nx; ++j)
            REQUIRE(actual(i,j) == expected(i,j));
}
