#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "integrator.cuh"
#include "foreach.cuh"


/// Type-traits to facilitate how tests are performed for different data.

template <typename T>
struct is_eigen_matrix : std::false_type {};
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct is_eigen_matrix <Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : std::true_type {};
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct is_eigen_matrix <const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : std::true_type {};

// Structured data can be cast to an std::tuple if it has the member function as_tuple().
template <typename T, typename = void>
struct has_as_tuple : std::false_type {};
template <typename T>
struct has_as_tuple<T, std::void_t<decltype(std::declval<T>().as_tuple())>>
    : std::true_type {};


/// Helper functions for equality assertions with vectorial data.

// Eigen arrays we have to check matrix size and then element-wise data.
template <typename T1, typename T2,
          typename = std::enable_if_t<is_eigen_matrix<std::decay_t<T1>>::value and
                                      is_eigen_matrix<std::decay_t<T2>>::value>>
void assert_equal(T1&& a, T2&& b)
{
    REQUIRE(a.rows() == b.rows());
    REQUIRE(a.cols() == b.cols());
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            REQUIRE(a(i,j) == b(i,j));
}

// Element-wise test in tuples.
template <typename... T>
void assert_equal(const std::tuple<T...>& a, const std::tuple<T...>& b)
{
    auto test = [&](auto m)
    {
        REQUIRE(std::get<m>(a) == std::get<m>(b));
    };
    for_each(std::index_sequence_for<T...>{}, test);
}

// Overload for structured data: cast to tuple then test.
template <typename Params,
          typename = std::enable_if_t<has_as_tuple<Params>::value>>
void assert_equal(const Params& a, const Params& b)
{
    assert_equal(a.as_tuple(), b.as_tuple());
}


/// The unit tests.


TEST_CASE("Constructor")
{
    int Nx{64}, Ny{64};
    Field initial = Field::Random(Ny, Nx);
    Stencil stencil{};
    Model model{};

    Integrator simulation(initial, stencil, model);

    assert_equal(initial, simulation.get_field());
    assert_equal(stencil, simulation.get_stencil());
    assert_equal(model, simulation.get_model());
}

TEST_CASE("MoveConstructor")
{
    int Nx{64}, Ny{64};
    Field initial = Field::Random(Ny, Nx);

    Integrator simulation(initial, Stencil{}, Model{});
    auto expected = simulation.get_field();
    Integrator move(std::move(simulation));
    auto actual = move.get_field();

    assert_equal(expected, actual);
    assert_equal(simulation.get_stencil(), simulation.get_stencil());
    assert_equal(simulation.get_model(), move.get_model());
}

inline Scalar bulk_chemical_potential(Scalar field, const Model& model)
{
    return model.a * field
         + model.b * field * field
         + model.c * field * field * field;
}

TEST_CASE("BulkCurrent")
{
    int Nx{64}, Ny{64};
    Field initial = Field::Random(Ny, Nx);
    Stencil stencil{1e-2, 1, 1};
    Model model{1, 2, 3, 0, 0, 0};

    Integrator simulation(initial, stencil, model);

    // Bulk chemical potential is evaluated point-wise
    Field mu = Field(Ny, Nx);
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            mu(i, j) = bulk_chemical_potential(initial(i, j), model);

    // Find current $\vec{J} = - \nabla \mu$ by second-order finite difference:
    Current expected{Field(Ny, Nx), Field(Ny, Nx)};
    for (int i = 1; i < Ny-1; ++i)
        for (int j = 1; j < Nx-1; ++j)
        {
            expected[0](i, j) = 0.5 * (mu(i+1, j  ) - mu(i-1, j  )) / stencil.dy;
            expected[1](i, j) = 0.5 * (mu(i  , j+1) - mu(i  , j-1)) / stencil.dx;
        }

    Current actual = simulation.get_current();
    assert_equal(expected[0], actual[0]);
    assert_equal(expected[1], actual[1]);
}
