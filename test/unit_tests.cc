#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "integrator.cuh"
#include "foreach.cuh"


// Numerical tolerance for equality with numerical calculations.
static constexpr Scalar tight_tol = 1e-12;
static constexpr Scalar loose_tol = 1e-6;


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

template <Scalar tolerance>
void assert_equal(Scalar a, Scalar b)
{
    CHECK(std::abs(a - b) < tolerance);
}

void assert_equal(Scalar a, Scalar b)
{
    CHECK(a == b);
}

// Eigen arrays we have to check matrix size and then element-wise data.
template <Scalar tolerance, typename T1, typename T2,
          typename = std::enable_if_t<is_eigen_matrix<std::decay_t<T1>>::value and
                                      is_eigen_matrix<std::decay_t<T2>>::value>>
void assert_equal(T1&& a, T2&& b)
{
    REQUIRE(a.rows() == b.rows());
    REQUIRE(a.cols() == b.cols());
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            if constexpr (tolerance == 0) CHECK (a(i,j) == b(i,j));
            else CHECK(std::abs(a(i,j) - b(i,j)) <= tolerance);
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_eigen_matrix<std::decay_t<T1>>::value and
                                      is_eigen_matrix<std::decay_t<T2>>::value>>
void assert_equal(T1&& a, T2&& b)
{
    return assert_equal<Scalar{0}>(std::forward<T1>(a), std::forward<T2>(b));
}

// Element-wise test in tuples.
template <typename... T>
void assert_equal(const std::tuple<T...>& a, const std::tuple<T...>& b)
{
    auto test = [&](auto m)
    {
        CHECK(std::get<m>(a) == std::get<m>(b));
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


/// Expected physical values.


inline Scalar bulk_chemical_potential(Scalar field, const Model& model)
{
    return model.a * field
         + model.b * field * field
         + model.c * field * field * field;
}

// Find gradient of field by second-order finite difference:
inline Gradient gradient(Field field, Stencil stencil)
{
    const auto Ny{field.rows()}, Nx{field.cols()};
    Gradient grad{Field(Ny, Nx), Field(Ny, Nx)};

    for (int i = 0; i < Ny; ++i)
    {
        // Nearest neighbours in y-direction w/ periodic boundaries:
        int ip{i+1}, im{i-1};
        if (im < 0) im += Ny;
        if (ip >= Ny) ip -= Ny;

        for (int j = 0; j < Nx; ++j)
        {
            // Nearest neighbours in x-direction w/ periodic boundaries:
            int jp{j+1}, jm{j-1};
            if (jm < 0) jm += Nx;
            if (jp >= Nx) jp -= Nx;

            grad[0](i, j) = 0.5 * (field(ip, j ) - field(im, j )) / stencil.dy;
            grad[1](i, j) = 0.5 * (field(i , jp) - field(i , jm)) / stencil.dx;
        }
    }

    return grad;
}

// Find laplacian of field by second-order finite difference:
inline Field laplacian(const FieldRef& field, Stencil stencil)
{
    const Scalar dxInv{1/stencil.dx}, dyInv{1/stencil.dy};
    const auto Ny{field.rows()}, Nx{field.cols()};
    Field lap{Ny, Nx};

    for (int i = 0; i < Ny; ++i)
    {
        // Nearest neighbours in y-direction w/ periodic boundaries:
        int ip{i+1}, im{i-1};
        if (im < 0) im += Ny;
        if (ip >= Ny) ip -= Ny;

        for (int j = 0; j < Nx; ++j)
        {
            // Nearest neighbours in x-direction w/ periodic boundaries:
            int jp{j+1}, jm{j-1};
            if (jm < 0) jm += Nx;
            if (jp >= Nx) jp -= Nx;

            lap(i, j) =   dyInv*dyInv * (field(ip,j) + field(im,j))
                        + dxInv*dxInv * (field(i,jp) + field(i,jm))
                        - 2*(dxInv*dxInv + dyInv*dyInv) * field(i,j);
        }
    }

    return lap;
}


/// The unit tests.


TEST_CASE("Constructor")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);
    Stencil stencil{};
    Model model{};

    Integrator simulation(initial, stencil, model);

    assert_equal<tight_tol>(initial, simulation.get_field());
    assert_equal(stencil, simulation.get_stencil());
    assert_equal(model, simulation.get_model());
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

        assert_equal<tight_tol>(expected, actual);
        assert_equal(stencil, move->get_stencil());
        assert_equal(model, move->get_model());
    }

    // Check destructor did not call twice (which would raise a CUDA error upon trying
    // to release the same resource).
    kernel::throw_errors();
}

TEST_CASE("BulkCurrent")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);
    Stencil stencil{1e-2, 1, 0.75};
    Model model{1, 2, 3, 0, 0, 0};

    Integrator simulation(initial, stencil, model);
    Field field = simulation.get_field();

    // Bulk chemical potential is evaluated point-wise
    Field mu(Ny, Nx);
    for (int i = 0; i < Ny; ++i)
        for (int j = 0; j < Nx; ++j)
            mu(i, j) = bulk_chemical_potential(field(i, j), model);

    // Current $\vec{J} = - \nabla \mu$:
    Current expected = gradient(mu, stencil);
    for (int c = 0; c < d; ++c) expected[c] *= -1;

    Current actual = simulation.get_current();
    assert_equal<tight_tol>(expected[0], actual[0]);
    assert_equal<tight_tol>(expected[1], actual[1]);
}

TEST_CASE("PassiveSurfaceCurrent")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);
    Stencil stencil{1e-2, 1, 0.75};
    Model model{0, 0, 0, 1, 0, 0};

    Integrator simulation(initial, stencil, model);
    Field field = simulation.get_field();

    // Current $\vec{J} = -\nabla \mu$ with $\mu = - \kappa \nabla^2 \phi$:
    Field mu = -model.kappa * laplacian(field, stencil);
    Current expected = gradient(mu, stencil);
    for (int c = 0; c < d; ++c) expected[c] *= -1;

    Current actual = simulation.get_current();
    assert_equal<tight_tol>(expected[0], actual[0]);
    assert_equal<tight_tol>(expected[1], actual[1]);
}

TEST_CASE("LocalActiveCurrent")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);
    Stencil stencil{1e-2, 1, 0.75};
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

    Current expected = gradient(mu, stencil);
    for (int c = 0; c < d; ++c) expected[c] *= -1;

    Current actual = simulation.get_current();
    assert_equal<tight_tol>(expected[0], actual[0]);
    assert_equal<tight_tol>(expected[1], actual[1]);
}
