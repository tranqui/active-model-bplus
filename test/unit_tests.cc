#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "integrator.h"
#include "foreach.h"


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
bool is_equal(Scalar a, Scalar b)
{
    return std::abs(a - b) < tolerance;
}

bool is_equal(Scalar a, Scalar b)
{
    return a == b;
}

// Eigen arrays we have to check matrix size and then element-wise data.
template <Scalar tolerance, typename T1, typename T2,
          typename = std::enable_if_t<is_eigen_matrix<std::decay_t<T1>>::value and
                                      is_eigen_matrix<std::decay_t<T2>>::value>>
bool is_equal(T1&& a, T2&& b)
{
    if (a.rows() != b.rows()) return false;
    if (a.cols() != b.cols()) return false;
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            if constexpr (tolerance == 0)
            {
                if (a(i,j) != b(i,j)) return false;
            }
            else if (std::abs(a(i,j) - b(i,j)) > tolerance) return false;
    return true;
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_eigen_matrix<std::decay_t<T1>>::value and
                                      is_eigen_matrix<std::decay_t<T2>>::value>>
bool is_equal(T1&& a, T2&& b)
{
    return is_equal<Scalar{0}>(std::forward<T1>(a), std::forward<T2>(b));
}

// Element-wise test in tuples.
template <typename... T>
bool is_equal(const std::tuple<T...>& a, const std::tuple<T...>& b)
{
    bool flag{true};
    auto test = [&](auto m)
    {
        if (std::get<m>(a) != std::get<m>(b)) flag = false;
    };
    for_each(std::index_sequence_for<T...>{}, test);
    return flag;
}

// Overload for structured data: cast to tuple then test.
template <typename Params,
          typename = std::enable_if_t<has_as_tuple<Params>::value>>
bool is_equal(const Params& a, const Params& b)
{
    return is_equal(a.as_tuple(), b.as_tuple());
}


/// Expected physical values.


inline Scalar bulk_chemical_potential(Scalar field, const Model& model)
{
    return model.a * field
         + model.b * field * field
         + model.c * field * field * field;
}

// Find gradient of field by second-order central finite differences.
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

// Find laplacian of field by second-order central finite differences.
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

/**
 * Find gradient of field on a staggered grid by second-order finite difference.
 * 
 * The gradient is evaluated at points on a staggered grid, i.e. the elements
 *   of grad are in between the elements of field(i, j). The staggered grid is
 *   sometimes written with half-integer indices, e.g. in 1d:
 * 
 *     phi(i - 1)                 phi(i)                 phi(i + 1)
 *                 grad(i - 1/2)          grad(i + 1/2)
 * 
 * We use a second-order finite-difference stencil to calculate the gradient, e.g.
 * 
 *   grad(i - 1/2) = (phi(i) - phi(i - 1)) / dx.
 * 
 * We cannot use half-integer indices internally for data representation, so we have
 *   to use an implicit offset from integral indices.
 * 
 * @tparam Offset: offset in half integers should be -1 or 1 to make indices integral.
 *           If Offset=Left : grad[i] -> grad(i - 1/2)
 *           If Offset=Right: grad[i] -> grad(i + 1/2)
 *                 Generally: grad[i] -> grad(i + Offset/2)
 */
template <StaggeredGridDirection Offset>
inline Gradient staggered_gradient(Field field, Stencil stencil)
{
    const auto Ny{field.rows()}, Nx{field.cols()};
    Gradient grad{Field(Ny, Nx), Field(Ny, Nx)};

    for (int i = 0; i < Ny; ++i)
    {
        // Nearest neighbours in y-direction w/ periodic boundaries:
        int ip{i};
        if constexpr (Offset == Right) ip++;
        int im{ip-1};
        if (im < 0) im += Ny;
        if (ip >= Ny) ip -= Ny;

        for (int j = 0; j < Nx; ++j)
        {
            // Nearest neighbours in x-direction w/ periodic boundaries:
            int jp{j};
            if constexpr (Offset == Right) jp++;
            int jm{jp-1};
            if (jm < 0) jm += Nx;
            if (jp >= Nx) jp -= Nx;

            grad[0](i, j) = (field(ip, j ) - field(im, j )) / stencil.dy;
            grad[1](i, j) = (field(i , jp) - field(i , jm)) / stencil.dx;
        }
    }

    return grad;
}

template <StaggeredGridDirection Offset>
inline Field staggered_divergence(Gradient grad, Stencil stencil)
{
    const auto Ny{grad[0].rows()}, Nx{grad[0].cols()};
    Field div(Ny, Nx);

    for (int i = 0; i < Ny; ++i)
    {
        // Nearest neighbours in y-direction w/ periodic boundaries:
        int ip{i};
        if constexpr (Offset == Right) ip++;
        int im{ip-1};
        if (im < 0) im += Ny;
        if (ip >= Ny) ip -= Ny;

        for (int j = 0; j < Nx; ++j)
        {
            // Nearest neighbours in x-direction w/ periodic boundaries:
            int jp{j};
            if constexpr (Offset == Right) jp++;
            int jm{jp-1};
            if (jm < 0) jm += Nx;
            if (jp >= Nx) jp -= Nx;

            div(i, j) = (grad[0](ip, j ) - grad[0](im, j )) / stencil.dy
                      + (grad[1](i , jp) - grad[1](i , jm)) / stencil.dx;
        }
    }

    return div;
}


/// The unit tests.


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
    Current expected = staggered_gradient<Right>(mu, stencil);
    for (int c = 0; c < d; ++c) expected[c] *= -1;

    Current actual = simulation.get_current();
    CHECK(is_equal<tight_tol>(expected[0], actual[0]));
    CHECK(is_equal<tight_tol>(expected[1], actual[1]));
}

// Test divergence of current is evaluated correctly during integration.
// After one step, the field should be $\phi(dt) - \phi(0) = -dt * \nabla \cdot \vec{J}$
TEST_CASE("DivergenceTest")
{
    int Nx{64}, Ny{32};
    Field initial = Field::Random(Ny, Nx);
    Scalar dt{1e-2};
    Stencil stencil{dt, 1, 0.75};
    Model model{1, 0, 0, 0, 0, 0}; // normal heat/diffusion equation with D=1

    Integrator simulation(initial, stencil, model);
    Field field = simulation.get_field();

    // Current $\vec{J} = - \nabla \mu$:
    Current current = staggered_gradient<Right>(field, stencil);
    for (int c = 0; c < d; ++c) current[c] *= -1;

    simulation.run(1);
    Field actual = simulation.get_field();

    {
        // Expect: $-\nabla \cdot \vec{J} = \nabla^2 \phi$.
        Field expected = field + dt * laplacian(field, stencil);
        CHECK(is_equal<tight_tol>(expected, actual));
    }

    {
        // Alternative: $-\nabla \cdot \vec{J} = \nabla^2 \phi$.
        Field expected = field - dt * staggered_divergence<Left>(current, stencil);
        CHECK(is_equal<tight_tol>(expected, actual));
    }
}

/*TEST_CASE("PassiveSurfaceCurrent")
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
    CHECK(is_equal<tight_tol>(expected[0], actual[0]));
    CHECK(is_equal<tight_tol>(expected[1], actual[1]));
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
    CHECK(is_equal<tight_tol>(expected[0], actual[0]));
    CHECK(is_equal<tight_tol>(expected[1], actual[1]));
}

TEST_CASE("ConservationTest")
{
    int Nx{512}, Ny{256};
    Field initial = 0.1 * Field::Random(Ny, Nx);
    initial -= Field::Constant(Ny, Nx, initial.mean());

    constexpr int nsteps = 1000;
    Stencil stencil{1e-1, 1, 0.75};

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

    Stencil stencil{1e-1, 1, 0.75};
    // Parameters for phase separation with binodal at \phi = \pm 1.
    Model model{-0.25, 0, 0.25, 1, 0, 0};

    Integrator simulation(initial, stencil, model);

    constexpr int nsteps = 10000;
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
}*/
