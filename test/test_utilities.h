#pragma once
#include "finite_difference.h"
#include "parameters.h"


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



namespace finite_difference
{
    // Loop through each tile for given stencil and apply operation.
    template <Derivative D, StaggerGrid Stagger, typename Operation>
    inline void for_each_tile(Field field, Operation operation)
    {
        const auto Ny{field.rows()}, Nx{field.cols()};
        using stencil = details::Stencil<D, order, Stagger>;

        for (int i = 0; i < Ny; ++i)
        {
            // Nearest neighbours in y-direction w/ periodic boundaries:
            std::array<int, stencil::size> iy;
            for (int k = 0; k < stencil::size; ++k)
            {
                iy[k] = i + k + stencil::start;
                if (iy[k] < 0) iy[k] += Ny;
                if (iy[k] >= Ny) iy[k] -= Ny;
            }

            for (int j = 0; j < Nx; ++j)
            {
                // Nearest neighbours in x-direction w/ periodic boundaries:
                std::array<int, stencil::size> ix;
                for (int k = 0; k < stencil::size; ++k)
                {
                    ix[k] = j + k + stencil::start;
                    if (ix[k] < 0) ix[k] += Nx;
                    if (ix[k] >= Nx) ix[k] -= Nx;
                }

                // Retrieve field points indexed by stencil.
                std::array<Scalar, stencil::size> tile_y, tile_x;
                for (int k = 0; k < stencil::size; ++k)
                {
                    tile_y[k] = field(iy[k], j);
                    tile_x[k] = field(i, ix[k]);
                }

                operation(i, j, tile_y, tile_x);
            }
        }
    }

    template <Derivative D, typename Operation>
    inline void for_each_tile(Field field, Operation operation)
    {
        for_each_tile<D, Central>(field, operation);
    }

    // Find gradient of field by central finite differences.
    template <StaggerGrid Stagger>
    inline Gradient gradient(Field field, Stencil stencil)
    {
        const auto Ny{field.rows()}, Nx{field.cols()};
        Gradient gradient{Field(Ny, Nx), Field(Ny, Nx)};

        auto grad = [&](int i, int j, auto tile_y, auto tile_x)
        {
            gradient[0](i, j) = first<order, Stagger>(tile_y) / stencil.dy;
            gradient[1](i, j) = first<order, Stagger>(tile_x) / stencil.dx;
        };
        for_each_tile<First, Stagger>(field, grad);

        return gradient;
    }

    // Find laplacian of field by central finite differences.
    template <StaggerGrid StaggerY, StaggerGrid StaggerX>
    inline Field laplacian(const FieldRef& field, Stencil stencil)
    {
        const Scalar dyInv{1/stencil.dy}, dxInv{1/stencil.dx};
        const auto Ny{field.rows()}, Nx{field.cols()};
        Field laplacian(Ny, Nx);

        auto lap = [&](int i, int j, auto tile_y, auto tile_x)
        {
            laplacian(i, j) = dyInv*dyInv * second<order>(tile_y)
                            + dxInv*dxInv * second<order>(tile_x);
        };
        for_each_tile<Second>(field, lap);

        // Interpolate $\partial_{xx}^2 \phi$ over any stagger in y-dir:
        if constexpr (StaggerY != Central)
        {
            Field tmp(Ny, Nx);
            auto stagger_y = [&](int i, int j, auto tile_y, auto tile_x)
            {
                tmp(i, j) = zero<order, StaggerY>(tile_y);
            };
            for_each_tile<Zero, StaggerY>(laplacian, stagger_y);
            laplacian = tmp;
        }

        // Interpolate $\partial_{yy}^2 \phi$ over any stagger in x-dir:
        if constexpr (StaggerX != Central)
        {
            Field tmp(Ny, Nx);
            auto stagger_x = [&](int i, int j, auto tile_y, auto tile_x)
            {
                tmp(i, j) = zero<order, StaggerX>(tile_x);
            };
            for_each_tile<Zero, StaggerX>(laplacian, stagger_x);
            laplacian = tmp;
        }

        return laplacian;
    }

    // Find divergence of vector field by central finite differences.
    template <StaggerGrid Stagger>
    inline Field divergence(const Gradient& grad, Stencil stencil)
    {
        const Scalar dyInv{1/stencil.dy}, dxInv{1/stencil.dx};
        const auto Ny{grad[0].rows()}, Nx{grad[0].cols()};
        Field divergence = Field::Zero(Ny, Nx);

        auto div_y = [&](int i, int j, auto tile_y, auto tile_x)
        {
            divergence(i, j) += dyInv * first<order, Stagger>(tile_y);
        };
        auto div_x = [&](int i, int j, auto tile_y, auto tile_x)
        {
            divergence(i, j) += dxInv * first<order, Stagger>(tile_x);
        };

        for_each_tile<First, Stagger>(grad[0], div_y);
        for_each_tile<First, Stagger>(grad[1], div_x);

        return divergence;
    }

    inline Gradient gradient(Field field, Stencil stencil)
    {
        return gradient<Central>(field, stencil);
    }

    inline Field laplacian(const FieldRef& field, Stencil stencil)
    {
        return laplacian<Central, Central>(field, stencil);
    }

    inline Field divergence(const Gradient& grad, Stencil stencil)
    {
        return divergence<Central>(grad, stencil);
    }
}
