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

// Loop through field and generate local stencil for central finite
// difference calculations.
template <typename Operation>
inline void apply_operation(Field field, Operation operation)
{
    const auto Ny{field.rows()}, Nx{field.cols()};
    constexpr std::size_t stencil_size = 1 + order;

    for (int i = 0; i < Ny; ++i)
    {
        // Nearest neighbours in y-direction w/ periodic boundaries:
        std::array<int, stencil_size> iy;
        for (int k = 0; k < stencil_size; ++k)
        {
            iy[k] = i - order + k + 1;
            if (iy[k] < 0) iy[k] += Ny;
            if (iy[k] >= Ny) iy[k] -= Ny;
        }

        for (int j = 0; j < Nx; ++j)
        {
            // Nearest neighbours in x-direction w/ periodic boundaries:
            std::array<int, stencil_size> ix;
            for (int k = 0; k < stencil_size; ++k)
            {
                ix[k] = j - order + k + 1;
                if (ix[k] < 0) ix[k] += Nx;
                if (ix[k] >= Nx) ix[k] -= Nx;
            }

            // Retrieve field points indexed by stencil.
            std::array<Scalar, stencil_size> stencil_y, stencil_x;
            for (int k = 0; k < stencil_size; ++k)
            {
                stencil_y[k] = field(iy[k], j);
                stencil_x[k] = field(i, ix[k]);
            }

            operation(i, j, stencil_y, stencil_x);
        }
    }
}

namespace finite_difference
{
    // Find gradient of field by central finite differences.
    inline Gradient gradient(Field field, Stencil stencil)
    {
        const auto Ny{field.rows()}, Nx{field.cols()};
        Gradient gradient{Field(Ny, Nx), Field(Ny, Nx)};

        auto grad = [&](int i, int j, auto stencil_y, auto stencil_x)
        {
            gradient[0](i, j) = first(stencil_y) / stencil.dy;
            gradient[1](i, j) = first(stencil_x) / stencil.dx;
        };
        apply_operation(field, grad);

        return gradient;
    }

    // Find laplacian of field by central finite differences.
    inline Field laplacian(const FieldRef& field, Stencil stencil)
    {
        const Scalar dxInv{1/stencil.dx}, dyInv{1/stencil.dy};
        const auto Ny{field.rows()}, Nx{field.cols()};
        Field laplacian{Ny, Nx};

        auto lap = [&](int i, int j, auto stencil_y, auto stencil_x)
        {
            laplacian(i, j) = dyInv*dyInv * second(stencil_y)
                            + dxInv*dxInv * second(stencil_x);
        };
        apply_operation(field, lap);

        return laplacian;
    }

    // Find divergence of vector field by central finite differences.
    inline Field divergence(const Gradient& grad, Stencil stencil)
    {
        const Scalar dxInv{1/stencil.dx}, dyInv{1/stencil.dy};
        const auto Ny{grad[0].rows()}, Nx{grad[0].cols()};
        Field divergence = Field::Zero(Ny, Nx);

        auto div_y = [&](int i, int j, auto stencil_y, auto stencil_x)
        {
            divergence(i, j) += dyInv * first(stencil_y);
        };
        auto div_x = [&](int i, int j, auto stencil_y, auto stencil_x)
        {
            divergence(i, j) += dxInv * first(stencil_x);
        };

        apply_operation(grad[0], div_y);
        apply_operation(grad[1], div_x);

        return divergence;
    }

    template <StaggerGrid Stagger>
    inline Gradient staggered_gradient(Field field, Stencil stencil)
    {
        const auto Ny{field.rows()}, Nx{field.cols()};
        Gradient grad{Field(Ny, Nx), Field(Ny, Nx)};

        for (int i = 0; i < Ny; ++i)
        {
            // Nearest neighbours in y-direction w/ periodic boundaries:
            int ip{i};
            if constexpr (Stagger == Right) ip++;
            int im{ip-1};
            if (im < 0) im += Ny;
            if (ip >= Ny) ip -= Ny;

            for (int j = 0; j < Nx; ++j)
            {
                // Nearest neighbours in x-direction w/ periodic boundaries:
                int jp{j};
                if constexpr (Stagger == Right) jp++;
                int jm{jp-1};
                if (jm < 0) jm += Nx;
                if (jp >= Nx) jp -= Nx;

                grad[0](i, j) = (field(ip, j ) - field(im, j )) / stencil.dy;
                grad[1](i, j) = (field(i , jp) - field(i , jm)) / stencil.dx;
            }
        }

        return grad;
    }

    template <StaggerGrid Stagger>
    inline Field staggered_laplacian(Field field, Stencil stencil)
    {
        const auto Ny{field.rows()}, Nx{field.cols()};
        Field lap(Ny, Nx);

        for (int i = 0; i < Ny; ++i)
        {
            // Nearest neighbours in y-direction w/ periodic boundaries:
            int i3{i}, i4{i+1};
            if constexpr (Stagger == Right)
            {
                i3++;
                i4++;
            }
            int i2{i3-1}, i1{i3-2};
            if (i1 < 0) i1 += Ny;
            if (i2 < 0) i2 += Ny;
            if (i3 >= Ny) i3 -= Ny;
            if (i4 >= Ny) i4 -= Ny;

            for (int j = 0; j < Nx; ++j)
            {
                // Nearest neighbours in x-direction w/ periodic boundaries:
                int j3{j}, j4{j+1};
                if constexpr (Stagger == Right)
                {
                    j3++;
                    j4++;
                }
                int j2{j3-1}, j1{j3-2};
                if (j1 < 0) j1 += Nx;
                if (j2 < 0) j2 += Nx;
                if (j3 >= Nx) j3 -= Nx;
                if (j4 >= Nx) j4 -= Nx;

                Scalar grad_x2 = 0.25 * (field(i2, j3) - field(i2, j1)
                                    + field(i3, j3) - field(i3, j1)) / stencil.dx;
                Scalar grad_x3 = 0.25 * (field(i2, j4) - field(i2, j2)
                                    + field(i3, j4) - field(i3, j2)) / stencil.dx;

                Scalar grad_y2 = 0.25 * (field(i3, j2) - field(i1, j2)
                                    + field(i3, j3) - field(i1, j3)) / stencil.dy;
                Scalar grad_y3 = 0.25 * (field(i4, j2) - field(i2, j2)
                                    + field(i4, j3) - field(i2, j3)) / stencil.dy;

                Scalar lap_x = (grad_x3 - grad_x2) / stencil.dx;
                Scalar lap_y = (grad_y3 - grad_y2) / stencil.dy;
                lap(i, j) = lap_x + lap_y;
            }
        }

        return lap;
    }

    template <StaggerGrid Stagger>
    inline Field staggered_divergence(Gradient grad, Stencil stencil)
    {
        const auto Ny{grad[0].rows()}, Nx{grad[0].cols()};
        Field div(Ny, Nx);

        for (int i = 0; i < Ny; ++i)
        {
            // Nearest neighbours in y-direction w/ periodic boundaries:
            int ip{i};
            if constexpr (Stagger == Right) ip++;
            int im{ip-1};
            if (im < 0) im += Ny;
            if (ip >= Ny) ip -= Ny;

            for (int j = 0; j < Nx; ++j)
            {
                // Nearest neighbours in x-direction w/ periodic boundaries:
                int jp{j};
                if constexpr (Stagger == Right) jp++;
                int jm{jp-1};
                if (jm < 0) jm += Nx;
                if (jp >= Nx) jp -= Nx;

                div(i, j) = (grad[0](ip, j ) - grad[0](im, j )) / stencil.dy
                        + (grad[1](i , jp) - grad[1](i , jm)) / stencil.dx;
            }
        }

        return div;
    }
}
