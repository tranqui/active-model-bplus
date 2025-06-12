#pragma once
#include "math_primitives.h"

#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST_DEVICE
#endif

namespace finite_difference
{
    // Indicating order of derivative (i.e. y(x), y'(x), y''(x)):
    enum Derivative { Zero, First, Second };

    namespace details
    {
        /// Finite difference coefficients for 0th, 1st and 2nd derivatives at various
        // orders of expansion.

        /**
         * Coefficients for derivatives on different grids.
         * 
         * If Stagger=Central, then the derivatives are calculated at the same points
         *   as the support points.
         * 
         * If Stagger=Right or Left, then elements of derivatives are in between those
         *   of the source field. The staggered grid is sometimes written with half-integer
         *   indices, e.g. in 1d the gradient with field phi:
         * 
         *     phi(i - 1)                 phi(i)                 phi(i + 1)
         *                 grad(i - 1/2)          grad(i + 1/2)
         * 
         * We cannot use half-integer indices internally for data representation, so we have
         *   to use an implicit offset from integral indices.
         * 
         * @tparam Order: Derivatives must be implemented for the specific
         *                Order of the numerical approximation, which is done below
         *                by complete class specialisation.
         * @tparam Stagger: direction of staggering of grid from integer indices.
         *           If Stagger=Left : grad[i] -> grad(i - 1/2)
         *           If Stagger=Right: grad[i] -> grad(i + 1/2)
         */
        template <std::size_t Order, StaggerGrid Stagger> struct Coefficients { };

        template <>
        struct Coefficients<2, Central>
        {
            static constexpr std::array<Scalar, 1> zero{1};
            static constexpr std::array<Scalar, 3> first{-0.5, 0, 0.5};
            static constexpr std::array<Scalar, 3> second{1, -2, 1};
        };

        template <>
        struct Coefficients<4, Central>
        {
            static constexpr std::array<Scalar, 1> zero{1};
            static constexpr std::array<Scalar, 5> first{1./12, -2./3, 0, 2./3, -1./12};
            static constexpr std::array<Scalar, 5> second{-1./12, 4./3, -5./2, 4./3, -1./12};
        };

        template <>
        struct Coefficients<6, Central>
        {
            static constexpr std::array<Scalar, 1> zero{1};
            static constexpr std::array<Scalar, 7> first{-1./60, 3./20, -3./4, 0, 3./4, -3./20, 1./60};
            static constexpr std::array<Scalar, 7> second{1./90, -3./20, 3./2, -49./18, 3./2, -3./20, 1./90};
        };

        template <>
        struct Coefficients<8, Central>
        {
            static constexpr std::array<Scalar, 1> zero{1};
            static constexpr std::array<Scalar, 9> first{1./280, -4./105, 1./5, -4./5, 0, 4./5, -1./5, 4./105, -1./280};
            static constexpr std::array<Scalar, 9> second{-1./560, 8./315, -1./5, 8./5, -205./72, 8./5, -1./5, 8./315, -1./560};
        };

        template <>
        struct Coefficients<2, Right>
        {
            static constexpr std::array<Scalar, 2> zero{1./2, 1./2};
            static constexpr std::array<Scalar, 2> first{-1, 1};
            static constexpr std::array<Scalar, 4> second{1./2, -1./2, -1./2, 1./2};
        };

        template <>
        struct Coefficients<4, Right>
        {
            static constexpr std::array<Scalar, 4> zero{-1./16, 9./16, 9./16, -1./16};
            static constexpr std::array<Scalar, 4> first{1./24, -27./24, 27./24, -1./24};
            static constexpr std::array<Scalar, 6> second{-5./48, 39./48, -34./48, -34./48, 39./48, -5./48};
        };

        template <>
        struct Coefficients<6, Right>
        {
            static constexpr std::array<Scalar, 6> zero{3./256, -25./256, 150./256, 150./256, -25./256, 3./256};
            static constexpr std::array<Scalar, 6> first{-9./1920, 125./1920, -2250./1920, 2250./1920, -125./1920, 9./1920};
            static constexpr std::array<Scalar, 8> second{  259./11520, -2495./11520, 11691./11520, -9455./11520,
                                                          -9455./11520, 11691./11520, -2495./11520,   259./11520};
        };

        template <>
        struct Coefficients<8, Right>
        {
            static constexpr std::array<Scalar, 8> zero{-5./2048, 49./2048, -245./2048, 1225./2048, 1225./2048, -245./2048, 49./2048, -5./2048};
            static constexpr std::array<Scalar, 8> first{    75./107520, -1029./107520, 8575./107520, -128625./107520,
                                                         128625./107520, -8575./107520, 1029./107520,     -75./107520};
            static constexpr std::array<Scalar, 10> second{  -3229./645120,  37107./645120, -204300./645120, 745108./645120, -574686./645120,
                                                           -574686./645120, 745108./645120, -204300./645120,  37107./645120,   -3229./645120};
        };

        // Coefficients for left stagger are the same as for right stagger.
        template <std::size_t Order>
        struct Coefficients<Order, Left> : public Coefficients<Order, Right> { };

        template <Derivative D, std::size_t Order, StaggerGrid Stagger>
        struct StencilBase
        {
            CUDA_HOST_DEVICE static constexpr auto get_coefficients()
            {
                static_assert(D == Zero or D == First or D == Second);
                if constexpr (D == Zero)
                    return Coefficients<Order, Stagger>::zero;
                else if constexpr (D == First)
                    return Coefficients<Order, Stagger>::first;
                else return Coefficients<Order, Stagger>::second;
            }

            CUDA_HOST_DEVICE static constexpr Scalar coefficient(int k)
            {
                static_assert(D == Zero or D == First or D == Second);
                const auto& coefficients = get_coefficients();
                return coefficients[k];
            }

            static constexpr auto coefficients = get_coefficients();
            static constexpr std::size_t size = coefficients.size();
        };

        template <Derivative D, std::size_t Order, StaggerGrid Stagger>
        struct Stencil : public StencilBase<D, Order, Stagger> { };

        template <Derivative D, std::size_t Order>
        struct Stencil<D, Order, Central> : public StencilBase<D, Order, Central>
        {
            using StencilBase<D, Order, Central>::size;
            static constexpr int start = -static_cast<int>(size)/2;
        };

        template <Derivative D, std::size_t Order>
        struct Stencil<D, Order, Left> : public StencilBase<D, Order, Left>
        {
            using StencilBase<D, Order, Left>::size;
            static constexpr int start = -static_cast<int>(size)/2;
        };

        template <Derivative D, std::size_t Order>
        struct Stencil<D, Order, Right> : public StencilBase<D, Order, Right>
        {
            using StencilBase<D, Order, Right>::size;
            static constexpr int start = -static_cast<int>(size)/2 + 1;
        };
    }

    // Apply central derivatives on a 1d set of support points.

    template <Derivative D, std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar apply(T&& data)
    {
        static_assert(Order >= 1);
        using stencil = details::Stencil<D, Order, Stagger>;
        // static_assert(data.size() == stencil::size);

        Scalar result{0};
        for (std::size_t k = 0; k < stencil::size; ++k)
            result += stencil::coefficient(k) * data[k];
        return result;
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar zero(T&& data)
    {
        return apply<Zero, Order, Stagger>(std::forward<T>(data));
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar first(T&& data)
    {
        return apply<First, Order, Stagger>(std::forward<T>(data));
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar second(T&& data)
    {
        return apply<Second, Order, Stagger>(std::forward<T>(data));
    }

    template <std::size_t Order, typename T>
    CUDA_HOST_DEVICE inline Scalar zero(T&& data)
    {
        return apply<Zero, Order, Central>(std::forward<T>(data));
    }

    template <std::size_t Order, typename T>
    CUDA_HOST_DEVICE inline Scalar first(T&& data)
    {
        return apply<First, Order, Central>(std::forward<T>(data));
    }

    template <std::size_t Order, typename T>
    CUDA_HOST_DEVICE inline Scalar second(T&& data)
    {
        return apply<Second, Order, Central>(std::forward<T>(data));
    }

    // Apply coefficients for a particular derivative to the stencil at position (i, j).

    template <Derivative D, std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar apply_x(T&& data, int i, int j)
    {
        using stencil = details::Stencil<D, Order, Stagger>;
        Scalar result{0};
        for (int k = 0; k < stencil::size; ++k)
            result += stencil::coefficient(k) * (data[i][j + k + stencil::start]);
        return result;
    }

    template <Derivative D, std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar apply_y(T&& data, int i, int j)
    {
        using stencil = details::Stencil<D, Order, Stagger>;
        Scalar result{0};
        for (int k = 0; k < stencil::size; ++k)
            result += stencil::coefficient(k) * (data[i + k + stencil::start][j]);
        return result;
    }

    // Aliases to apply coefficients for derivatives of different orders.

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar zero_x(T&& data, int i, int j)
    {
        return apply_x<Zero, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar zero_y(T&& data, int i, int j)
    {
        return apply_y<Zero, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar first_x(T&& data, int i, int j)
    {
        return apply_x<First, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar first_y(T&& data, int i, int j)
    {
        return apply_y<First, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar second_x(T&& data, int i, int j)
    {
        return apply_x<Second, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    CUDA_HOST_DEVICE inline Scalar second_y(T&& data, int i, int j)
    {
        return apply_y<Second, Order, Stagger>(std::forward<T>(data), i, j);
    }

    // Aliases to default to central derivatives if no staggering of grid selected.

    template <std::size_t Order, typename T>
    CUDA_HOST_DEVICE inline Scalar zero_x(T&& data, int i, int j)
    {
        return zero_x<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    CUDA_HOST_DEVICE inline Scalar zero_y(T&& data, int i, int j)
    {
        return zero_y<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    CUDA_HOST_DEVICE inline Scalar first_x(T&& data, int i, int j)
    {
        return first_x<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    CUDA_HOST_DEVICE inline Scalar first_y(T&& data, int i, int j)
    {
        return first_y<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    CUDA_HOST_DEVICE inline Scalar second_x(T&& data, int i, int j)
    {
        return second_x<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    CUDA_HOST_DEVICE inline Scalar second_y(T&& data, int i, int j)
    {
        return second_y<Order, Central>(std::forward<T>(data), i, j);
    }

    namespace isotropic
    {
        // Operators on a 3x3 stencil at quadratic order where the leading error
        // is isotropic. See e.g.:
        // https://en.wikipedia.org/wiki/Nine-point_stencil#Implementation
        CUDA_HOST_DEVICE constexpr inline Scalar second_x_coefficients(int i, int j)
        {
            Scalar coefficients[3][3] = {
                {1./12, -1./6, 1./12},
                {5./6 , -5./3, 5./6 },
                {1./12, -1./6, 1./12}
            };
            return coefficients[i][j];
        }

        /// Coefficients in y direction are simply the transpose of the x coefficients.

        CUDA_HOST_DEVICE constexpr inline Scalar second_y_coefficients(int i, int j)
        {
            return second_x_coefficients(j, i);
        }

        /// Apply square stencil to tile at (i, j)
        template <typename T, typename Coefficients>
        CUDA_HOST_DEVICE inline Scalar apply_square_grid(T&& data, int i, int j,
                                                         Coefficients coefficients)
        {
            return coefficients(0,0) * data[i-1][j-1]
                 + coefficients(0,1) * data[i-1][ j ]
                 + coefficients(0,2) * data[i-1][j+1]
                 + coefficients(1,0) * data[ i ][j-1]
                 + coefficients(1,1) * data[ i ][ j ]
                 + coefficients(1,2) * data[ i ][j+1]
                 + coefficients(2,0) * data[i+1][j-1]
                 + coefficients(2,1) * data[i+1][ j ]
                 + coefficients(2,2) * data[i+1][j+1];
        }

        // template <typename T>
        // CUDA_HOST_DEVICE inline Scalar laplacian(T&& data, int i, int j)
        // {
        //     return apply_square_grid(std::forward<T>(data), i, j, laplacian_coefficients);
        // }

        template <typename T>
        CUDA_HOST_DEVICE inline Scalar second_x(T&& data, int i, int j)
        {
            return apply_square_grid(std::forward<T>(data), i, j, second_x_coefficients);
        }

        template <typename T>
        CUDA_HOST_DEVICE inline Scalar second_y(T&& data, int i, int j)
        {
            return apply_square_grid(std::forward<T>(data), i, j, second_y_coefficients);
        }
    }

    namespace tjhung
    {
        // Operators on a 3x3 stencil at quadratic order.
        // Note these are not identical to standard isotropic stencil, see e.g.:
        // https://en.wikipedia.org/wiki/Nine-point_stencil#Implementation
        // Rather, these come from Tjhung et al's discretisation, cf:
        // https://elsentjhung.github.io/posts/2020/12/discretization/
        // and https://github.com/elsentjhung/active-model-B-plus
        CUDA_HOST_DEVICE constexpr inline Scalar first_x_coefficients(int i, int j)
        {
            Scalar coefficients[3][3] = {
                {-1./10, 0, 1./10},
                {-3./10, 0, 3./10},
                {-1./10, 0, 1./10}
            };
            return coefficients[i][j];
        }

        CUDA_HOST_DEVICE constexpr inline Scalar second_x_coefficients(int i, int j)
        {
            // Tjhung et al (2018) use these, cf.:
            // https://elsentjhung.github.io/posts/2020/12/discretization/
            // https://github.com/elsentjhung/active-model-B-plus
            Scalar coefficients[3][3] = {
                {-0.25,  0.5, -0.25},
                { 1.50, -3.0,  1.50},
                {-0.25,  0.5, -0.25}
            };
            return coefficients[i][j];
        }

        /// Coefficients in y direction are simply the transpose of the x coefficients.

        CUDA_HOST_DEVICE constexpr inline Scalar first_y_coefficients(int i, int j)
        {
            return first_x_coefficients(j, i);
        }

        CUDA_HOST_DEVICE constexpr inline Scalar second_y_coefficients(int i, int j)
        {
            return second_x_coefficients(j, i);
        }

        /// Apply square stencil to tile at (i, j)
        template <typename T, typename Coefficients>
        CUDA_HOST_DEVICE inline Scalar apply_square_grid(T&& data, int i, int j,
                                                         Coefficients coefficients)
        {
            return coefficients(0,0) * data[i-1][j-1]
                 + coefficients(0,1) * data[i-1][ j ]
                 + coefficients(0,2) * data[i-1][j+1]
                 + coefficients(1,0) * data[ i ][j-1]
                 + coefficients(1,1) * data[ i ][ j ]
                 + coefficients(1,2) * data[ i ][j+1]
                 + coefficients(2,0) * data[i+1][j-1]
                 + coefficients(2,1) * data[i+1][ j ]
                 + coefficients(2,2) * data[i+1][j+1];
        }

        template <typename T>
        CUDA_HOST_DEVICE inline Scalar first_x(T&& data, int i, int j)
        {
            return apply_square_grid(std::forward<T>(data), i, j, first_x_coefficients);
        }

        template <typename T>
        CUDA_HOST_DEVICE inline Scalar first_y(T&& data, int i, int j)
        {
            return apply_square_grid(std::forward<T>(data), i, j, first_y_coefficients);
        }

        template <typename T>
        CUDA_HOST_DEVICE inline Scalar second_x(T&& data, int i, int j)
        {
            return apply_square_grid(std::forward<T>(data), i, j, second_x_coefficients);
        }

        template <typename T>
        CUDA_HOST_DEVICE inline Scalar second_y(T&& data, int i, int j)
        {
            return apply_square_grid(std::forward<T>(data), i, j, second_y_coefficients);
        }
    }
}
