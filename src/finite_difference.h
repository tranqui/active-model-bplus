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
        /// Finite difference coefficients for 0th, 1st and 2nd derivatives at various orders of expansion.

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
            static constexpr std::array<Scalar, 5> first{1/12, -2/3, 0, 2/3, -1/12};
            static constexpr std::array<Scalar, 5> second{-1/12, 4/3, -5/2, 4/3, -1/12};
        };

        template <>
        struct Coefficients<6, Central>
        {
            static constexpr std::array<Scalar, 1> zero{1};
            static constexpr std::array<Scalar, 7> first{-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60};
            static constexpr std::array<Scalar, 7> second{1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90};
        };

        template <>
        struct Coefficients<8, Central>
        {
            static constexpr std::array<Scalar, 1> zero{1};
            static constexpr std::array<Scalar, 9> first{1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280};
            static constexpr std::array<Scalar, 9> second{-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560};
        };

        template <>
        struct Coefficients<2, Right>
        {
            static constexpr std::array<Scalar, 2> zero{1/2, 1/2};
            static constexpr std::array<Scalar, 2> first{-1, 1};
            static constexpr std::array<Scalar, 4> second{1/2, -1/2, -1/2, 1/2};
        };

        template <>
        struct Coefficients<4, Right>
        {
            static constexpr std::array<Scalar, 4> zero{-1/16, 9/16, 9/16, -1/16};
            static constexpr std::array<Scalar, 4> first{1/24, -27/24, 27/24, -1/24};
            static constexpr std::array<Scalar, 6> second{-5/48, 39/48, -34/48, -34/48, 39/48, -5/48};
        };

        template <>
        struct Coefficients<6, Right>
        {
            static constexpr std::array<Scalar, 6> zero{3/256, -25/256, 150/256, 150/256, -25/256, 3/256};
            static constexpr std::array<Scalar, 6> first{-9/1920, 125/1920, -2250/1920, 2250/1920, -125/1920, 9/1920};
            static constexpr std::array<Scalar, 8> second{  259/11520, -2495/11520, 11691/11520, -9455/11520,
                                                          -9455/11520, 11691/11520, -2495/11520,   259/11520};
        };

        template <>
        struct Coefficients<8, Right>
        {
            static constexpr std::array<Scalar, 8> zero{-5/2048, 49/2048, -245/2048, 1225/2048, 1225/2048, -245/2048, 49/2048, -5/2048};
            static constexpr std::array<Scalar, 8> first{    75/107520, -1029/107520, 8575/107520, -128625/107520,
                                                         128625/107520, -8575/107520, 1029/107520,     -75/107520};
            static constexpr std::array<Scalar, 10> second{  -3229/645120,  37107/645120, -204300/645120, 745108/645120, -574686/645120,
                                                           -574686/645120, 745108/645120, -204300/645120,  37107/645120,   -3229/645120};
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
}
