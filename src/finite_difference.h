#pragma once
#include "math_primitives.h"


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
            // static constexpr std::array<Scalar, 4> zero{};
            // static constexpr std::array<Scalar, 5> first{};
            // static constexpr std::array<Scalar, 5> second{-1/12, 4/3, -5/2, 4/3, -1/12};
        };

        template <>
        struct Coefficients<6, Right>
        {
            // static constexpr std::array<Scalar, 6> zero{};
            // static constexpr std::array<Scalar, 7> first{};
            // static constexpr std::array<Scalar, 7> second{1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90};
        };

        template <>
        struct Coefficients<8, Right>
        {
            // static constexpr std::array<Scalar, 8> zero{};
            // static constexpr std::array<Scalar, 9> first{};
            // static constexpr std::array<Scalar, 9> second{-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560};
        };

        // Coefficients for left stagger are the same as for right stagger.
        // template <std::size_t Order>
        // struct Coefficients<Order, Left> : public Coefficients<Order, Right> { };

        template <Derivative D, std::size_t Order, StaggerGrid Stagger>
        struct StencilBase
        {
            static constexpr auto get_coefficients()
            {
                static_assert(D == Zero or D == First or D == Second);
                if constexpr (D == Zero)
                    return Coefficients<Order, Stagger>::zero;
                else if constexpr (D == First)
                    return Coefficients<Order, Stagger>::first;
                else return Coefficients<Order, Stagger>::second;
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
            static constexpr int start = -size/2;
        };

        template <Derivative D, std::size_t Order>
        struct Stencil<D, Order, Left> : public StencilBase<D, Order, Left>
        {
            using StencilBase<D, Order, Left>::size;
            static constexpr int start = -size/2 - 1;
        };

        template <Derivative D, std::size_t Order>
        struct Stencil<D, Order, Right> : public StencilBase<D, Order, Right>
        {
            using StencilBase<D, Order, Right>::size;
            static constexpr int start = -size/2 + 1;
        };
    }

    // Apply central derivatives on a 1d set of support points.

    template <Derivative D, std::size_t StencilSize>
    inline Scalar apply(const std::array<Scalar, StencilSize>& data)
    {
        static_assert(StencilSize % 2 == 1);
        static constexpr int Order = StencilSize - 1;
        static_assert(Order >= 1);
        using stencil = details::Stencil<D, Order, Central>;
        static_assert(StencilSize == stencil::size);

        Scalar result{0};
        for (std::size_t k = 0; k < StencilSize; ++k)
            result += stencil::coefficients[k] * data[k];
        return result;
    }

    template <std::size_t StencilSize>
    inline Scalar zero(const std::array<Scalar, StencilSize>& data)
    {
        return apply<Zero>(data);
    }

    template <std::size_t StencilSize>
    inline Scalar first(const std::array<Scalar, StencilSize>& data)
    {
        return apply<First>(data);
    }

    template <std::size_t StencilSize>
    inline Scalar second(const std::array<Scalar, StencilSize>& data)
    {
        return apply<Second>(data);
    }

    // Apply coefficients for a particular derivative to the stencil at position (i, j).

    template <Derivative D, std::size_t Order, StaggerGrid Stagger, typename T>
    inline Scalar apply_x(T&& data, int i, int j)
    {
        using stencil = details::Stencil<D, Order, Stagger>;
        Scalar result{0};
        for (std::size_t k = 0; k < stencil::size; ++k)
            result += stencil::coefficients[k] * (data[i][j + k + stencil::start]);
        return result;
    }

    template <Derivative D, std::size_t Order, StaggerGrid Stagger, typename T>
    inline Scalar apply_y(T&& data, int i, int j)
    {
        using stencil = details::Stencil<D, Order, Stagger>;
        Scalar result{0};
        for (std::size_t k = 0; k < stencil::size; ++k)
            result += stencil::coefficients[k] * (data[i + k + stencil::start][j]);
        return result;
    }

    // Aliases to apply coefficients for derivatives of different orders.

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    inline Scalar zero_x(T&& data, int i, int j)
    {
        return apply_x<Zero, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    inline Scalar zero_y(T&& data, int i, int j)
    {
        return apply_y<Zero, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    inline Scalar first_x(T&& data, int i, int j)
    {
        return apply_x<First, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    inline Scalar first_y(T&& data, int i, int j)
    {
        return apply_y<First, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    inline Scalar second_x(T&& data, int i, int j)
    {
        return apply_x<Second, Order, Stagger>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, StaggerGrid Stagger, typename T>
    inline Scalar second_y(T&& data, int i, int j)
    {
        return apply_y<Second, Order, Stagger>(std::forward<T>(data), i, j);
    }

    // Aliases to default to central derivatives if no staggering of grid selected.

    template <std::size_t Order, typename T>
    inline Scalar zero_x(T&& data, int i, int j)
    {
        return zero_x<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    inline Scalar zero_y(T&& data, int i, int j)
    {
        return zero_y<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    inline Scalar first_x(T&& data, int i, int j)
    {
        return first_x<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    inline Scalar first_y(T&& data, int i, int j)
    {
        return first_y<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    inline Scalar second_x(T&& data, int i, int j)
    {
        return second_x<Order, Central>(std::forward<T>(data), i, j);
    }

    template <std::size_t Order, typename T>
    inline Scalar second_y(T&& data, int i, int j)
    {
        return second_y<Order, Central>(std::forward<T>(data), i, j);
    }
}
