#pragma once
#include "parameters.h"

namespace finite_difference
{
    namespace central_stencil
    {
        /**
         * Fill in antisymmetric (for odd derivatives) finite difference
         *   coefficients from partially specified coefficients.
         * 
         * Numerical optimisations are possible using just the partial
         *   coefficients, so this convenience function ensures consistency
         *   between the partial and complete sets of coefficients.
         * 
         * Example (in pseudocode) for sixth-order first derivative:
         *   >>> from_partial({3/4, -3/20, 1/60})
         *   {-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60}
         */
        template <std::size_t N>
        constexpr auto from_partial(const std::array<Scalar, N>& partial)
        {
            constexpr std::size_t stencil_size = 1 + 2*N;
            std::array<Scalar, stencil_size> coefficients{};

            for (std::size_t i = 0; i < N; ++i)
            {
                coefficients[i] = -partial[i];
                coefficients[i + N + 1] = partial[i];
            }
            coefficients[N] = 0;

            return coefficients;
        }

        /// Basic types.

        template <int Order>
        struct FirstDerivative
        {
            static_assert(Order > 0);
            static_assert((Order % 2) == 0);
        };

        template <int Order>
        struct SecondDerivative
        {
            static_assert((Order % 2) == 0);
        };

        /// Specific coefficients.

        template <>
        struct FirstDerivative<2>
        {
            static constexpr std::array<Scalar, 1> partial_coefficients{0.5};
            static constexpr auto coefficients = from_partial(partial_coefficients);
        };

        template <>
        struct FirstDerivative<4>
        {
            static constexpr std::array<Scalar, 2> partial_coefficients{2/3, -1/12};
            static constexpr auto coefficients = from_partial(partial_coefficients);
        };

        template <>
        struct FirstDerivative<6>
        {
            static constexpr std::array<Scalar, 3> partial_coefficients{3/4, -3/20, 1/60};
            static constexpr auto coefficients = from_partial(partial_coefficients);
        };

        template <>
        struct FirstDerivative<8>
        {
            static constexpr std::array<Scalar, 4> partial_coefficients{4/5, -1/5, 4/105, -1/280};
            static constexpr auto coefficients = from_partial(partial_coefficients);
        };

        template <>
        struct SecondDerivative<2>
        {
            static constexpr std::array<Scalar, 3> coefficients{1, -2, 1};
        };

        template <>
        struct SecondDerivative<4>
        {
            static constexpr std::array<Scalar, 5> coefficients{-1/12, 4/3, -5/2, 4/3, -1/12};
        };

        template <>
        struct SecondDerivative<6>
        {
            static constexpr std::array<Scalar, 7> coefficients{1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90};
        };

        template <>
        struct SecondDerivative<8>
        {
            static constexpr std::array<Scalar, 9> coefficients{-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560};
        };
    }

    using FirstDerivative = central_stencil::FirstDerivative<order>;
    using SecondDerivative = central_stencil::SecondDerivative<order>;

    static constexpr std::size_t stencil_size = 1 + order;

    inline Scalar first(const std::array<Scalar, stencil_size>& stencil)
    {
        Scalar result{0};
        for (std::size_t i = 0; i < stencil_size; ++i)
            result += FirstDerivative::coefficients[i] * stencil[i];
        return result;
    }

    inline Scalar second(const std::array<Scalar, stencil_size>& stencil)
    {
        Scalar result{0};
        for (std::size_t i = 0; i < stencil_size; ++i)
            result += SecondDerivative::coefficients[i] * stencil[i];
        return result;
    }
}
