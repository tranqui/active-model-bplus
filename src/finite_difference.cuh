#pragma once
#include "integrator.h"
#include "parameters.cuh"
#include "finite_difference.h"


namespace kernel
{
    namespace details
    {
        /// Helper functions.

        template <typename T>
        __forceinline__ __device__ Scalar square(T&& val)
        {
            return val * val;
        }


        /** Common interface for derivatives.
         * @tparam Derived: the specific implementation of the derivatives in each
         *           dimension. First-order derivatives must defined via
         *           Derived::first_x and Derived::first_y, whereas second-order
         *           derivatives are defined with Derived::second_x and Derived::second_y.
         */
        template <typename Derived> struct BaseDerivative
        {
            template <typename T>
            static __forceinline__ __device__ Scalar laplacian(T&& tile, int i, int j)
            {
                return Derived::second_x(tile, i, j) + Derived::second_y(tile, i, j);
            }

            template <typename T>
            static __forceinline__ __device__ Scalar grad_squ(T&& tile, int i, int j)
            {
                return square(Derived::first_x(tile, i, j))
                     + square(Derived::first_y(tile, i, j));
            }
        };

        /**
         * Base for central derivative stencils.
         * 
         * @tparam Order: Derivatives must be implemented for the specific
         *                Order of the numerical approximation, which is done below
         *                by complete class specialisation.
         */
        template <std::size_t Order>
        struct CentralDerivative : public BaseDerivative<CentralDerivative<Order>>
        {
            template <typename T>
            static __forceinline__ __device__ Scalar first_x(T&& tile, int i, int j)
            {
                return stencil.dxInv * finite_difference::first_x<Order>(std::forward<T>(tile), i, j);
            }

            template <typename T>
            static __forceinline__ __device__ Scalar first_y(T&& tile, int i, int j)
            {
                return stencil.dyInv * finite_difference::first_y<Order>(std::forward<T>(tile), i, j);
            }

            template <typename T>
            static __forceinline__ __device__ Scalar second_x(T&& tile, int i, int j)
            {
                return stencil.dxInv*stencil.dxInv * finite_difference::second_x<Order>(std::forward<T>(tile), i, j);
            }

            template <typename T>
            static __forceinline__ __device__ Scalar second_y(T&& tile, int i, int j)
            {
                return stencil.dyInv*stencil.dyInv * finite_difference::second_y<Order>(std::forward<T>(tile), i, j);
            }
        };

        /**
         * Base for derivatives on a staggered grid.
         * 
         * Elements of derivatives are in between those of the source field.
         *   The staggered grid is sometimes written with half-integer
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
        template <std::size_t Order, StaggerGrid Stagger>
        struct StaggeredDerivative
            : public BaseDerivative<StaggeredDerivative<Order, Stagger>>
        {
            template <typename T>
            static __forceinline__ __device__ Scalar first_x(T&& tile, int i, int j)
            {
                return stencil.dxInv * finite_difference::first_x<Order, Stagger>(std::forward<T>(tile), i, j);
            }

            template <typename T>
            static __forceinline__ __device__ Scalar first_y(T&& tile, int i, int j)
            {
                return stencil.dyInv * finite_difference::first_y<Order, Stagger>(std::forward<T>(tile), i, j);
            }

            template <typename T>
            static __forceinline__ __device__ Scalar second_x(T&& tile, int i, int j)
            {
                return stencil.dxInv * stencil.dxInv *
                       finite_difference::first_x<Order, Stagger>(std::forward<T>(tile), i, j);
            }

            template <typename T>
            static __forceinline__ __device__ Scalar second_y(T&& tile, int i, int j)
            {
                return stencil.dyInv * stencil.dyInv *
                       finite_difference::first_y<Order, Stagger>(std::forward<T>(tile), i, j);
            }
        };
    }

    using CentralDifference = details::CentralDerivative<order>;
    template <StaggerGrid Stagger>
    using StaggeredDifference = details::StaggeredDerivative<order, Stagger>;
}
