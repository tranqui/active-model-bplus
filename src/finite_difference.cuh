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
        __device__ inline Scalar square(T&& val)
        {
            return val * val;
        }


        /** Common interface for derivatives.
         * @tparam Derived: the specific implementation of the derivatives in each
         *           dimension. First-order derivatives must defined via
         *           Derived::grad_x and Derived::grad_y, whereas second-order
         *           derivatives are defined with Derived::lap_x and Derived::lap_y.
         */
        template <typename Derived> struct BaseDerivative
        {
            template <typename T>
            static __device__ inline Scalar laplacian(T&& tile, int i, int j)
            {
                return Derived::lap_x(tile, i, j) + Derived::lap_y(tile, i, j);
            }

            template <typename T>
            static __device__ inline Scalar grad_squ(T&& tile, int i, int j)
            {
                return square(Derived::grad_x(tile, i, j))
                     + square(Derived::grad_y(tile, i, j));
            }
        };

        /**
         * Base for central derivative stencils.
         * 
         * @tparam Order: Derivatives must be implemented for the specific
         *                Order of the numerical approximation, which is done below
         *                by complete class specialisation.
         */
        template <int Order>
        struct CentralDerivative : public BaseDerivative<CentralDerivative<Order>> { };


        /// Second-order central finite-difference derivatives.
        template <>
        struct CentralDerivative<2> : public BaseDerivative<CentralDerivative<2>>
        {
            template <typename T>
            static __device__ inline Scalar grad_x(T&& tile, int i, int j)
            {
                return 0.5 * stencil.dxInv * (tile[i][j+1] - tile[i][j-1]);
            }

            template <typename T>
            static __device__ inline Scalar grad_y(T&& tile, int i, int j)
            {
                return 0.5 * stencil.dyInv * (tile[i+1][j] - tile[i-1][j]);
            }

            template <typename T>
            static __device__ inline Scalar lap_x(T&& tile, int i, int j)
            {
                return stencil.dxInv*stencil.dxInv * (tile[i][j+1] - 2*tile[i][j] + tile[i][j-1]);
            }

            template <typename T>
            static __device__ inline Scalar lap_y(T&& tile, int i, int j)
            {
                return stencil.dyInv*stencil.dyInv * (tile[i+1][j] - 2*tile[i][j] + tile[i-1][j]);
            }
        };


        /// Fourth-order central finite-difference derivatives.
        template <>
        struct CentralDerivative<4> : public BaseDerivative<CentralDerivative<4>>
        {
            template <typename T>
            static __device__ inline Scalar grad_x(T&& tile, int i, int j)
            {
                return stencil.dxInv * (-tile[i][j+2] + 8*tile[i][j+1] - 8*tile[i][j-1] + tile[i][j-2]) / 12;
            }

            template <typename T>
            static __device__ inline Scalar grad_y(T&& tile, int i, int j)
            {
                return stencil.dyInv * (-tile[i+2][j] + 8*tile[i+1][j] - 8*tile[i-1][j] + tile[i-2][j]) / 12;
            }

            template <typename T>
            static __device__ inline Scalar lap_x(T&& tile, int i, int j)
            {
                return stencil.dxInv*stencil.dxInv * (-tile[i][j+2] + 16*tile[i][j+1] - 30*tile[i][j] + 16*tile[i][j-1] - tile[i][j-2]) / 12;
            }

            template <typename T>
            static __device__ inline Scalar lap_y(T&& tile, int i, int j)
            {
                return stencil.dyInv*stencil.dyInv * (-tile[i+2][j] + 16*tile[i+1][j] - 30*tile[i][j] + 16*tile[i-1][j] - tile[i-2][j]) / 12;
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
         * @tparam Stagger: direction of staggering of grid from integer indices.
         *           If Stagger=Left : grad[i] -> grad(i - 1/2)
         *           If Stagger=Right: grad[i] -> grad(i + 1/2)
         * @tparam Order: Derivatives must be implemented for the specific
         *                Order of the numerical approximation, which is done below
         *                by complete class specialisation.
         */
        template <StaggerGrid Stagger, int Order>
        struct StaggeredDerivative
            : public BaseDerivative<StaggeredDerivative<Stagger, Order>>
        { };

        /// Second-order staggered finite-difference derivatives.
        template <StaggerGrid Stagger>
        struct StaggeredDerivative<Stagger, 2>
            : public BaseDerivative<StaggeredDerivative<Stagger, 2>>
        {
            template <typename T>
            static __device__ inline Scalar grad_x(T&& tile, int i, int j)
            {
                if constexpr (Stagger == Right)
                    return stencil.dxInv * (tile[i][j+1] - tile[i][j]);
                else
                    return stencil.dxInv * (tile[i][j] - tile[i][j-1]);
            }

            template <typename T>
            static __device__ inline Scalar grad_y(T&& tile, int i, int j)
            {
                if constexpr (Stagger == Right)
                    return stencil.dyInv * (tile[i+1][j] - tile[i][j]);
                else
                    return stencil.dyInv * (tile[i][j] - tile[i-1][j]);
            }

            template <typename T>
            static __device__ inline Scalar lap_x(T&& tile, int i, int j)
            {
                if constexpr (Stagger == Right)
                    return 0.25 * stencil.dxInv * stencil.dxInv * (
                        (tile[i][j+2] - tile[i][ j ] + tile[i+1][j+2] - tile[i+1][ j ])
                      - (tile[i][j+1] - tile[i][j-1] + tile[i+1][j+1] - tile[i+1][j-1]) );
                else
                    return 0.25 * stencil.dxInv * stencil.dxInv * (
                        (tile[i][j+1] - tile[i][j-1] + tile[i-1][j+1] - tile[i-1][j-1])
                      - (tile[i][ j ] - tile[i][j-2] + tile[i-1][ j ] - tile[i-1][j-2]) );
            }

            template <typename T>
            static __device__ inline Scalar lap_y(T&& tile, int i, int j)
            {
                if constexpr (Stagger == Right)
                    return 0.25 * stencil.dyInv * stencil.dyInv * (
                        (tile[i+2][j] - tile[ i ][j] + tile[i+2][j+1] - tile[ i ][j+1])
                      - (tile[i+1][j] - tile[i-1][j] + tile[i+1][j+1] - tile[i-1][j+1]) );
                else
                    return 0.25 * stencil.dyInv * stencil.dyInv * (
                        (tile[i+1][j] - tile[i-1][j] + tile[i+1][j-1] - tile[i-1][j-1])
                      - (tile[ i ][j] - tile[i-2][j] + tile[ i ][j-1] - tile[i-2][j-1]) );
            }
        };
    }

    using CentralDifference = details::CentralDerivative<order>;
    template <StaggerGrid Stagger>
    using StaggeredDifference = details::StaggeredDerivative<Stagger, order>;
}
