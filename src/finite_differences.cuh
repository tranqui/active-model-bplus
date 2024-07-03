#pragma once
#include "integrator.h"
#include "parameters.cuh"


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

        /// Derivatives must be specialised for the Order of the numerical approximation.
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
    }

    using CentralDifference = details::CentralDerivative<order>;
}
