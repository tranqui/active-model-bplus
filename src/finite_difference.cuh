#pragma once
#include "integrator.h"
#include "parameters.cuh"
#include "finite_difference.h"


namespace kernel
{
    template <typename T>
    __forceinline__ __device__ Scalar square(T&& val)
    {
        return val * val;
    }

    namespace details
    {
        template <std::size_t Order, StaggerGrid Stagger, typename T>
        static __forceinline__ __device__ Scalar zero_x(T&& tile, int i, int j)
        {
            return finite_difference::zero_x<Order, Stagger>(std::forward<T>(tile), i, j);
        }

        template <std::size_t Order, StaggerGrid Stagger, typename T>
        static __forceinline__ __device__ Scalar zero_y(T&& tile, int i, int j)
        {
            return finite_difference::zero_y<Order, Stagger>(std::forward<T>(tile), i, j);
        }

        template <std::size_t Order, StaggerGrid Stagger, typename T>
        static __forceinline__ __device__ Scalar first_x(T&& tile, int i, int j)
        {
            return stencil.dxInv * finite_difference::first_x<Order, Stagger>(std::forward<T>(tile), i, j);
        }

        template <std::size_t Order, StaggerGrid Stagger, typename T>
        static __forceinline__ __device__ Scalar first_y(T&& tile, int i, int j)
        {
            return stencil.dyInv * finite_difference::first_y<Order, Stagger>(std::forward<T>(tile), i, j);
        }

        template <std::size_t Order, StaggerGrid Stagger, typename T>
        static __forceinline__ __device__ Scalar second_x(T&& tile, int i, int j)
        {
            return stencil.dxInv*stencil.dxInv * finite_difference::second_x<Order, Stagger>(std::forward<T>(tile), i, j);
        }

        template <std::size_t Order, StaggerGrid Stagger, typename T>
        static __forceinline__ __device__ Scalar second_y(T&& tile, int i, int j)
        {
            return stencil.dyInv*stencil.dyInv * finite_difference::second_y<Order, Stagger>(std::forward<T>(tile), i, j);
        }

        template <std::size_t Order, StaggerGrid Stagger, typename T>
        static __forceinline__ __device__ Scalar laplacian(T&& tile, int i, int j)
        {
            return second_x<Order, Stagger>(std::forward<T>(tile), i, j)
                 + second_y<Order, Stagger>(std::forward<T>(tile), i, j);
        }

        template <std::size_t Order, StaggerGrid Stagger, typename T>
        static __forceinline__ __device__ Scalar grad_squ(T&& tile, int i, int j)
        {
            return square(first_x<Order, Stagger>(std::forward<T>(tile), i, j))
                 + square(first_y<Order, Stagger>(std::forward<T>(tile), i, j));
        }
    }

    template <StaggerGrid Stagger, typename T>
    static __forceinline__ __device__ Scalar zero_x(T&& tile, int i, int j)
    {
        return details::zero_x<order, Stagger>(std::forward<T>(tile), i, j);
    }

    template <StaggerGrid Stagger, typename T>
    static __forceinline__ __device__ Scalar zero_y(T&& tile, int i, int j)
    {
        return details::zero_y<order, Stagger>(std::forward<T>(tile), i, j);
    }

    template <StaggerGrid Stagger, typename T>
    static __forceinline__ __device__ Scalar first_x(T&& tile, int i, int j)
    {
        return details::first_x<order, Stagger>(std::forward<T>(tile), i, j);
    }

    template <StaggerGrid Stagger, typename T>
    static __forceinline__ __device__ Scalar first_y(T&& tile, int i, int j)
    {
        return details::first_y<order, Stagger>(std::forward<T>(tile), i, j);
    }

    template <StaggerGrid Stagger, typename T>
    static __forceinline__ __device__ Scalar second_x(T&& tile, int i, int j)
    {
        return details::second_x<order, Stagger>(std::forward<T>(tile), i, j);
    }

    template <StaggerGrid Stagger, typename T>
    static __forceinline__ __device__ Scalar second_y(T&& tile, int i, int j)
    {
        return details::second_y<order, Stagger>(std::forward<T>(tile), i, j);
    }

    template <StaggerGrid Stagger, typename T>
    static __forceinline__ __device__ Scalar laplacian(T&& tile, int i, int j)
    {
        return details::laplacian<order, Stagger>(std::forward<T>(tile), i, j);
    }

    template <StaggerGrid Stagger, typename T>
    static __forceinline__ __device__ Scalar grad_squ(T&& tile, int i, int j)
    {
        return details::grad_squ<order, Stagger>(std::forward<T>(tile), i, j);
    }

    /// Aliases to default central grid when Stagger not passed explicitly:

    template <typename T>
    static __forceinline__ __device__ Scalar zero_x(T&& tile, int i, int j)
    {
        return zero_x<Central>(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar zero_y(T&& tile, int i, int j)
    {
        return zero_y<Central>(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar first_x(T&& tile, int i, int j)
    {
        return first_x<Central>(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar first_y(T&& tile, int i, int j)
    {
        return first_y<Central>(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar second_x(T&& tile, int i, int j)
    {
        return second_x<Central>(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar second_y(T&& tile, int i, int j)
    {
        return second_y<Central>(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar laplacian(T&& tile, int i, int j)
    {
        return laplacian<Central>(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar grad_squ(T&& tile, int i, int j)
    {
        return grad_squ<Central>(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar isotropic_first_x(T&& tile, int i, int j)
    {
        return finite_difference::isotropic::first_x(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar isotropic_first_y(T&& tile, int i, int j)
    {
        return finite_difference::isotropic::first_y(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar isotropic_laplacian(T&& tile, int i, int j)
    {
        return finite_difference::isotropic::laplacian(std::forward<T>(tile), i, j);
    }

    template <typename T>
    static __forceinline__ __device__ Scalar isotropic_grad_squ(T&& tile, int i, int j)
    {
        return square(isotropic_first_y(std::forward<T>(tile), i, j))
             + square(isotropic_first_x(std::forward<T>(tile), i, j));
    }
}
