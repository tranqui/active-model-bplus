#pragma once
#include "math_primitives.h"


/// Compile-time parameters affecting algorithm implementation.

static constexpr int order = 2; // order of error in finite-difference approximations

namespace kernel
{
    // Implementation is on a 2d grid with periodic boundary conditions.
    // GPU divided into an (tile_rows x tile_cols) tile (blocks) with
    // a CUDA thread for each tile sharing this memory. Varying the tile size
    // will potentially improve performance on different hardware - I found
    // 16x16 was close to optimum on my machine for simulations on a 1024x1024 grid.
    static constexpr int tile_rows = 16;
    static constexpr int tile_cols = 16;
    // We need ghost points for each tile so we can evaluate derivatives at tile borders.
    static constexpr int num_ghost = 1 + order / 2; // <- minimum for fourth derivatives
    // One less support point needed in integrator because it involves one fewer derivative.
    static constexpr int num_ghost_integrator = num_ghost - 1;
}


/// Run-time parameters to specific simulations.

struct HostStencilParams
{
    Scalar dt, dx, dy;

    inline auto as_tuple()
    {
        return std::tie(dt, dx, dy);
    }

    inline auto as_tuple() const
    {
        return std::tie(dt, dx, dy);
    }

    inline friend bool operator==(const HostStencilParams& a, const HostStencilParams& b)
    {
        return a.dt == b.dt and a.dx == b.dx and a.dy == b.dy;
    }
};

struct DeviceStencilParams
{
    Scalar dt, dxInv, dyInv;

    DeviceStencilParams() noexcept = default;
    inline DeviceStencilParams(const HostStencilParams& host)
      : dt(host.dt), dxInv(1/host.dx), dyInv(1/host.dy) { }
};

struct ActiveModelBPlusParams
{
    Scalar a, b, c;
    Scalar kappa;
    Scalar lambda;
    Scalar zeta;
    Scalar temperature;

    inline auto as_tuple()
    {
        return std::tie(a, b, c, kappa, lambda, zeta, temperature);
    }

    inline auto as_tuple() const
    {
        return std::tie(a, b, c, kappa, lambda, zeta, temperature);
    }

    inline friend bool operator==(const ActiveModelBPlusParams& a, const ActiveModelBPlusParams& b)
    {
        return a.a == b.a and a.b == b.b and a.c == b.c and
            a.kappa == b.kappa and a.lambda == b.lambda and a.zeta == b.zeta
            and a.temperature == b.temperature;
    }
};

using Stencil = HostStencilParams;
using Model = ActiveModelBPlusParams;
