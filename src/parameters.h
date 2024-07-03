#pragma once
#include <Eigen/Eigen>


/// Basic constants and data types.

static constexpr int d = 2;      // number of spatial dimensions
using Scalar = double;           // type for numeric data
static constexpr int order = 2;  // order of finite-difference calculations

using Field = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FieldRef = Eigen::Ref<const Field>;

using HostField = Field;
using HostFieldRef = FieldRef;
using DeviceField = Scalar*;

using Gradient = std::array<Field, d>;
using HostGradient = Gradient;
using DeviceGradient = std::array<Scalar*, d>;

using Current = Gradient;
using HostCurrent = HostGradient;
using DeviceCurrent = DeviceGradient;

/// Grid parameters

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
}

// Direction of index offset when derivatives are taken on staggered grids.
// See StaggeredDerivative in finite_differences.cuh for more info.
enum StaggeredGridDirection { Left=-1, Right=1 };


/// Parameters to specific simulations.

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

    inline auto as_tuple()
    {
        return std::tie(a, b, c, kappa, lambda, zeta);
    }

    inline auto as_tuple() const
    {
        return std::tie(a, b, c, kappa, lambda, zeta);
    }

    inline friend bool operator==(const ActiveModelBPlusParams& a, const ActiveModelBPlusParams& b)
    {
        return a.a == b.a and a.b == b.b and a.c == b.c and
            a.kappa == b.kappa and a.lambda == b.lambda and a.zeta == b.zeta;
    }
};

using Stencil = HostStencilParams;
using Model = ActiveModelBPlusParams;

