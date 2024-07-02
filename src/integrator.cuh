#pragma once
#include <Eigen/Eigen>

static constexpr int d = 2; // dimensionality


namespace kernel
{
    // Generic python-safe exception to contain errors on GPU execution.
    // Note that the actual catching and throwing of errors has to handle on the host if
    // the CUDA kernel sets the error flag - this is handled in kernel::throw_errors().
    class CudaError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    // Check CUDA for errors after GPU execution and throw them.
    void throw_errors();
}


/// Data types for system state.

using Scalar = double;
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


/// Simulation parameters.


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


/// Simulation controller


class Integrator
{
public:
    // Copy constructors are not safe because GPU device memory will not be copied.
    Integrator(const Integrator&) = delete;
    Integrator& operator=(const Integrator&) = delete;
    // Move constructors are fine though.
    Integrator& operator=(Integrator&&) noexcept = default;
    Integrator(Integrator&&) noexcept;

    Integrator(const HostFieldRef& field, Stencil stencil, Model model);

    ~Integrator();

    Stencil get_stencil() const;
    Model get_model() const;
    HostField get_field() const;
    HostCurrent get_current();

    inline int get_timestep() const
    {
        return timestep;
    }

    inline Scalar get_time() const
    {
        return timestep * stencil.dt;
    }

    void run(int nsteps);

protected:
    Stencil stencil;
    Model model;
    int nrows, ncols;
    size_t pitch_width, mem_size;

    size_t field_pitch;
    DeviceField field;
    std::array<size_t, d> current_pitch;
    DeviceCurrent current;

    int timestep = 0;
    int timestep_calculated_current = -1;

    void set_device_parameters();
    void calculate_current();
};
