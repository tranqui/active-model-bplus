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
using Current = std::array<Field, d>;


/// Simulation parameters.


struct StencilParams
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
};

using Stencil = StencilParams;
using Model = ActiveModelBPlusParams;


/// Simulation controller


class Integrator
{
public:
    using HostField = FieldRef;
    using DeviceField = Scalar*;

    // Copy constructors are not safe because GPU device memory will not be copied.
    Integrator(const Integrator&) = delete;
    Integrator& operator=(const Integrator&) = delete;
    // Move constructors are fine though.
    Integrator& operator=(Integrator&&) noexcept = default;
    Integrator(Integrator&&) noexcept;

    Integrator(const HostField& field, Stencil stencil, Model model);

    ~Integrator();

    Stencil get_stencil() const;
    Model get_model() const;
    Field get_field() const;
    Current get_current();

protected:
    Stencil stencil;
    Model model;
    int nrows, ncols;
    size_t pitch_width, mem_size;

    size_t pitch;
    DeviceField field;
    Current current;

    int timestep = 0;
    int timestep_calculated_current = -1;

    void calculate_current();
};
