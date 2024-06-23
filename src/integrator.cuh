#pragma once
#include <Eigen/Eigen>


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

using Scalar = double;
using Field = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FieldRef = Eigen::Ref<const Field>;


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

    Integrator(const HostField& field, Scalar dt, Scalar dx, Scalar dy);

    ~Integrator();

    Field get_field() const;

protected:
    Scalar dt, dx, dy;
    int nrows, ncols;
    size_t pitch_width, mem_size;

    size_t pitch;
    DeviceField field;
};
