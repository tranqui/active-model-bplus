#pragma once
#include "parameters.h"


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
