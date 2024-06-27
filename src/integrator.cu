#include "integrator.cuh"
#include "foreach.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <string>
#include <cmath>
#include <stdexcept>


/// Main execution on GPU device.

namespace kernel
{
    // Check CUDA for errors after GPU execution and throw them.
    __host__ void throw_errors()
    {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::string message = "CUDA Kernel Error: "
                                + std::string(cudaGetErrorString(error));
            throw CudaError(message);
        }
    }
}


/// Host device definitions.

Integrator::Integrator(const HostFieldRef& initial_field,
                       Stencil stencil, Model model)
    : stencil(stencil), model(model),
    nrows(initial_field.rows()),
    ncols(initial_field.cols()),
    pitch_width(initial_field.cols() * sizeof(Scalar)),
    mem_size(initial_field.rows() * initial_field.cols() * sizeof(Scalar))
{
    // Initialise device memory.
    cudaMallocPitch(&field, &field_pitch, pitch_width, nrows);
    cudaMemcpy(field, initial_field.data(), mem_size, cudaMemcpyHostToDevice);
    for (int c = 0; c < d; ++c)
        cudaMallocPitch(&current[c], &current_pitch[c], pitch_width, nrows);

    kernel::throw_errors();
}

Integrator::Integrator(Integrator&& other) noexcept
    : stencil(other.stencil), model(other.model),
      nrows(other.nrows), ncols(other.ncols),
      pitch_width(other.pitch_width), mem_size(other.mem_size),
      field_pitch(std::move(other.field_pitch)), field(std::move(other.field)),
      current_pitch(std::move(other.current_pitch)), current(std::move(other.current)),
      timestep(other.timestep), timestep_calculated_current(other.timestep_calculated_current)
{
    kernel::throw_errors();
}

Integrator::~Integrator()
{
    cudaFree(field);
    for (int c = 0; c < d; ++c) cudaFree(current[c]);
}

Stencil Integrator::get_stencil() const
{
    return stencil;
}

Model Integrator::get_model() const
{
    return model;
}

Integrator::HostField Integrator::get_field() const
{
    HostField out(nrows, ncols);
    cudaMemcpy(out.data(), field, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

inline Scalar bulk_chemical_potential(Scalar field, const Model& model)
{
    return model.a * field
         + model.b * field * field
         + model.c * field * field * field;
}

Integrator::HostCurrent Integrator::get_current()
{
    if (timestep > timestep_calculated_current)
        calculate_current();

    HostCurrent out = repeat_array<HostField, d>(nrows, ncols);
    for (int c = 0; c < d; ++c)
        cudaMemcpy(out[c].data(), current[c], mem_size, cudaMemcpyDeviceToHost);
    return out;
}

void Integrator::calculate_current()
{
    if (timestep_calculated_current == timestep) return;

    Field field = get_field();

    Field mu = Field(nrows, ncols);
    for (int i = 0; i < ncols; ++i)
        for (int j = 0; j < nrows; ++j)
            mu(i, j) = bulk_chemical_potential(field(i, j), model);

    Current host_current{Field(nrows, ncols), Field(nrows, ncols)};
    for (int i = 0; i < nrows; ++i)
    {
        // Nearest neighbours in y-direction w/ periodic boundaries:
        int ip{i+1}, im{i-1};
        if (im < 0) im += nrows;
        if (ip >= nrows) ip -= nrows;

        for (int j = 0; j < ncols; ++j)
        {
            // Nearest neighbours in x-direction w/ periodic boundaries:
            int jp{j+1}, jm{j-1};
            if (jm < 0) jm += ncols;
            if (jp >= ncols) jp -= ncols;

            host_current[0](i, j) = 0.5 * (mu(ip, j ) - mu(im, j )) / stencil.dy;
            host_current[1](i, j) = 0.5 * (mu(i , jp) - mu(i , jm)) / stencil.dx;
        }
    }

    for (int c = 0; c < d; ++c)
        cudaMemcpy(current[c], host_current[c].data(), mem_size, cudaMemcpyHostToDevice);

    timestep_calculated_current = timestep;
}