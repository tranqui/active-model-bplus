#include "integrator.cuh"

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

Integrator::Integrator(const HostField& initial_field,
                       Scalar dt, Scalar dx, Scalar dy)
    : dt(dt), dx(dx), dy(dy),
    nrows(initial_field.rows()),
    ncols(initial_field.cols()),
    pitch_width(initial_field.cols() * sizeof(Scalar)),
    mem_size(initial_field.rows() * initial_field.cols() * sizeof(Scalar))
{
    // Initialise device memory.
    cudaMallocPitch(&field, &pitch, pitch_width, nrows);
    cudaMemcpy(field, initial_field.data(), mem_size, cudaMemcpyHostToDevice);
 
    kernel::throw_errors();
}

Integrator::Integrator(Integrator&& other) noexcept
    : dt(other.dt), dx(other.dx), dy(other.dy),
      nrows(other.nrows), ncols(other.ncols),
      pitch_width(other.pitch_width), mem_size(other.mem_size),
      pitch(std::move(other.pitch)),
      field(std::move(other.field))
{
    kernel::throw_errors();
}

Integrator::~Integrator()
{
    cudaFree(field);
}

Field Integrator::get_field() const
{
    Field out(nrows, ncols);
    cudaMemcpy(out.data(), field, mem_size, cudaMemcpyDeviceToHost);
    return out;
}
