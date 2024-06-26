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
                       Stencil stencil, Model model)
    : stencil(stencil), model(model),
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
    : stencil(other.stencil), model(other.model),
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

Stencil Integrator::get_stencil() const
{
    return stencil;
}

Model Integrator::get_model() const
{
    return model;
}

Field Integrator::get_field() const
{
    Field out(nrows, ncols);
    cudaMemcpy(out.data(), field, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

inline Scalar bulk_chemical_potential(Scalar field, const Model& model)
{
    return model.a * field
         + model.b * field * field
         + model.c * field * field * field;
}

Current Integrator::get_current() const
{
    // Field field = get_field();

    // Field mu = Field(nrows, ncols);
    // for (int i = 0; i < ncols; ++i)
    //     for (int j = 0; j < nrows; ++j)
    //         mu(i, j) = bulk_chemical_potential(field(i, j), model);

    // Current expected{Field(nrows, ncols), Field(nrows, ncols)};
    // for (int i = 1; i < nrows-1; ++i)
    //     for (int j = 1; j < ncols-1; ++j)
    //     {
    //         expected[0](i, j) = 0.5 * (mu(i+1, j  ) - mu(i-1, j  )) / stencil.dy;
    //         expected[1](i, j) = 0.5 * (mu(i  , j+1) - mu(i  , j-1)) / stencil.dx;
    //     }

    // return expected;
    return Current{Field(nrows, ncols), Field(nrows, ncols)};
}