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

    // Implementation is on a 2d grid with periodic boundary conditions.
    // GPU divided into an (tile_rows x tile_cols) tile (blocks) with
    // a CUDA thread for each tile sharing this memory. Varying the tile size
    // will potentially improve performance on different hardware - I found
    // 16x16 was close to optimum on my machine for simulations on a 1024x1024 grid.
    static constexpr int tile_rows = 16;
    static constexpr int tile_cols = 16;
    // We need ghost points for each tile so we can evaluate derivatives
    // (specifically the Laplacian for diffusion) at the tile borders.
    static constexpr int num_ghost = 1; // <- minimum for second-order finite-difference stencil.

    // Stencil parameters - 2d space (x, y), and time t.
    __constant__ DeviceStencilParams stencil;
    __constant__ int nrows, ncols;        // number of points in spatial grid
    __constant__ Model model;

    __device__ inline Scalar bulk_chemical_potential(Scalar field)
    {
        return model.a * field + model.b * field * field + model.c * field * field * field;
    }

    __device__ inline Scalar deriv_bulk_chemical_potential(Scalar field)
    {
        return model.a + 2 * model.b * field + 3 * model.c * field * field;
    }

    /// Kernel to determine current.

    __global__ void calculate_current(DeviceField field, DeviceCurrent current)
    {
        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        // Local indices.
        const int i = threadIdx.y + num_ghost;
        const int j = threadIdx.x + num_ghost;

        // Load tile into shared memory.

        __shared__ Scalar tile[tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];
        tile[i][j] = field[index];

        // Fill in ghost points.

        if (threadIdx.y < num_ghost)
        {
            tile[i - num_ghost][j] = field[col + ((row - num_ghost + nrows) % nrows) * ncols];
            tile[i + tile_rows][j] = field[col + ((row + tile_rows) % nrows) * ncols];
        }

        if (threadIdx.x < num_ghost)
        {
            tile[i][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + row * ncols];
            tile[i][j + tile_cols] = field[(col + tile_cols) % ncols         + row * ncols];
        }

        if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
        {
            tile[i - num_ghost][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
            tile[i - num_ghost][j + tile_cols] = field[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
            tile[i + tile_rows][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
            tile[i + tile_rows][j + tile_cols] = field[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];
        }

        __syncthreads();

        // Scalar grad_field[d];
        // grad_field[0] = 0.5 * (tile[i+1][j  ] - tile[i-1][j  ]) * stencil.dyInv;
        // grad_field[1] = 0.5 * (tile[i  ][j+1] - tile[i  ][j-1]) * stencil.dxInv;

        // Scalar dmu = deriv_bulk_chemical_potential(tile[i][j]);
        // for (int c = 0; c < d; ++c)
        // {
        //     current[c][index] = dmu * grad_field[c];
        // }

        Scalar grad_field[d];
        Scalar mu1, mu2;

        mu2 = bulk_chemical_potential(tile[i+1][j]);
        mu1 = bulk_chemical_potential(tile[i-1][j]);
        current[0][index] = 0.5 * (mu2 - mu1) * stencil.dyInv;

        mu2 = bulk_chemical_potential(tile[i][j+1]);
        mu1 = bulk_chemical_potential(tile[i][j-1]);
        current[1][index] = 0.5 * (mu2 - mu1) * stencil.dxInv;
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
    Field empty = Field::Zero(nrows, ncols);
    for (int c = 0; c < d; ++c)
    {
        cudaMallocPitch(&current[c], &current_pitch[c], pitch_width, nrows);
        cudaMemcpy(current[c], empty.data(), mem_size, cudaMemcpyHostToDevice);
    }

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

HostField Integrator::get_field() const
{
    HostField out(nrows, ncols);
    cudaMemcpy(out.data(), field, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

HostCurrent Integrator::get_current()
{
    calculate_current();

    HostCurrent out = repeat_array<HostField, d>(nrows, ncols);
    for (int c = 0; c < d; ++c)
        cudaMemcpy(out[c].data(), current[c], mem_size, cudaMemcpyDeviceToHost);
    return out;
}

void Integrator::set_device_parameters()
{
    DeviceStencilParams device_stencil(stencil);
    cudaMemcpyToSymbol(kernel::stencil, &device_stencil, sizeof(DeviceStencilParams));
    cudaMemcpyToSymbol(kernel::nrows, &nrows, sizeof(int));
    cudaMemcpyToSymbol(kernel::ncols, &ncols, sizeof(int));
    cudaMemcpyToSymbol(kernel::model, &model, sizeof(Model));
}

void Integrator::calculate_current()
{
    if (timestep_calculated_current == timestep) return;

    set_device_parameters();
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);
    kernel::calculate_current<<<grid_size, block_dim>>>(field, current);
    cudaDeviceSynchronize();
    kernel::throw_errors();

    timestep_calculated_current = timestep;
}