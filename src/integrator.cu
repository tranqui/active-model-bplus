#include "integrator.h"
#include "parameters.cuh"
#include "foreach.h"
#include "finite_differences.cuh"

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

    /// Physical calculations

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
        __shared__ Scalar mu[tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];
        tile[i][j] = field[index];
        mu[i][j] = bulk_chemical_potential(tile[i][j]);

        // Fill in ghost points.

        if (threadIdx.y < num_ghost)
        {
            tile[i - num_ghost][j] = field[col + ((row - num_ghost + nrows) % nrows) * ncols];
            tile[i + tile_rows][j] = field[col + ((row + tile_rows) % nrows) * ncols];

            mu[i - num_ghost][j] = bulk_chemical_potential(tile[i - num_ghost][j]);
            mu[i + tile_rows][j] = bulk_chemical_potential(tile[i + tile_rows][j]);
        }

        if (threadIdx.x < num_ghost)
        {
            tile[i][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + row * ncols];
            tile[i][j + tile_cols] = field[(col + tile_cols) % ncols         + row * ncols];

            mu[i][j - num_ghost] = bulk_chemical_potential(tile[i][j - num_ghost]);
            mu[i][j + tile_cols] = bulk_chemical_potential(tile[i][j + tile_cols]);
        }

        if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
        {
            tile[i - num_ghost][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
            tile[i - num_ghost][j + tile_cols] = field[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
            tile[i + tile_rows][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
            tile[i + tile_rows][j + tile_cols] = field[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];

            mu[i - num_ghost][j - num_ghost] = bulk_chemical_potential(tile[i - num_ghost][j - num_ghost]);
            mu[i - num_ghost][j + tile_cols] = bulk_chemical_potential(tile[i - num_ghost][j + tile_cols]);
            mu[i + tile_rows][j - num_ghost] = bulk_chemical_potential(tile[i + tile_rows][j - num_ghost]);
            mu[i + tile_rows][j + tile_cols] = bulk_chemical_potential(tile[i + tile_rows][j + tile_cols]);
        }

        __syncthreads();

        // Surface terms involve derivatives of the field.

        Scalar lap = CentralDifference::laplacian(tile, i, j);
        mu[i][j] -= model.kappa * lap;
        mu[i][j] += model.lambda * CentralDifference::grad_squ(tile, i, j);

        constexpr int row_shift{tile_rows - 1}, col_shift{tile_cols - 1};
        constexpr int min_index = num_ghost - 1; // need one fewer for this higher derivative

        if (threadIdx.y < num_ghost and threadIdx.y >= min_index)
        {
            mu[i - num_ghost][j] -= model.kappa * CentralDifference::laplacian(tile, i - num_ghost, j);
            mu[i + row_shift][j] -= model.kappa * CentralDifference::laplacian(tile, i + row_shift, j);

            mu[i - num_ghost][j] += model.lambda * CentralDifference::grad_squ(tile, i - num_ghost, j);
            mu[i + row_shift][j] += model.lambda * CentralDifference::grad_squ(tile, i + row_shift, j);
        }

        if (threadIdx.x < num_ghost and threadIdx.x >= min_index)
        {
            mu[i][j - num_ghost] -= model.kappa * CentralDifference::laplacian(tile, i, j - num_ghost);
            mu[i][j + col_shift] -= model.kappa * CentralDifference::laplacian(tile, i, j + col_shift);

            mu[i][j - num_ghost] += model.lambda * CentralDifference::grad_squ(tile, i, j - num_ghost);
            mu[i][j + col_shift] += model.lambda * CentralDifference::grad_squ(tile, i, j + col_shift);
        }

        if (threadIdx.y < num_ghost and threadIdx.y >= min_index and threadIdx.x < num_ghost and threadIdx.x >= min_index)
        {
            mu[i - num_ghost][j - num_ghost] -= model.kappa * CentralDifference::laplacian(tile, i - num_ghost, j - num_ghost);
            mu[i - num_ghost][j + col_shift] -= model.kappa * CentralDifference::laplacian(tile, i - num_ghost, j + col_shift);
            mu[i + row_shift][j - num_ghost] -= model.kappa * CentralDifference::laplacian(tile, i + row_shift, j - num_ghost);
            mu[i + row_shift][j + col_shift] -= model.kappa * CentralDifference::laplacian(tile, i + row_shift, j + col_shift);

            mu[i - num_ghost][j - num_ghost] += model.lambda * CentralDifference::grad_squ(tile, i - num_ghost, j - num_ghost);
            mu[i - num_ghost][j + col_shift] += model.lambda * CentralDifference::grad_squ(tile, i - num_ghost, j + col_shift);
            mu[i + row_shift][j - num_ghost] += model.lambda * CentralDifference::grad_squ(tile, i + row_shift, j - num_ghost);
            mu[i + row_shift][j + col_shift] += model.lambda * CentralDifference::grad_squ(tile, i + row_shift, j + col_shift);
        }

        __syncthreads();

        current[0][index] = -CentralDifference::grad_y(mu, i, j);
        current[1][index] = -CentralDifference::grad_x(mu, i, j);

        current[0][index] += model.zeta * lap * CentralDifference::grad_y(tile, i, j);
        current[1][index] += model.zeta * lap * CentralDifference::grad_x(tile, i, j);
    }

    __global__ void step(DeviceField field, DeviceCurrent current)
    {
        constexpr int num_ghost = 2;

        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        // Local indices.
        const int i = threadIdx.y + num_ghost;
        const int j = threadIdx.x + num_ghost;

        // Load current tile into shared memory.

        __shared__ Scalar tile[d][tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];

        for (int c = 0; c < d; ++c)
        {
            tile[c][i][j] = current[c][index];

            // Fill in ghost points.

            if (threadIdx.y < num_ghost)
            {
                tile[c][i - num_ghost][j] = current[c][col + ((row - num_ghost + nrows) % nrows) * ncols];
                tile[c][i + tile_rows][j] = current[c][col + ((row + tile_rows) % nrows) * ncols];
            }

            if (threadIdx.x < num_ghost)
            {
                tile[c][i][j - num_ghost] = current[c][(col - num_ghost + ncols) % ncols + row * ncols];
                tile[c][i][j + tile_cols] = current[c][(col + tile_cols) % ncols         + row * ncols];
            }

            if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
            {
                tile[c][i - num_ghost][j - num_ghost] = current[c][(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
                tile[c][i - num_ghost][j + tile_cols] = current[c][(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
                tile[c][i + tile_rows][j - num_ghost] = current[c][(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
                tile[c][i + tile_rows][j + tile_cols] = current[c][(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];
            }
        }

        __syncthreads();

        // Integration rule from continuity equation $\partial_t \phi = -\nabla \cdot \vec{J}$:
        Scalar divJ = CentralDifference::grad_y(tile[0], i, j)
                    + CentralDifference::grad_x(tile[1], i, j);
        field[index] -= stencil.dt * divJ;
    }

    // Basic kernel to check for errors (e.g. if field become nan or inf).
    __global__ void check_finite(DeviceField field, bool* finite)
    {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;
        if (not std::isfinite(field[index])) *finite = false;
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

void Integrator::run(int nsteps [[maybe_unused]])
{
    set_device_parameters();
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);


    for (int i = 0; i < nsteps; ++i)
    {
        kernel::calculate_current<<<grid_size, block_dim>>>(field, current);
        kernel::step<<<grid_size, block_dim>>>(field, current);
    }

    cudaDeviceSynchronize();
    kernel::throw_errors();

    // Numerical errors in integration often cause fields to diverge or go to nan, so we
    // need to check for these on the device and raise them up the stack.
    bool finite{true}, *device_finite;
    cudaMalloc(&device_finite, sizeof(bool));
    cudaMemcpy(device_finite, &finite, sizeof(bool), cudaMemcpyHostToDevice);
    kernel::check_finite<<<grid_size, block_dim>>>(field, device_finite);
    cudaMemcpy(&finite, device_finite, sizeof(bool), cudaMemcpyDeviceToHost);

    if (not finite)
    {
        std::string message = "an unknown numerical error occurred during simulation";
        throw kernel::CudaError(message);
    }

    timestep += nsteps;
}
